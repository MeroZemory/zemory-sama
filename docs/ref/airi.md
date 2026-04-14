# AIRI (moeru-ai) 대화 파이프라인 분석

> GitHub: https://github.com/moeru-ai/airi (38k stars, v0.9.0-beta.2)
> 분석일: 2026-04-15

## 아키텍처 개요

Vue 3 + TypeScript 모노레포(pnpm workspace). 웹/Electron(Tamagotchi)/모바일(Capacitor) 다중 런타임을 apps/에 두고, 음성·추론·렌더링 로직은 packages/에 분리, 플랫폼 연동(Discord/Telegram/Minecraft/Factorio)은 services/에 배치한다. 백엔드(`apps/server`)는 Hono + PostgreSQL + Redis + Drizzle 기반 LLM 게이트웨이 + 멀티인스턴스 브로드캐스트를 담당한다.

구성 요약:

- **Frontend**: Vue 3, Vite, Pinia, VueUse, UnoCSS
- **Graphics**: Three.js (VRM, 운영 수준) · Live2D(`packages/stage-ui-live2d/`, WIP)
- **Audio**: Web Audio API + AudioWorklet + Silero VAD (브라우저) + Kokoro TTS (WASM/WebGPU) + Whisper (transformers.js 로컬 또는 원격)
- **LLM 라우팅**: xsai 어댑터 (40+ providers)
- **Desktop**: Electron (Tauri 아님, AGENTS.md:18 명시)
- **관측**: OpenTelemetry 트레이스/메트릭/로그

---

## 1. VAD (브라우저 Silero VAD)

ONNX Silero 모델을 `@huggingface/transformers`로 로드해 Web Worker에서 추론. AudioWorklet이 16 kHz·512-sample chunk(~32 ms)로 프레임을 공급한다.

**주요 파일**:
- `packages/stage-ui/src/workers/vad/vad.ts` — VAD 클래스, HF 로드
- `packages/stage-ui/src/workers/vad/process.worklet.ts` — AudioWorklet 프로세서
- `packages/stage-ui/src/libs/audio/vad.ts` — BaseVAD 인터페이스, Web Audio 라우팅
- `packages/stage-ui/src/workers/vad/manager.ts` — 메시지 오케스트레이션

**파라미터** (`vad.ts:24-37`):

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `speechThreshold` | 0.3 | 음성 확률 임계 |
| `exitThreshold` | 0.1 | 이탈 임계 |
| `minSilenceDurationMs` | 400 | 종료 침묵 |
| `speechPadMs` | 80 | 앞뒤 패딩 |
| `minSpeechDurationMs` | 250 | 최소 발화 길이 |
| `maxBufferDuration` | 30 s | 버퍼 상한 |
| `newBufferSize` | 512 | ~32 ms / 프레임 |

이벤트: `speech-start`, `speech-end`, `speech-ready`, `status`, `debug`.

**오디오 그래프**: `MediaStreamSource → AudioWorklet → GainNode(gain=0) → destination`. gain=0을 두어 그래프 활성 상태를 유지하면서도 스피커 재귀 루프를 차단한다.

---

## 2. STT (Whisper / Web Speech / Aliyun NLS)

`@xsai/generate-transcription` 추상화 위에 다중 어댑터.

- **브라우저**: Web Speech API (fallback)
- **원격**: OpenAI Whisper / OpenAI-호환 엔드포인트
- **로컬**: transformers.js 기반 Whisper.cpp ONNX (Web Worker)
- **실시간 스트리밍**: Aliyun NLS (`stores/providers/aliyun/stream-transcription.ts`)

핵심 파일:
- `packages/stage-ui/src/stores/modules/hearing.ts` — VAD ↔ 전사 오케스트레이션, confidence 필터
- `packages/stage-ui/src/libs/inference/adapters/whisper.ts`

`hearing.ts:197-199`에 VAD 연동 세션 리스타트 로직이 TODO로 남아 있어 현재는 VAD 이벤트가 ASR 세션을 자동 재시작하지 않는다.

---

## 3. LLM 라우팅 (xsai)

`packages/stage-ui/src/stores/providers.ts` 가 통합 팩토리. `createModelProvider()`, `createSpeechProvider()`, `createTranscriptionProvider()` 로 동일 인터페이스 아래 40+ 공급자를 노출한다.

지원 목록(README:328-349): OpenAI/Azure, Anthropic, Groq, DeepSeek, Qwen, Gemini, Grok, Mistral, vLLM, Ollama, SGLang, OpenRouter, Cloudflare Workers AI, Together.ai, Fireworks 등. 각 provider는 `capabilities` (listModels/listVoices/loadModel)를 선언, `validators.validateProviderConfig()` 가 토큰 낭비 방지용 옵트인 ping 체크를 수행한다.

**전이중(full-duplex) 네이티브 스트리밍은 미지원** — 턴 관리는 애플리케이션 계층에서 동기 request/response.

---

## 4. TTS (Kokoro / ElevenLabs / 기타)

### Kokoro (로컬, 운영 가능)
- `packages/stage-ui/src/workers/kokoro/` — `kokoro-js` npm 패키지 + Web Worker
- 양자화: fp32 / fp16 / fp16-webgpu
- Float32 PCM 스트림을 Web Audio로 바로 재생

### ElevenLabs (원격)
- `packages/stage-ui/src/stores/providers/elevenlabs/`
- SSML 지원 (`speech.ts:76-82`), 모델/보이스 목록 조회

### 기타
- OpenRouter OpenAI-호환 speech endpoint (`providers/openrouter/audio-speech.ts`)
- Browser Web Speech API (fallback)

`stores/modules/speech.ts` 에서 `activeSpeechProvider` / `activeSpeechVoiceId` / pitch / rate / SSML 여부를 Pinia 스토어로 영속화(localStorage). TTS 합성을 ASR 입력으로 되돌리지 않는 **구조적 half-duplex**.

---

## 5. 에코·턴테이킹 전략

- **에코 방지**: Web Worker 격리 + 마이크/스피커 MediaStream 분리 + 브라우저/OS 레벨 AEC. 서버측 바지인(barge-in) 인터럽트 없음.
- **턴 종료**: Silero 상태 머신(3 연속 hit → ACTIVE, 24 연속 miss → IDLE) 위에서 `speech-ready` emit.
- **인터럽트 미지원**: 사용자가 AI 발화 도중 끼어들어도 TTS가 중단되지 않는다(설계상). 자연스러운 오버랩보다 **스크립트형 순차 QA** 에 적합.

---

## 6. 아바타 렌더링

### VRM — 운영 수준
- `packages/stage-ui-three/` + `@pixiv/three-vrm`
- `utils/vrm-preview.ts:41-80` 로딩/애니메이션/머티리얼 업데이트
- MToon 셰이더, auto-blink, auto-look-at, idle-eye (README:242-245)

### Live2D — **미완**
- `packages/stage-ui-live2d/` 패키지는 존재하나 구현 최소. README 주장은 대부분 aspirational, 운영 불가.

감정 시스템: `packages/stage-ui/src/constants/emotions.ts` 가 LLM 응답 분류 → 표정 블렌딩.

---

## 7. 메모리 / RAG

- **pgvector 모듈** (`packages/memory-pgvector/src/index.ts`) 가 `@proj-airi/server-sdk` Client 로 `module:configure` 이벤트를 수신하는 인프라까지는 구현됨.
- 브라우저 옵션: DuckDB WASM, pglite
- **하지만 채팅 루프에 자동 RAG 회수가 통합되어 있지 않음**. pgvector 배선은 있고 검색 로직은 비어 있다. 컨텍스트 관리는 현재 LLM provider 토큰 한도에만 의존.

---

## 8. 스트리밍·게임 연동 (실제 vs 광고)

| 통합 | 상태 | 경로 |
|------|------|------|
| Discord (VC + 채팅) | ✅ 운영 | `apps/server` WebSocket + `services/discord-bot` |
| Telegram | ✅ 운영 | `services/telegram-bot` |
| Minecraft (Mineflayer) | ⚠️ 동작하나 Fabric mod 로 이관 예정 | `services/minecraft/` 4-layer cognitive stack (Perception/Reflex/Conscious/Action) |
| Factorio | 🧪 PoC | `moeru-ai/airi-factorio` 별도 저장소 |
| Twitch/YouTube | ❌ 미구현 | README 에만 언급 |
| VRChat | ❌ 미구현 | |

Minecraft 서비스의 **4-layer cognitive 아키텍처**(perception → reflex → conscious → action) 는 다른 도메인 에이전트로 이식 가능한 패턴.

---

## 9. 서버/멀티인스턴스

- `apps/server`: Hono + Better Auth (OIDC) + Stripe (billing) + Drizzle ORM
- Redis Pub/Sub 로 WebSocket 이벤트를 크로스 인스턴스 브로드캐스트 → 수평 확장 가능한 stateless backend
- `packages/server-sdk/src/client.ts` 에 자동 재연결 WebSocket 클라이언트

---

## 10. zemory-sama 가 차용할만한 패턴

1. **xsai 스타일 provider 추상** — 향후 Realtime API ↔ ElevenLabs ↔ 로컬 스왑을 코드 변경 없이 지원
2. **Web Worker / AudioWorklet 격리** — 메인 루프 블로킹 제거 (현재 zemory_vad 는 Python asyncio 이므로 동등 패턴 = 별도 프로세스 또는 실행자)
3. **통합 inference protocol** (`libs/inference/protocol.ts`) — Whisper/Kokoro/LLM 에 공통 메시지 타입
4. **Minecraft 4-layer cognitive stack** — 추후 VTuber 자율 행동 확장 시 참고
5. **module:configure 이벤트로 서비스 구성을 런타임 주입** — 설정 핫리로드 기반 형태

## 11. 성숙도 매트릭스

| 구성 | 상태 |
|------|------|
| VAD / STT / LLM 라우팅 / Kokoro·ElevenLabs TTS | 운영 |
| VRM 렌더 | 운영 |
| Discord 통합 | 운영 |
| Minecraft | 유지보수 (Fabric 이관 중) |
| Live2D | 패키지뿐, 미완 |
| 메모리/RAG | 배선만 존재 |
| Twitch/YouTube | 코드 없음 |
| Full-duplex / barge-in | 설계상 미지원 |

---

## 레퍼런스 코드 위치

```
_ref/airi/
├── apps/
│   ├── stage-web/              # Vue 3 PWA 프런트
│   ├── stage-tamagotchi/       # Electron 데스크톱
│   ├── stage-pocket/           # Capacitor 모바일
│   └── server/                 # Hono + pg + Redis 백엔드
├── packages/
│   ├── stage-ui/               # 오디오·스토어·워커 (핵심)
│   │   ├── src/workers/vad/    # Silero VAD
│   │   ├── src/workers/kokoro/ # Kokoro TTS
│   │   ├── src/libs/inference/ # 통합 추론 프로토콜
│   │   └── src/stores/         # Pinia: hearing, speech, providers
│   ├── stage-ui-three/         # VRM 렌더 (운영)
│   ├── stage-ui-live2d/        # Live2D (WIP)
│   ├── memory-pgvector/        # 메모리 모듈 (배선)
│   └── server-sdk/             # WebSocket 클라이언트
└── services/
    ├── discord-bot/
    ├── telegram-bot/
    └── minecraft/              # 4-layer cognitive stack
```
