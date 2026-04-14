# 레퍼런스 프로젝트 비교 및 zemory-sama 적용 방향

> 분석일: 2026-04-15 (2026-04-13 초판 → 7개 레퍼런스로 확장)

## 1. 레퍼런스 한눈에

| # | 프로젝트 | 스타 | 스택 | 포지셔닝 |
|---|---------|-----|------|---------|
| 1 | [Open-LLM-VTuber](./open-llm-vtuber.md) | 6.8k | Python + FastAPI + Vue | 웹 기반 Live2D, 교과서적 VAD 상태머신 |
| 2 | [RealtimeVoiceChat](./realtime-voice-chat.md) | - | Python + FastAPI + WebSocket | 저지연 full-duplex, ML 턴 감지 |
| 3 | [airi](./airi.md) | 38k | Vue 3 모노레포 + Electron | 최대 활성도, xsai 라우팅 + Kokoro 로컬 TTS |
| 4 | [Neuro](./neuro.md) | 1.9k | Python 로컬 | Chroma reflection 장기기억, Patience 자율발화 |
| 5 | [AI-Waifu-Vtuber](./ai-waifu-vtuber.md) | 1.1k | Python 단일 스크립트 | Twitch/YouTube 채팅 중심, VAD 부재 |
| 6 | [AIRIS-VtuberAI](./airis-vtuber-ai.md) | 145 | Python 로컬 (transformers) | 문장 단위 조기 TTS, 완전 로컬 추론 |
| 7 | [awesome-ai-vtubers](./awesome-ai-vtubers.md) | 393 | 리스트 | 추가 발굴 카탈로그 |

---

## 2. 핵심 패턴 비교 (파이프라인)

| 항목 | OLV | RVC | airi | Neuro | AI-Waifu | AIRIS | zemory-sama |
|------|-----|-----|------|-------|----------|-------|-------------|
| **런타임** | 웹 + FastAPI | 웹 + FastAPI | Vue/Electron | Python 로컬 | Python 스크립트 | Python 로컬 | CLI asyncio |
| **VAD** | Silero 자체 구현 | Silero (RealtimeSTT) | Silero (브라우저 워커) | Silero (RealtimeSTT) | 없음 (PTT) | 없음 (PTT) | OpenAI server_vad + 로컬 Silero (`zemory_vad`) |
| **STT** | 다양 | Faster-Whisper | Whisper WASM / Aliyun / Web Speech | Faster-Whisper tiny.en | OpenAI Whisper API | Faster-Whisper `distil-large-v3` | OpenAI Realtime |
| **LLM** | 다양 | Ollama 로컬 | xsai 40+ providers | text-gen-webui (Llama3) | GPT-3.5-turbo | HF transformers 로컬 | OpenAI Realtime |
| **TTS** | 다양 | RealtimeTTS | Kokoro WASM / ElevenLabs | Coqui XTTS-v2 | VoiceVox / Silero | OpenVoice | ElevenLabs Flash v2.5 |
| **에코 방지** | 프런트/백 분리 | `interrupted` 플래그 | Worker 격리 + 브라우저 AEC | 장치 분리 | N/A (chat) | N/A (PTT) | `speaking` 이벤트 게이트 |
| **인터럽트** | `Task.cancel()` | abort 체인 | 없음 | **permission check 만** | 없음 | 없음 | half-duplex (비활성) |
| **TTS 순서보장** | 시퀀스 넘버 | Quick/Final 2단 | - | 단일 덩어리 | 단일 재생 | 문장 단위 조기 트리거 | 순차 큐 |
| **장기 메모리** | 없음 | 없음 | pgvector (미통합) | **ChromaDB + reflection** | 없음 | 최근 N턴 절단만 | 없음 |
| **자율 발화** | 없음 | 없음 | 없음 | **Patience 타임아웃** | 없음 | 없음 | 없음 |
| **Twitch/YouTube** | 부분 | 없음 | 코드 없음 | Twitch 모듈 | Raw IRC + pytchat | twitchio + pytchat | 없음 |
| **아바타** | Live2D | 없음 | VRM (운영), Live2D WIP | VtubeStudio pyvts | VB-Cable 루프백 | 없음 | 없음 |

**약어**: OLV=Open-LLM-VTuber, RVC=RealtimeVoiceChat.

---

## 3. 에코 방지 / 턴테이킹 스펙트럼

아키텍처적으로 엄격한 순서:

```
확실함 ◄──────────────────────────────────────────────────► 위험
  OLV       RVC        airi        zemory-sama    Neuro      AI-Waifu/AIRIS
  (웹 분리)  (플래그+  (Worker    (이벤트 게이트) (장치     (VAD 자체 없음,
            1초 타이머) 격리+AEC)                 분리만)    PTT 의존)
```

- **OLV**: 프런트/백 MediaStream 물리 분리 + 브라우저 AEC + 프런트 "playback-complete" 왕복
- **RVC**: `interrupted` 플래그 + 첫 TTS 청크 후 1초 타이머
- **airi**: Worker·MediaStream 격리 + 브라우저/OS AEC. 단 **바지인 없음**
- **zemory-sama**: `speaking` asyncio.Event + `input_audio_buffer.clear()` 이중 클리어 + `speaker.wait_until_done()`
- **Neuro**: 인터럽트 permission check 만 — 실질적 바지인 불가
- **AI-Waifu / AIRIS**: VAD 없이 push-to-talk 만 — 에코 문제 자체가 회피됨

---

## 4. 주요 패턴 발췌 (신규 레퍼런스에서)

### airi — xsai provider 추상화
40+ LLM / TTS / STT provider 를 `createModelProvider()` / `createSpeechProvider()` / `createTranscriptionProvider()` 팩토리 한 겹으로 감싸 코드 변경 없이 스왑. zemory-sama 가 향후 Realtime API ↔ 로컬 모델 ↔ ElevenLabs 간 스왑을 지원할 때 직접 차용 가능한 설계.

### airi — 통합 Inference Protocol
`packages/stage-ui/src/libs/inference/protocol.ts` — Whisper/Kokoro/LLM 에 공통 메시지 타입. 컴포넌트가 추론 엔진 구현을 몰라도 동작.

### Neuro — Reflection 기반 장기 기억
20+ 신규 메시지마다 LLM 에 "가장 중요한 Q&A 3쌍" 을 요청 → `{qa}` 구분자 분리 → ChromaDB 저장. 질의는 최근 5 메시지 → top-5 회수 → priority 60 으로 프롬프트 인젝션. **zemory-sama 가 장기 기억 추가 시 거의 그대로 이식 가능**.

### Neuro — Priority 기반 Prompt Injection
```
System(10) → History(50) → Memory(60) → Twitch(150) → Custom(200)
```
우선순위 오름차순 정렬 후 concat → 높은 priority 가 프롬프트 **끝**(최근 주의). 모듈 추가 시 숫자 하나만 지정.

### Neuro — Patience 자율 발화
`PATIENCE=60s` 동안 사용자 입력/Twitch 메시지 없으면 prompter 가 LLM 을 자율 호출. 진행률을 `patience_update` 이벤트로 50 ms 간격 UI 송출. **방송형 콘텐츠로 확장 시 필수 장치**.

### AIRIS — 문장 단위 조기 TTS 트리거
LLM 토큰 스트림을 누적하다 구두점(`['?','!','.',':']`) 발견 시 **즉시** TTS 합성을 트리거. 전체 응답 완료 대기 없이 첫 문장부터 재생 시작 → 체감 지연 대폭 감소. zemory-sama 순차 큐에 **1일 내 구현 가능한 저렴한 개선**.

### AIRIS / AI-Waifu — 시스템 프롬프트 파일 분리
constants.py 대신 `system_message.txt`, `characterConfig/<name>/identity.txt` 로 핫스왑 가능. 다중 페르소나 지원 시 기반.

### AI-Waifu — Raw IRC 소켓 Twitch
라이브러리 없이 `socket.socket()` 직접으로 Twitch IRC 수신. 의존성 최소 + 디버깅 쉬움. 추후 한국어 방송 확장 시 가장 간단한 진입점.

---

## 5. zemory-sama 적용 로드맵 (갱신)

### Tier A — 저렴하고 즉시 적용 (≤ 1일)

| 개선 | 출처 | 구현 난이도 |
|------|------|-----------|
| **문장 단위 조기 TTS 트리거** | AIRIS | 낮음 — 토큰 버퍼에 구두점 감지 추가 |
| `speech_stopped` + 마이크 해제 시 `input_audio_buffer.clear()` 이중 호출 | zemory-sama 자체 (기존) | 낮음 |
| `speaker.wait_until_done()` 후 0.5-1 s 안전 딜레이 | RVC | 낮음 |
| 시스템 프롬프트 파일 분리 (`prompts/*.txt`) | AIRIS / AI-Waifu | 낮음 |

### Tier B — 중규모 (≤ 1주)

| 개선 | 출처 | 구현 난이도 |
|------|------|-----------|
| **TTS 병렬 생성 + 순서 보장** (TTSTaskManager 패턴) | OLV | 중간 |
| **Quick Answer 2단 청크** (첫 문장 작은 청크) | RVC | 중간 |
| **Priority 기반 Prompt Injection** | Neuro | 중간 |
| Provider 추상화 (Realtime ↔ ElevenLabs ↔ 로컬 스왑) | airi `xsai` | 중간 |

### Tier C — 대규모 아키텍처 변경

| 개선 | 출처 | 구현 난이도 |
|------|------|-----------|
| **ChromaDB + reflection 장기 메모리** | Neuro | 중간~높음 |
| **Patience 자율 발화 루프** (방송형) | Neuro | 중간 |
| **Twitch/YouTube 채팅 수신** | AI-Waifu (IRC) / AIRIS | 중간 |
| **로컬 Silero VAD 전환** (이미 `zemory_vad/` 로 시작) | OLV / airi / Neuro | 높음 (진행 중) |
| **웹 UI 전환** (VRM/Live2D) | OLV / airi (VRM) | 높음 |

---

## 6. 아키텍처 결정 매트릭스

**즉시 답이 나오는 트레이드오프:**

| 질문 | 권고 | 근거 |
|------|------|------|
| 로컬 vs 클라우드 LLM? | 클라우드(Realtime) 유지, 추상화 뒤에 로컬 옵션 | Neuro/AIRIS 로컬은 인터럽트·저지연·품질 모두 손해 |
| Live2D vs VRM? | 결정 지연. airi VRM 은 운영, Live2D 는 어디나 WIP | 렌더 갭이 크지 않고 zemory-sama 현재는 UI 필요성 낮음 |
| TTS 엔진? | ElevenLabs Flash v2.5 유지 + Kokoro 로컬 fallback | Kokoro 는 WASM/WebGPU 수준이라 오프라인 대비에 적합 |
| 메모리 저장? | ChromaDB (Neuro 방식) | Postgres/pgvector 는 서버 요구 시에만 |
| 방송 수신? | IRC 소켓부터(AI-Waifu), 안정되면 twitchio | 의존성 최소화 우선 |

---

## 7. 레퍼런스별 깊이 안내

자세한 내용은 각 문서 참조:

- [open-llm-vtuber.md](./open-llm-vtuber.md) — VAD 상태머신, TTSTaskManager, 시그널 프로토콜
- [realtime-voice-chat.md](./realtime-voice-chat.md) — Quick/Final, ML 턴 감지
- [airi.md](./airi.md) — 모노레포 구조, Kokoro 워커, VRM 렌더
- [neuro.md](./neuro.md) — Signals 허브, Chroma reflection, priority injection
- [ai-waifu-vtuber.md](./ai-waifu-vtuber.md) — Raw IRC, pytchat, VB-Cable 립싱크
- [airis-vtuber-ai.md](./airis-vtuber-ai.md) — 문장 단위 TTS, 로컬 추론
- [awesome-ai-vtubers.md](./awesome-ai-vtubers.md) — 추가 클론 후보 카탈로그
