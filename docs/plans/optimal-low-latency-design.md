# zemory-sama 저지연 실시간 채팅 최적 설계

> 작성일: 2026-04-15
> 7개 레퍼런스(`docs/ref/`) 의 장점을 통합하고 단점을 제거한 단일 아키텍처 설계.

## 1. Context & Goals

### 현재 상태

- **Phase 1 `zemory/`** (운영): OpenAI Realtime API + ElevenLabs Flash v2.5. 단일 `speaking: asyncio.Event` half-duplex. 저지연이지만 LLM 고정·메모리 없음·인터럽트 없음.
- **Phase 2 `zemory_vad/`** (WIP 95%): 로컬 Silero VAD + Whisper STT + 3-phase state machine + 백그라운드 DuckDuckGo 검색. 유연하지만 STT 왕복 ~300-500 ms 추가, 여전히 half-duplex.

두 경로가 충분히 갈라져 있어 수정이 상호 반영되지 않는다(예: `SentenceChunker` 가 두 `tts.py` 에 중복). 본 설계는 두 경로를 **단일 `zemory/` 패키지 + 런타임 프로파일 선택** 으로 통합하고, 레퍼런스에서 식별된 모든 갭을 메운다.

### 1차 목표

**VAD `speech_end` (Realtime 의 `speech_stopped`) 로부터 첫 오디오 출력까지 p50 ≤ 800 ms.**

### 지연 예산 (Realtime 프로파일 800 ms p50)

| 단계 | 예산 | 비고 |
|------|------|------|
| VAD 침묵 확정 | t=0 이전 (24 × 32 ms = 768 ms 이미 소비) | 예산 외 |
| STT (로컬 프로파일만) | 200 ms | `gpt-4o-transcribe` 비스트리밍 |
| LLM first-token | 250 ms | Realtime `response.create` → 첫 `response.text.delta` |
| 문장 트리거 대기 | 30 ms | AIRIS: 첫 `.?!,:？！。` 또는 40자 윈도우 |
| TTS first-chunk | 250 ms | ElevenLabs Flash v2.5 `pcm_24000` TTFB |
| 오디오 큐 → 스피커 버퍼 | 70 ms | `httpx.aiter_bytes(4096)` → queue → `_buffer` |
| **Realtime 합계** | **~600 ms** | |
| **Local 합계** | **~800 ms** | |

### 성공 지표

1. p50 ≤ 800 ms, p95 ≤ 1200 ms (첫 오디오 출력).
2. Barge-in: AI 발화 중 사용자 음성 → 스피커 무음 ≤ **150 ms**.
3. 프로파일 스왑이 config 한 줄.
4. ChromaDB top-k 회수 ≤ 50 ms.
5. p99 무크래시 단일 세션 ≥ 30분.

## 2. Target Architecture

### 단일 패키지, 이중 프로파일

```
┌───────────────────────────────────────────────────────────────────┐
│                     zemory.orchestrator.run()                     │
│                                                                   │
│   mic (sd) ──PCM24k──┐                                            │
│                      ▼                                            │
│         TurnDetector (Protocol)                                   │
│           • ServerVADTurnDetector  (Realtime profile)             │
│           • SileroTurnDetector     (Local profile)                │
│                      │ speech_end                                 │
│                      ▼                                            │
│         STT (Protocol, optional)                                  │
│           • NullSTT                (Realtime inline)              │
│           • WhisperSTT                                            │
│                      │ text                                       │
│                      ▼                                            │
│         PromptAssembler (Neuro priority)                          │
│           system=10  history=50  memory=60                        │
│           chat=150  patience=180  custom=200                      │
│                      ▼                                            │
│         LLMProvider (Protocol)                                    │
│           • OpenAIRealtimeLLM   • OpenAIChatLLM                   │
│                      │ token stream                               │
│                      ▼                                            │
│         SentenceChunker + TTSTaskManager                          │
│           (AIRIS early trigger + OLV seq-ordered parallel)        │
│                      │ sequenced audio                            │
│                      ▼                                            │
│         TTSProvider (Protocol)                                    │
│           • ElevenLabsTTS   • KokoroTTS (stub)                    │
│                      ▼                                            │
│                SpeakerStream (24k PCM)                            │
│                                                                   │
│   Side channels:                                                  │
│     • MemoryStore (ChromaDB) — reflection every 20 msgs           │
│     • ChatSources (Twitch IRC, YouTube pytchat) — prio 150        │
│     • Patience — 60 s idle → autonomous trigger                   │
│     • InterruptBus — VAD speech_start → abort chain               │
└───────────────────────────────────────────────────────────────────┘
```

## 3. 지연 최적화 결정

### 3.1 기본 프로파일 = Realtime + server_vad

`turn_detection=server_vad`, `modalities=["text"]`, ElevenLabs 는 audio. **한국어 튜닝**: `silence_duration_ms: 500 → 700` (`SESSION_CONFIG`), `VAD_REQUIRED_MISSES: 24 → 28` (= 896 ms, 로컬 프로파일).

**RVC DistilBERT 턴 감지는 거부**: 70 MB 영어 튜닝 모델, 40-80 ms 추론 비용, 한국어 어미 오작동.

### 3.2 문장 단위 조기 TTS 트리거 (AIRIS, 기존 유지 + 강화)

`SentenceChunker` (현 `zemory/tts.py:20-45`) 의 `.?!。？！\n` 경계 유지. 추가: `,:;、，` **soft** boundary — 버퍼 ≥ 40 자일 때만 발화. 긴 절에서 첫 음성이 더 빨라지면서 짧은 어구 중간에 끊기지 않음.

### 3.3 TTSTaskManager (OLV, 현 sequential 교체)

현 `tts_worker` (`zemory/client.py:82-108`) 는 HTTP 스트림을 직렬 소비. 교체안:

- 각 `chunker.add()` 결과마다 `asyncio.create_task(synthesize(sentence, seq))`.
- 디스패처 코루틴이 `buffered_payloads: dict[int, bytes]` 를 유지 → seq 순서대로만 `SpeakerStream.queue` 로 포워드.
- 동시성 cap: `semaphore = asyncio.Semaphore(3)`. ElevenLabs rate-limit 준수 + 태스크 무한 증가 방지.
- 4-문장 응답 기준 전체 wall time 1-2 초 절감.

### 3.4 RVC Quick/Final (ElevenLabs 파라미터로 재현)

- 첫 문장: `optimize_streaming_latency=4` (빠름)
- 이후 문장: `optimize_streaming_latency=2` (품질)
- **별도 Quick LLM 패스 거부** — Realtime 이미 토큰 스트리밍이라 "첫 문장" 자체가 Quick.

### 3.5 Worker 격리 (airi → Python 번역)

- Silero VAD 추론 ~3 ms CPU: **동기 유지**, 스레드 오버헤드 회피.
- Whisper STT API: 이미 `httpx` async. 유지.
- DuckDuckGo `DDGS`: 동기 → `asyncio.to_thread` 래핑 유지.
- 향후 로컬 Kokoro/Whisper: `ProcessPoolExecutor(max_workers=2)` 공유. 스레드 말고 프로세스 (PyTorch GIL 해제 비일관성).

### 3.6 SpeakerStream (변경 없음)

`zemory/audio.py:56-67` 의 triple-buffered callback + asyncio.Queue + `_buffer` 는 그대로. `first_write_at` 타임스탬프 훅만 추가 (§8).

## 4. 인터럽트 & Barge-in 체인

**RVC granular abort 채택**. OLV 의 단일 `task.cancel()` 은 거칠어 WebSocket 까지 취소되고 재연결 비용 발생.

### 트리거

`InterruptBus.trigger(reason)` 호출 지점은 2곳:
1. Realtime 프로파일: `input_audio_buffer.speech_started` 가 `Phase != LISTENING` 일 때.
2. Local 프로파일: `VADStateMachine.process()` 가 `"speech_start"` 반환하고 `Phase != LISTENING` 일 때.

### Abort 시퀀스 (≤ 150 ms 완료)

```
InterruptBus.trigger("user_barge_in")
 ├─[1] SpeakerStream.clear()              # ~0 ms
 ├─[2] TTSTaskManager.abort()             # abort_flag + task cancel, ~5 ms
 ├─[3] LLMProvider.cancel_current()       # conn.response.cancel(), ~20-50 ms
 ├─[4] record_partial_assistant()         # {role=assistant, interrupted=true} 로 history 기록 (OLV 패턴)
 ├─[5] phase = Phase.ACTIVE
 └─[6] input_audio_buffer.clear() + mic unmute
```

### 가치

- Neuro: 취소 없었음 → 해결.
- OLV: 단일 Task.cancel → 커넥션 재연결. 본 설계는 LLM 연결 유지 + TTS 만 취소.
- RVC 패턴 복사 + [4] OLV 의 partial-response history 추가.

### 디바운스

- 마지막 트리거 < 250 ms 이면 무시.
- RESPONDING 단계에서 VAD 연속 2 hit 필요 (단일 hit 는 AI self-audio 오인 가능).

## 5. 메모리 & 프롬프트 인젝션

### ChromaDB 스키마

- 컬렉션: `zemory_{character_id}_longterm`
- 임베딩: `all-MiniLM-L6-v2` 기본. 한국어 품질 이슈 시 `dragonkue/multilingual-e5-small-ko` 로 config 스왑.
- 문서 스키마:
  ```
  {id: UUID, text: "Q: ... A: ...",
   metadata: {turn_ids, timestamp, source: "reflection"|"bootstrap", importance: 1-10}}
  ```

### Reflection

- 어시스턴트 턴마다 카운터 증가.
- 20턴마다 백그라운드 태스크 (inline 아님 — Neuro 의 async 안 blocking 버그 회피):
  ```python
  asyncio.create_task(reflect_and_store(recent_20_turns))
  ```
- 프롬프트: "대화에서 가장 중요한 Q&A 3쌍을 추출하여 `{q, a, importance}` JSON 리스트로 출력하라."
- 각 Q&A 를 독립 Chroma 문서로 저장.

### 회수

- 사용자 턴 STT/전사 직후 `collection.query(query_texts=[last_user_text], n_results=5)` 를 LLM 요청과 **병렬**로 실행.
- 회수 ~20-50 ms, LLM first-token ~250 ms → 메모리 주입은 `conversation.item.create` 로 `response.create` 이후에 (단, LLM 이 >200자 생성 전까지만). 이후면 다음 턴으로 연기.
- **핵심**: hot path 를 메모리 회수로 블로킹하지 않는다.

### Priority (Neuro 그대로)

| Source | Priority | 위치 |
|--------|----------|------|
| System / persona | 10 | 프롬프트 시작 |
| Conversation history | 50 | 중간 |
| Memory (RAG top-5) | 60 | 중간 |
| Twitch chat buffer | 150 | 후반 |
| Patience autonomous | 180 | 후반 |
| Custom user override | 200 | 끝 |

높은 priority = 프롬프트 끝 근처 (LLM recency bias). `PromptAssembler.register_injection(source_id, priority, callable)` 인터페이스.

### Patience

- 타이머는 `Phase.LISTENING` 진입 시 리셋, 전이 시 취소.
- 60 s 유휴에서 priority-180 인젝션: `"1분간 아무도 말하지 않았다. 캐릭터를 유지하며 자연스럽게 말해라 — 침묵을 언급하거나 방송을 코멘트하거나 열린 질문을 던져라. 2문장 이내."`
- `PromptAssembler.has_trigger()` True → 오케스트레이터가 `LLMProvider.generate_turn(user_input=None, injections=...)`.

## 6. Provider 추상화

Python Protocol (airi xsai 영향).

```python
class TurnDetector(Protocol):
    async def feed(self, pcm24k: bytes) -> None: ...
    events: asyncio.Queue  # Literal["speech_start","speech_end"]

class STTProvider(Protocol):
    async def transcribe(self, pcm_chunks: list[bytes]) -> str: ...
    async def transcribe_stream(self, audio_gen): ...  # optional

class LLMProvider(Protocol):
    async def open_session(self, system_prompt, tools): ...
    async def send_user_text(self, text, injections): ...
    async def stream_response(self) -> AsyncIterator[str]: ...
    async def cancel_current(self) -> None: ...
    async def close(self) -> None: ...

class TTSProvider(Protocol):
    async def synthesize(self, text, seq, quick) -> AsyncIterator[bytes]: ...
```

### 동봉 어댑터

| Protocol | 구현 |
|----------|------|
| TurnDetector | `server_vad.py`, `silero.py` |
| STT | `null.py` (Realtime inline), `openai_whisper.py` |
| LLM | `openai_realtime.py`, `openai_chat.py` |
| TTS | `elevenlabs.py`, `kokoro.py` (스텁) |

### Profile config

```toml
# config.toml
[profile]
name = "realtime"   # or "local"

[profiles.realtime]
turn_detector = "server_vad"
stt = "null"
llm = "openai_realtime"
tts = "elevenlabs"

[profiles.local]
turn_detector = "silero"
stt = "openai_whisper"
llm = "openai_realtime"
tts = "elevenlabs"
```

오케스트레이터는 profile → registry 에서 4개 provider 인스턴스화 → 와이어링.

## 7. 통합 State Machine

Phase-1 `speaking: Event` + Phase-2 `Phase.{LISTENING,ACTIVE,RESPONDING}` 을 단일 enum 으로 통합.

```
        ┌─────────────────────┐
        │                     │ (TTS done + SAFETY_DELAY + speaker empty)
        ▼                     │
  ┌──────────────┐            │
  │  LISTENING   │            │
  └──────────────┘            │
        │ speech_start        │
        ▼                     │
  ┌──────────────┐            │ (no-speech / too-short)
  │    ACTIVE    │────────────┤
  └──────────────┘            │
        │ speech_end          │
        ▼                     │
  ┌──────────────┐ barge_in   │
  │  RESPONDING  │────────────┤ (InterruptBus → abort chain → ACTIVE)
  └──────────────┘            │
        │ response.done       │
        └─────────────────────┘
```

- 전이는 `asyncio.Lock`(`state_lock`) 아래 atomic.
- Patience 타이머: LISTENING 중에만.
- 마이크 뮤트: **별도 상태 아님**. `mute_mic = Phase != ACTIVE` 파생. 이전 `speaking: Event` 불필요.

## 8. 오류 복구 & 관측성

### 재시도 정책

| 호출 | 정책 |
|------|------|
| OpenAI Realtime connect | 3회 재시도 exp backoff 0.5→1→2 s. 실패 시 Chat 프로파일 폴스루(설정 시) 또는 raise. |
| ElevenLabs TTS HTTP | 429/5xx 2회 재시도, 0.3→0.8 s. 최종 실패: 문장 스킵 + 로그, 응답 차단 안 함. |
| Whisper STT | 2회 재시도, `[transcription failed]` 텍스트 주입 (LLM 에 신호). |
| DuckDuckGo | 1회 재시도. 실패 무음 (검색은 선택적). |
| ChromaDB query | 재시도 없음. 200 ms 타임아웃. 실패 무음. |

### Circuit Breaker

서비스별 (OpenAI, ElevenLabs) 1개. CLOSED → OPEN (5연속 실패) → HALF_OPEN (30 s 후). OPEN 시 `ProviderUnavailable` 즉시 raise — 오케스트레이터가 로그 + 계속(TTS 는 문장 스킵, LLM 은 턴 포기 + 사과 메시지).

80 LOC 자체 구현 (Python 생태계의 circuit breaker 라이브러리는 opaque failure mode).

### Structured Logging (structlog)

모든 `print(file=sys.stderr)` 를 `structlog.get_logger()` 로 교체. Base context: `{session_id, phase, profile}`. 단계별:

```python
log.info("stage.start", stage="tts", seq=3, char_len=42)
log.info("stage.end",   stage="tts", seq=3, duration_ms=215)
```

### 메트릭

`zemory.observability.metrics.Metrics` — `dict[str, Histogram]` (last 500 samples, `p50()`/`p95()` on demand). 키:

- `ttfb.stt`, `ttfb.llm`, `ttfb.tts`, `ttfb.speaker` (ms)
- `interrupt.chain_total_ms`
- `chroma.query_ms`, `chroma.embed_ms`
- `turn.total_ms` (speech_end → speaker first write — **핵심 메트릭**)

`zemory.observability.stats_server` — 선택적 aiohttp 엔드포인트(포트 9100), Prometheus text format. `[project.optional-dependencies].observability` extras.

### 지연 측정 훅

`SpeakerStream`: 버퍼 첫 non-empty 시 `_first_write_at` 기록. 오케스트레이터: `turn.total_ms = speaker._first_write_at - turn_detector._speech_end_at`.

## 9. 파일 레이아웃

**결정: `zemory/` 로 수렴, `zemory_vad/` 는 한 릴리스 `zemory/legacy/` 보관 후 삭제**.

```
zemory/
├── __init__.py
├── __main__.py
├── audio.py                          # 확장 (first_write_at 훅)
├── config.py                         # Pydantic Settings + TOML + 프로파일
├── orchestrator.py                   # 신규 — run() + 상태 머신
├── state.py                          # 신규 — Phase enum + state_lock
├── vad.py                            # zemory_vad/vad.py 이동
├── observability/
│   ├── log.py
│   ├── metrics.py
│   └── stats_server.py
├── providers/
│   ├── base.py                       # Protocol + Injection dataclass
│   ├── turn/{server_vad,silero}.py
│   ├── stt/{null,openai_whisper}.py
│   ├── llm/{openai_realtime,openai_chat}.py
│   └── tts/{elevenlabs,kokoro}.py
├── pipeline/
│   ├── chunker.py                    # SentenceChunker 통합
│   ├── tts_manager.py                # OLV 패턴
│   ├── prompt_assembler.py           # Neuro priority
│   ├── memory.py                     # ChromaDB + reflection
│   ├── interrupt_bus.py
│   └── circuit_breaker.py
├── sources/{twitch,youtube,patience,web_search}.py
├── characters/zemory/
│   ├── persona.md
│   ├── voice.toml
│   └── memory_bootstrap.json
├── prompts/{reflection,uncertainty_classifier,patience}.md
└── legacy/                           # zemory_vad/ 한시 보관

tests/
├── conftest.py                       # FakeLLM/FakeTTS/FakeSpeaker
├── test_chunker.py
├── test_state_machine.py
├── test_prompt_assembler.py
└── test_interrupt_chain.py

config.toml
scripts/
├── bench_latency.py
└── interrupt_stress.py
```

## 10. 의존성 델타

추가:

| 패키지 | 제약 | 이유 |
|--------|------|------|
| `chromadb` | `>=0.5,<0.6` | 장기 메모리. Neuro 0.5.0 호환, 0.6 은 breaking API. |
| `sentence-transformers` | `>=3.0` | Chroma 임베딩 기본. |
| `structlog` | `>=24.1` | 구조적 로깅. |
| `pydantic-settings` | `>=2.4` | Config + 검증 + env override. |
| `httpx` | `>=0.27` | 이미 openai 통해 간접 포함, 명시화. |
| `pytchat` | `>=0.5` | YouTube 채팅. |
| `aiohttp` | `>=3.9` | stats server. `[observability]` extras. |
| `pytest` | `>=8.0` | dev group. |
| `pytest-asyncio` | `>=0.23` | dev group. |

유지: `openai[realtime]`, `sounddevice`, `numpy`, `python-dotenv`, `onnxruntime`, `silero-vad`, `duckduckgo-search`.

**Silero `onnx=True` 전환** — CPU 2× 빠름, Torch 를 VAD 임계경로에서 제거.

## 11. Migration 10단계

각 단계는 독립 배포 가능. 매 단계 후 `python -m zemory` 로 회귀 검증.

1. **스캐폴딩** — 디렉터리 + `__init__.py` + Pydantic Settings(TOML+env). `SentenceChunker` → `pipeline/chunker.py`, 이전 위치에 import-shim.
2. **Provider Protocol + Realtime 어댑터 추출** — `providers/base.py` 정의. `client.py` 이벤트 핸들링을 `providers/llm/openai_realtime.py` 로 추출. `orchestrator.py` 가 registry 로 인스턴스화. `test_chunker.py` 통과.
3. **Unified state machine + 타이밍 훅** — `Phase` enum + `state_lock`. `speaking: Event` 패턴 제거. `SpeakerStream._first_write_at` 훅. structlog 기본. **baseline p50 측정**.
4. **TTSTaskManager** — `pipeline/tts_manager.py` + seq-ordered 버퍼 + Semaphore(3). `tts_worker` 교체. 멀티문장 응답 wall-time 30-50% 개선.
5. **인터럽트 체인** — `InterruptBus`. Realtime `speech_started` + Silero `speech_start` 와이어링. Realtime 세션 `interrupt_response=true` 전환. 테스트: 인터럽트 → 스피커 무음 < 150 ms.
6. **로컬 프로파일 폴드-인** — `zemory_vad/vad.py` → `zemory/vad.py`. `SileroTurnDetector`, `WhisperSTT` 어댑터. `profile=local` 동작. 양 프로파일 동일 오케스트레이터.
7. **ChromaDB 메모리 + Reflection** — `pipeline/memory.py`. `PromptAssembler` 회수 주입(priority 60). 20턴 Reflection. 캐릭터 bootstrap JSON.
8. **Priority PromptAssembler + Patience** — 인젝션 시스템 일반화. `PatienceSource` 60 s. Neuro priority.
9. **Twitch/YouTube sources** — Raw IRC + pytchat. Rate-limit 5 msg / 10 s. priority 150 주입.
10. **Observability + Circuit Breakers + legacy 삭제** — 전체 메트릭. OpenAI + ElevenLabs circuit breaker. Prometheus 엔드포인트 (`[observability]` extras). `zemory/legacy/` 삭제.

1-5 = 저지연 임계, 6-10 = capability.

## 12. 검증

### End-to-end 지연 측정

매 턴마다 `log.info("turn.complete", ...)` 로 다음 필드 JSONL:
```json
{"turn_id", "speech_end_ts", "first_llm_delta_ts", "first_tts_byte_ts",
 "speaker_first_write_ts", "total_ms", "profile", "interrupted"}
```

`scripts/bench_latency.py` 로 최근 100 턴:
```
profile=realtime  n=100  p50=612ms  p95=1031ms  p99=1544ms
profile=local     n=100  p50=814ms  p95=1203ms  p99=1876ms
```

### 인터럽트 지연 테스트

응답 시작 400 ms 후 fake `speech_start` 주입 → `interrupt.chain_total_ms` 측정. **목표: p95 < 150 ms**.

### Dev 명령

```bash
uv run python -m zemory
ZEMORY_PROFILE=local uv run python -m zemory
uv run python scripts/bench_latency.py logs/latest.jsonl
uv run python scripts/interrupt_stress.py
uv run pytest tests/
uv run ruff check zemory/
```

### 테스트 인프라 (최소)

- `pyproject.toml` 에 `[tool.pytest.ini_options] asyncio_mode = "auto"`.
- `tests/conftest.py`: `FakeLLMProvider`, `FakeTTSProvider`, `FakeSpeaker` — 네트워크 없이 integration 테스트.
- 최소 4 테스트:
  1. `test_chunker.py` — 한국어+영어 혼합, 말줄임표 경계.
  2. `test_state_machine.py` — 모든 유효 전이 + invalid 거부.
  3. `test_prompt_assembler.py` — priority 순서 + 토큰 예산 가지치기.
  4. `test_interrupt_chain.py` — 단계 순서 + 타이밍 (asyncio.sleep mock).

CI 는 out of scope.

### 릴리스 수동 체크리스트

1. 한국어 30초 대화, 인터럽트 없음 — p50 < 800 ms 로깅.
2. 3연속 중간 인터럽트 — 각 < 200 ms 무음.
3. 60초 유휴 → Patience 자율 발화.
4. env var 로 프로파일 스왑, 5턴 대화 양쪽 동작.
5. 네트워크 차단 → circuit breaker OPEN, 명확한 로그 + 깔끔한 종료.

## 13. Out of Scope

Phase 4+ 로 연기:

- 웹 UI (OLV/RVC FastAPI+WebSocket)
- Live2D / VRM / 아바타 렌더 (airi stage-ui-three, Neuro vtube_studio)
- 멀티모달 비전 (Neuro imageLLMWrapper + mss 캡처)
- 한국어+영어 외 언어
- 분산/멀티 인스턴스 (airi Redis pub/sub)
- 보이스 클로닝 (Neuro XTTS-v2)
- 로컬 LLM (AIRIS transformers, Ollama)
- Tool/function calling
- Discord/Telegram/Minecraft

## 14. 레퍼런스 충돌 시 결정 근거

1. **OLV 단일 task.cancel vs RVC granular abort** → RVC. granular cancel 이 연결 상태 보존; OLV 패턴은 frontend-backend 격리 없으면 취약.
2. **Neuro constants.py 하드코딩 vs airi TOML 핫리로드** → TOML. Neuro 자체가 결함으로 명기.
3. **AIRIS 블로킹 TTS 문장별 vs OLV parallel-with-seq-order** → OLV. AIRIS 약점 명시.
4. **RVC DistilBERT 턴 감지 vs server_vad** → server_vad. 한국어 ergonomics + zero inference cost.
5. **Neuro 4-daemon-thread vs 순수 asyncio** → asyncio. Neuro 자체의 threading/async 혼합 버그 문서화.
6. **airi VAD Worker 격리** → skip. Silero ~3 ms 는 프로세스 경계 정당화 불가. `ProcessPoolExecutor` 는 향후 로컬 TTS/STT 전용 예약.
