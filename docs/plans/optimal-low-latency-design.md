# zemory-sama 실시간 음성 에이전트 설계

> 개정일: 2026-06-27
> 근거 스냅샷: [docs/ref/current-state-2026-06-27.md](../ref/current-state-2026-06-27.md)

## 0. Assumptions

- 이 설계는 현재 Python CLI/서버형 코어를 유지한다. 브라우저 WebRTC 클라이언트, VRM, OBS 연동은 다음 단계다.
- 1차 사용자 경험은 다국어 저지연 음성 대화다. 한국어와 영어를 우선 fixture 로 검증하고, 텍스트 채팅, 방송 채팅, 게임/도구 연동은 사이드 입력이다.
- 기존 구현의 provider/profile 방향은 유지하되, 기본 fast path 는 외부 TTS 체인이 아니라 OpenAI Realtime GA의 오디오 네이티브 세션으로 바꾼다.
- 최신 풀듀플렉스 음성 모델 연구는 중요하지만, 지금 기본 런타임으로 채택하지 않는다. 검증 전까지는 실험 프로파일로 둔다.

## 1. 변경 결론

### 이전 설계에서 바꾸는 것

- **기본 프로파일 변경**: `gpt-4o-mini-realtime-preview` + Realtime text + ElevenLabs TTS가 아니라 `gpt-realtime-2` Realtime audio-in/audio-out을 기본값으로 둔다.
- **VAD 기본값 변경**: 침묵 기반 `server_vad`만 튜닝하지 말고 `semantic_vad`를 우선 사용한다. `server_vad`와 로컬 Silero는 fallback 이다.
- **RAG/메모리 변경**: Chroma 조회를 턴 생성 전에 동기 차단하지 않는다. 메모리와 도구 호출은 deadline 을 가진 비동기 작업으로 보내고, 늦으면 다음 발화나 다음 턴에 반영한다.
- **외부 TTS 지위 변경**: ElevenLabs는 개성 있는 목소리가 필요할 때 쓰는 `realtime_text_external_tts` 프로파일이다. 최저 지연 기본 경로가 아니다.
- **풀듀플렉스 채택 방식 변경**: Moshi, MoshiRAG, Raon-Speech, BayLing-Duplex는 연구/벤치마크 트랙이다. 기본 프로덕션 경로는 OpenAI Realtime 세션의 interruption/barge-in에 집중한다.

### 유지하는 것

- provider abstraction, runtime profile 선택, `InterruptBus`, `TTSTaskManager`, `SpeakerStream` 같은 기존 경계는 유지한다.
- 로컬 cascade 프로파일은 개발/비상 fallback 으로 남긴다.
- 대화 기록, partial assistant 기록, 메모리, 방송 채팅, patience는 core loop 옆의 sideband 기능으로 둔다.

## 2. Success Criteria

| 항목 | 목표 |
| --- | --- |
| Realtime audio 첫 출력 | `speech_stopped` 또는 semantic turn end 이후 `response.output_audio.delta` p50 <= 700 ms, p95 <= 1200 ms |
| Barge-in | assistant 발화 중 사용자 speech start 이후 로컬 스피커 무음 <= 150 ms |
| RAG 영향 | 메모리/도구 작업을 켜도 첫 오디오 p50 증가 <= 50 ms |
| 장기 세션 | 60분 세션에서 context truncation 전 자체 compaction 또는 durable summary 수행 |
| 언어별 턴 종료 | 한국어/영어 fixture 에서 조기 끊김과 늦은 응답을 별도 지표로 측정 |
| 회귀 검증 | unit test, lint, latency fixture, interruption fixture가 통과해야 release 후보 |

## 3. Runtime Profiles

| Profile | 기본 여부 | 구성 | 목적 |
| --- | --- | --- | --- |
| `realtime_audio` | 기본 | OpenAI Realtime GA, `gpt-realtime-2`, audio input/output, `semantic_vad`, WebSocket server transport | 가장 낮은 체감 지연, barge-in, 자연스러운 턴 처리 |
| `realtime_text_external_tts` | 선택 | Realtime text output, SentenceChunker, TTSTaskManager, ElevenLabs/Kokoro TTS | 특정 목소리 품질이나 외부 TTS가 더 중요한 캐릭터 |
| `local_cascade` | fallback | Silero VAD, Whisper/STT, text LLM, external/local TTS | API 장애, 비용 제어, 오프라인/개발 환경 |
| `research_full_duplex` | 실험 | Moshi/Raon/BayLing 계열 또는 compatible local SLM | 논문 재현, 다국어 풀듀플렉스 평가 |

## 4. Target Architecture

```
mic pcm24k
  |
  v
AudioIngress
  |
  +--> TurnLayer
  |      - semantic_vad (Realtime default)
  |      - server_vad fallback
  |      - Silero fallback/local
  |
  v
RealtimeSessionAdapter / CascadeSessionAdapter
  |
  +--> TranscriptLedger
  +--> InterruptController
  +--> AsyncContextScheduler
  |      - memory recall
  |      - tool/RAG calls
  |      - chat side input
  |      - compaction
  |
  v
ResponseStream
  |
  +--> audio-native: response.output_audio.delta -> SpeakerStream
  |
  +--> external TTS: text delta -> SentenceChunker -> TTSTaskManager -> SpeakerStream
```

## 5. Realtime Audio Profile

### Session shape

`realtime_audio`는 GA interface 를 기준으로 한다.

- `session.type = "realtime"`
- `model = "gpt-realtime-2"`
- `output_modalities = ["audio"]`
- input format: PCM 24 kHz
- output voice: `marin` 또는 `cedar`를 기본 후보로 둔다.
- turn detection: `semantic_vad` with `eagerness = "medium"` 기본, 한국어/영어 fixture 에서 조정한다.
- beta header (`OpenAI-Beta: realtime=v1`)와 오래된 event shape는 사용하지 않는다.
- 새 event 이름(`response.output_audio.delta`, `response.output_audio_transcript.delta`, `response.output_text.delta`)을 기준으로 adapter 를 작성한다.

### Reasoning and prompts

- production 기본값은 낮은 latency 를 위해 `reasoning.effort = "low"`에서 시작한다.
- 예전 prompt 를 그대로 쓰지 않는다. 최신 Realtime 모델은 instruction adherence 가 강하므로, 캐릭터 규칙은 짧고 테스트 가능한 문장으로 둔다.
- hosted prompt 는 선택 기능이다. 먼저 로컬 prompt 파일과 versioned config 로 재현성을 확보한다.

### Long-session handling

Realtime 세션은 60분까지 갈 수 있지만 context window 는 무한하지 않다. 자동 truncation 에 맡기면 캐릭터 상태와 장기 기억 연결이 끊길 수 있다.

- `TranscriptLedger`는 모든 user/assistant transcript, interruption, tool result 를 durable log 로 남긴다.
- `ContextCompactor`는 일정 토큰 또는 시간 기준으로 요약을 생성해 session 앞부분 손실 전에 durable memory 로 보낸다.
- session truncation 설정은 명시적으로 config 에 둔다. 기본은 자동 truncation 허용 + 자체 summary 선행이다.

## 6. TurnLayer

### Default: semantic VAD

`semantic_vad`는 사용자가 실제로 발화를 끝냈는지를 단어 의미 기반으로 판단한다. 한국어의 조사/어미 끝맺음과 영어의 짧은 backchannel/fragment 모두 침묵 시간만 보는 로직으로는 다루기 어렵기 때문에 기본값으로 둔다.

튜닝 값:

- `eagerness = "medium"` 기본.
- 끊김이 많으면 `low`.
- 응답이 느리면 `high`를 실험하되 한국어/영어 fixture 에서만 승격한다.

### Fallback: server VAD

`server_vad`는 네트워크/모델/세션에서 semantic mode 가 불안정할 때 사용한다.

- noisy 환경에서는 `threshold`를 올린다.
- 빠른 응답이 필요하면 `silence_duration_ms`를 줄인다.
- idle prompt 는 `idle_timeout_ms`로 처리한다. 별도 patience timer 와 중복되지 않게 하나만 켠다.

### Local fallback: Silero

Silero는 `local_cascade`와 self-audio 방어용 로컬 힌트로만 사용한다. Realtime 세션의 source of truth 는 서버 VAD event 다.

### Research: endpoint anticipation

Endpoint Anticipation 연구는 최대 2.56초 선행 예측과 speculative LLM/TTS 실행을 제안한다. 적용은 실험 옵션으로 제한한다.

- partial transcript 가 안정된 뒤 speculative turn 을 시작한다.
- 사용자가 계속 말하면 즉시 cancel 한다.
- 추가 compute 비율, 잘못 시작한 응답 비율, 언어별 오검출률을 metrics 로 기록한다.
- 기본값으로 켜지 않는다.

## 7. Interrupt and Barge-in

### Trigger

- Realtime: `input_audio_buffer.speech_started`가 assistant output 중 발생.
- Local: Silero speech start 가 `RESPONDING` 또는 `SPEAKING` 중 발생.
- 디바운스: 마지막 trigger 이후 250 ms 이내면 무시한다.

### Abort sequence

```
InterruptController.trigger("user_barge_in")
  1. SpeakerStream.clear()
  2. external TTS task cancel if active
  3. realtime response.cancel() or cascade LLM cancel_current()
  4. partial assistant transcript 기록
  5. input buffer 정책 적용
  6. phase = LISTENING/USER_SPEAKING
```

구현 상태: `orchestrator.py`는 `InterruptBus(..., on_partial=on_partial_abort)`로 연결되어 있으며, Realtime `speech_started` 처리도 partial text를 reset 하기 전에 interrupt를 트리거하도록 분리되어 있다. 회귀 테스트는 `tests/test_realtime_event_handlers.py`가 담당한다.

## 8. External TTS Profile

`realtime_text_external_tts`는 더 이상 기본 fast path 가 아니다. 그래도 캐릭터 음색 때문에 필요하다.

- Realtime text delta 또는 text LLM token stream 을 `SentenceChunker`로 넘긴다.
- soft boundary 는 `, : ; 、 ，`를 쓰되 최소 길이 조건을 둔다.
- `TTSTaskManager`는 sentence 단위 병렬 합성 + sequence-ordered playback 을 유지한다.
- first sentence 는 낮은 latency 옵션, 이후 문장은 품질 옵션을 쓸 수 있다.
- abort 는 Realtime response cancel 과 TTS task cancel 을 분리한다.

## 9. Memory, RAG, Tools

### Principle

첫 오디오를 막는 memory/RAG는 금지한다. 검색은 실시간 대화를 보강해야지 턴 시작을 붙잡아서는 안 된다.

### AsyncContextScheduler

각 user turn 에서 다음 작업을 병렬 실행한다.

- `memory_recall`: 최근 user text 기준 top-k recall. deadline 80 ms.
- `tool_or_rag`: 검색, MCP, 내부 도구. deadline 은 tool 별 config.
- `chat_context`: Twitch/YouTube/Discord side input compact.
- `compaction`: 오래된 transcript summary.

결과 정책:

- deadline 안에 도착하면 현재 response 의 context/tool result 로 반영한다.
- assistant 핵심 답변이 이미 시작됐으면 끼워 넣지 않고 follow-up 또는 다음 턴에 반영한다.
- 도구 결과가 pending 이면 모델이 결과를 지어내지 않도록 pending 상태를 명시한다.

### Storage interface

ChromaDB를 특정하지 말고 interface 를 둔다.

```python
class MemoryStore(Protocol):
    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]: ...
    async def write_reflection(self, items: list[MemoryItem]) -> None: ...
    async def compact_session(self, transcript: TranscriptWindow) -> SessionSummary: ...
```

초기 구현:

- vector: ChromaDB 또는 sentence-transformers 기반 local vector store.
- metadata/log: SQLite.
- web/browser 확장 시 DuckDB/pglite 계열도 고려한다.

### Research influence

MoshiRAG, WavRAG, Stream RAG의 공통 신호는 검색과 도구 호출을 발화 종료 뒤의 blocking 단계로만 두지 말고, 가능한 한 대화 흐름과 병렬화하라는 것이다. zemory-sama에서는 우선 partial transcript 기반 async query 로 흡수하고, audio-native RAG는 연구 트랙으로 둔다.

## 10. Prompt and State

PromptAssembler는 유지하되, Realtime audio profile 에 맞게 역할을 바꾼다.

- session instructions: 캐릭터 핵심, 안전, 말투, 답변 길이.
- dynamic context: 최근 transcript summary, memory hits, side input.
- tool policy: 언제 도구를 부를지, pending 결과를 어떻게 말할지.
- speech policy: 사용자가 말한 언어로 짧게 시작하고, 불확실하면 확인 질문을 먼저 한다.

Priority 는 단순하게 유지한다.

| Source | Priority | 처리 |
| --- | --- | --- |
| Persona/system | 10 | session start/update |
| Durable summary | 40 | session update 또는 hidden context |
| Recent history | 50 | TranscriptLedger window |
| Memory hits | 60 | deadline 내 도착 시 |
| Tool result | 80 | 도착 즉시, stale 검사 |
| Chat side input | 150 | rate limit + compact |
| User override | 200 | 명시 입력일 때만 |

## 11. Observability and Tests

필수 timestamp:

- mic frame received
- `speech_started`
- `speech_stopped` 또는 semantic turn end
- `response.created`
- first `response.output_audio.delta`
- first speaker buffer write and first playback callback
- interrupt trigger
- speaker cleared
- retrieval start/ready/late
- tool call start/output

필수 fixture:

- 한국어 짧은 명령
- 한국어 긴 종결어미 문장
- 영어 짧은 명령
- 영어 긴 fragment/backchannel 문장
- assistant 발화 중 barge-in
- tool/RAG 지연 0 ms, 200 ms, 1000 ms
- noisy silence

Release gate:

- `uv run pytest tests/` (core coverage gate: >= 80%)
- `uv run ruff check zemory tests`
- `uv run python -m compileall zemory tests scripts`
- latency fixture p50/p95 threshold 통과
- interruption fixture <= 150 ms

## 12. Reference Project Weighting

| Tier | Project | 쓰는 방식 |
| --- | --- | --- |
| A | OpenAI Realtime official docs | 기본 runtime/API/event 모델 |
| A | AIRI | 멀티모달/브라우저/메모리/voice provider 방향성 |
| A- | Open-LLM-VTuber | v2 rewrite 관찰, provider/interrupt/agent interface 패턴 |
| B | RealtimeVoiceChat | unmaintained 이므로 패턴만 참고: interrupt, local cascade, Docker |
| B | Neuro | memory/RAG, prompt priority, module injection 패턴 |
| C | AI-Waifu-Vtuber | legacy streaming reference. 새 설계 근거로 쓰지 않음 |
| C | AIRIS-VtuberAI | GPU/local pipeline 관찰용. planned interrupt/tool calling 확인 후 재평가 |
| Watch | projectBEA, nekro-agent, Xiao8 | awesome list에서 새 후보로 추적. core 설계 의존 없음 |

## 13. Implementation Roadmap

### Phase 1: Realtime GA audio migration

- config 기본 모델을 `gpt-realtime-2`로 바꾼다.
- session shape 를 GA event schema 로 맞춘다.
- output audio delta 를 SpeakerStream 으로 직접 연결한다.
- `semantic_vad`와 `server_vad` fallback 을 profile config 로 노출한다.

### Phase 2: Interrupt correctness

- `InterruptBus` partial callback wiring 을 고친다. (구현됨)
- Realtime `response.cancel()`과 local TTS cancel 을 분리한다.
- barge-in fixture 를 추가한다.

### Phase 3: Metrics and benchmark

- timestamp hooks 를 추가한다.
- Korean/English audio fixtures 로 latency report 를 만든다. (JSONL latency gate 구현됨)
- p50/p95와 interruption latency 를 CI 또는 local benchmark 에서 확인한다.

### Phase 4: Async memory/tool layer

- `TranscriptLedger`, `MemoryStore`, `AsyncContextScheduler`를 추가한다. (SQLite local store 구현됨)
- memory recall deadline 과 late-result policy 를 테스트한다. (구현됨)
- tool/RAG callable deadline 과 late-result policy 를 테스트한다. (구현됨)
- Realtime async function calling/MCP 경로는 tool policy 가 준비된 뒤 켠다.

### Phase 5: External TTS profile cleanup

- ElevenLabs 경로를 선택 프로파일로 정리한다.
- SentenceChunker/TTSTaskManager test 를 보강한다.
- audio-native profile 과 code path 를 혼동하지 않게 interface 를 분리한다.

### Phase 6: Research profile

- Moshi/MoshiRAG/Raon-Speech/BayLing 계열을 별도 branch/profile 로 평가한다.
- 한국어/영어 풀듀플렉스, interruption, backchannel, factuality benchmark 를 만든다.
- production 승격은 latency, stability, hardware cost 를 통과한 뒤만 한다.

## 14. Deferred Decisions

- VRM/OBS/Twitch UI는 voice core 안정화 전까지 설계 범위 밖이다.
- WebRTC browser client 는 필요해질 때 별도 frontend 설계로 분리한다.
- full-duplex local SLM은 research profile 에서만 다룬다.
- audio-native RAG는 논문 추적 대상이며 즉시 구현하지 않는다.
- 자체 turn classifier 학습은 fixture 가 semantic/server VAD의 한계를 증명할 때만 시작한다.
