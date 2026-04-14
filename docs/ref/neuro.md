# Neuro (kimjammer) 대화 파이프라인 분석

> GitHub: https://github.com/kimjammer/Neuro (1.9k stars, 7-day 해커톤 기반)
> 분석일: 2026-04-15

## 아키텍처 개요

Python 기반 로컬 단일 프로세스. 전역 `Signals` 싱글톤을 허브로 두고 STT·TTS·LLM·Prompter·Socket.io 서버 + 7개 옵션 모듈을 데몬 스레드로 동시 구동. Open-LLM-VTuber 처럼 프런트/백 네트워크 경계가 없이 모두 한 프로세스에서 실행되며, 프런트 UI 는 Socket.io 로 상태만 수신한다.

**엔트리** (`main.py:25-115`): `Signals` → STT/TTS/LLMState/TextLLMWrapper/ImageLLMWrapper/Prompter 생성 → 모듈 7개 생성 → 4개 데몬 스레드(prompter_loop, STT.listen_loop, SocketIOServer, 각 모듈 async event loop) 기동 → 메인 스레드는 `signals.terminate` 를 100 ms 간격 폴링.

---

## 1. STT (RealtimeSTT + Faster-Whisper)

**엔진**: RealtimeSTT v0.3.7 내부 Faster-Whisper `tiny.en`. VAD 는 Silero 지만 RealtimeSTT 내부 암묵. `silero_sensitivity=0.6`.

**설정** (`stt.py:36-53`):

| 파라미터 | 값 |
|---------|----|
| `realtime_model_type` | `tiny.en` |
| `silero_sensitivity` | 0.6 |
| `post_speech_silence_duration` | 0.4 s |
| `min_gap_between_recordings` | 0.2 s |
| `enable_realtime_transcription` | True |

**콜백** (`stt.py:14-32`): 발화 종료 시 `signals.history` 에 `role="user"` 로 추가, `signals.human_speaking` on/off, `signals.new_message=True` (AI 가 말하고 있지 않을 때만).

`recorder.text(callback)` 는 **블로킹 호출**이며 이 스레드 안에서 VAD+Whisper+후침묵을 모두 처리한다. 명시적 상태 머신은 없고 Open-LLM-VTuber 의 3-hit / 24-miss 패턴과 달리 라이브러리 내부 휴리스틱에 의존.

---

## 2. LLM (text-generation-webui HTTP + SSE)

**백엔드**: text-generation-webui 의 `/v1/chat/completions` 엔드포인트 (기본 `http://127.0.0.1:5000`). 모델은 Llama 3 8B 가정(stop_strings `["\n", "<|eot_id|>"]`).

### 듀얼 래퍼 패턴

- `llmWrappers/abstractLLMWrapper.py` — 공통 프롬프트 조립 + 토큰 예산 + 스트리밍
- `llmWrappers/textLLMWrapper.py` — 텍스트 전용
- `llmWrappers/imageLLMWrapper.py` — 스크린샷 캡처(mss) → JPEG base64 → MiniCPM-Llama3-V-2.5-int4

### 프롬프트 조립 (`abstractLLMWrapper.py:63-101`)

우선순위 기반 인젝션 시스템:

| 인젝션 | priority |
|--------|----------|
| System Prompt | 10 |
| Chat History | 50 |
| Memory (RAG) | 60 |
| Twitch Messages | 150 |
| Custom Prompt | 200 |

우선순위 오름차순 정렬 → 높은 priority 가 프롬프트 **끝**에 배치. HF tokenizer 로 토큰 수 추정, `CONTEXT_SIZE` 의 90% 초과 시 가장 오래된 메시지를 drop.

### 스트리밍 (`abstractLLMWrapper.py:106-148`)

`sseclient.SSEClient` 로 Server-Sent Events 수신, 토큰마다 `signals.sio_queue.put(("next_chunk", chunk))` → Socket.io 프런트로 즉시 브로드캐스트. `llmState.next_cancelled` 플래그로 스트림 드레이닝, `/v1/internal/stop-generation` POST 로 서버측 취소.

---

## 3. TTS (Coqui XTTS-v2 via RealtimeTTS)

**엔진** (`tts.py:13-24`):

```python
engine = CoquiEngine(use_deepspeed=True, voice="./voices/neuro.wav", speed=1.1)
stream = TextToAudioStream(engine,
    on_audio_stream_start=self.audio_started,
    on_audio_stream_stop=self.audio_ended,
    output_device_index=OUTPUT_DEVICE_INDEX)
```

5-30 초 참조 음성으로 음색 복제. `play_async()` 는 논블로킹 전체 메시지 재생 — Open-LLM-VTuber 의 **문장 단위 순차 큐** 와 달리 문장 분할 없이 한 덩어리로 합성.

**상태** (`tts.py:42-47`): 재생 시작 시 `signals.AI_speaking=True`, 종료 시 False + `last_message_time` 갱신.

---

## 4. 에코·인터럽트 (불완전)

**에코 방지**: 소프트웨어 게이팅 없음. `INPUT_DEVICE_INDEX` / `OUTPUT_DEVICE_INDEX` 하드웨어 분리에만 의존. 사용자가 장치를 잘못 배정하면 루프백 발생.

**인터럽트**: **실질적으로 없음**. Prompter 가 `human_speaking` / `AI_speaking` / `AI_thinking` 중 어느 것이라도 True 면 **새 프롬프트를 생성하지 않는** *permission check* 만 존재(`prompter.py:21-22`). AI 가 이미 발화 중이면 사용자가 말을 걸어도 TTS/LLM 이 중단되지 않는다. README 의 Discord 음성 통합은 `streamingSink.py` 미완결 상태로 남음.

이는 Open-LLM-VTuber 의 `asyncio.Task.cancel()` 기반 능동 취소나 RealtimeVoiceChat 의 `abort` 체인 대비 명백한 회귀점.

---

## 5. 턴테이킹

RealtimeSTT 내부 0.4 s 후침묵 기준만 사용. zemory-sama / Open-LLM-VTuber 의 ~768 ms 보다 짧아 한국어처럼 체언+조사 구조가 긴 언어에서는 조기 종료 가능성. 외부 VAD 파라미터로 제어할 수 없고, 실패 시 턴이 사일런트 손실된다.

---

## 6. 메모리 (Chroma + 반성 패턴)

**백엔드**: ChromaDB v0.5.0 (`memories/chroma.db`, Sentence Transformers 임베딩). Generative Agents (Park et al., 2023) 스타일 **reflection**:

1. 20+ 신규 메시지 누적 시 (`memory.py:63`) LLM 에게 "가장 중요한 Q&A 3쌍"을 요청
2. `{qa}` 구분자로 분할된 결과를 short-term 문서로 저장
3. 프롬프트 조립 시 최근 5개 메시지를 쿼리로 top-5 회수 → priority 60 으로 인젝션

**API**: `create_memory` / `delete_memory` / `wipe` / `clear_short_term` / `import_json` / `export_json` / `get_memories(query)`. Socket.io 이벤트로 UI 에서 직접 편집 가능.

zemory-sama 가 로컬 장기 기억을 추가할 때 **가장 모방 가치가 높은 패턴**.

---

## 7. 모듈 시스템 (우선순위 인젝션)

`modules/module.py`:

```python
class Module:
    def __init__(self, signals, enabled=True): ...
    def init_event_loop(self): asyncio.run(self.run())
    def get_prompt_injection(self) -> Injection: ...
    async def run(self): pass
```

등록된 모듈(`main.py:44-73`): `twitch`, `audio_player`, `vtube_studio`, `multimodal`, `custom_prompt`, `memory`, `discord`(commented). **런타임 활성화 불가** — 시작 시 비활성이면 그대로.

모듈 예:
- **TwitchClient** — 비동기 Twitch Chat API 루프, 최대 10개 버퍼, priority 150 인젝션 ("가장 양질의 메시지에 응답하라")
- **VtubeStudio** — pyvts 로 VtubeStudio API 연결, 큐 기반 액션 프로세서로 hotkey / 모델 위치 / 마이크 스프라이트 등을 제어
- **Memory** — 5초 주기 asyncio 태스크. 하지만 **async 컨텍스트 내에서 blocking `requests.post`** 를 호출하는 설계 결함
- **CustomPrompt** — 사용자 실시간 텍스트 주입, priority 200 으로 프롬프트 최후미

---

## 8. Prompter (턴 조율)

`prompter.py` — 100 ms 폴링 루프:

```python
def prompt_now(self):
    if not stt_ready or not tts_ready: return False
    if human_speaking or AI_thinking or AI_speaking: return False
    if new_message: return True
    if recentTwitchMessages: return True
    if time_since_last > PATIENCE: return True   # PATIENCE=60s
```

Patience 타임아웃으로 **침묵 60초 후 자율 발화** 트리거 — "AI VTuber 가 방송 중 말을 계속 하게" 만드는 핵심 장치. `patience_update` 이벤트를 50 ms 간격으로 프런트로 송출해 진행률 바 UI 제공.

멀티모달 선택 로직: `multimodal` 모듈이 활성이면 `imageLLMWrapper` 로 라우팅(데스크톱 캡처 + Vision LLM).

---

## 9. Socket.io 이벤트 허브

**포트**: 8080 하드코딩. `python-socketio` + `aiohttp`.

핵심 이벤트:

| 이벤트 | 방향 | 용도 |
|-------|------|------|
| `next_chunk` | B→F | LLM 토큰 스트림 |
| `current_message` | B→F | 현재 TTS 메시지 |
| `human_speaking` / `AI_speaking` / `AI_thinking` | B→F | 상태 UI |
| `patience_update` | B→F | 자율 발화 진행률 |
| `disable_LLM/TTS/STT` | F→B | 토글 |
| `fun_fact` / `new_topic` / `cancel_next_message` | F→B | 프롬프트 조작 |
| `nuke_history` | F→B | 히스토리 리셋 |
| `create_memory` / `delete_memory` / `get_memories` | 양방향 | 메모리 CRUD |
| `set_custom_prompt` | F→B | 즉석 인젝션 |

Signals property setter 가 `sio_queue.put` 을 자동 호출하는 **암묵적 사이드 이펙트** 패턴 — 구독 없이 상태 변경이 프런트로 전파됨.

---

## 10. 설정 (constants.py)

주요 튜너블:

| 상수 | 기본값 | 용도 |
|------|--------|------|
| `PATIENCE` | 60 | 자율 발화 대기 시간 |
| `CONTEXT_SIZE` | 8192 | 토큰 예산 (90% 사용) |
| `MULTIMODAL_CONTEXT_SIZE` | 1000 | Vision 모델 |
| `MEMORY_QUERY_MESSAGE_COUNT` | 5 | 쿼리용 최근 메시지 |
| `MEMORY_RECALL_COUNT` | 5 | 회수 개수 |
| `VOICE_REFERENCE` | `neuro.wav` | 음색 참조 |

`Neuro.yaml` 은 현재 문서/참조용에 가깝고 실제 시스템 프롬프트는 `constants.py:SYSTEM_PROMPT` 가 로드됨.

---

## 11. 성숙도 / 한계

**장점**:
- 완전 로컬, 외부 API 불필요 (text-generation-webui 포함)
- ChromaDB 반성형 장기 메모리 → 드문 구현
- 모듈 인젝션 + priority 기반 프롬프트 조립 → 재사용 가능한 패턴
- XTTS-v2 음색 복제 + Socket.io 실시간 UI

**한계**:
- 사용자가 말하는 중에도 AI 가 계속 발화 (인터럽트 미구현)
- async 루프 안의 blocking requests.post (Memory 모듈)
- 런타임 모듈 on/off 불가
- Socket.io 포트·시스템 프롬프트·스레드 슬립 등 하드코딩
- 문장 단위 TTS 스트리밍 없음 → 긴 응답 체감 지연
- 오류 복구 부재 (LLM endpoint 다운 시 행)

---

## 12. zemory-sama 가 차용할 패턴

1. **ChromaDB + reflection** — 장기 메모리 구현 시 거의 그대로 이식 가능
2. **Priority 기반 prompt injection** — 모듈 추가 시 우선순위만 지정
3. **Patience 타임아웃 자율 발화** — 방송형 콘텐츠 확장 시 필수
4. **Signals property setter → 이벤트 브로드캐스트** — Python asyncio 환경에서도 간단히 모방 가능

**반면교사**:
1. 블로킹 STT 루프 → 타임아웃·재시도 필요
2. async 안의 blocking call → `asyncio.to_thread` 로 반드시 래핑
3. 하드코딩된 constants → YAML 핫리로드로 전환
4. 인터럽트 생략 → `human_speaking → TTS.abort + LLM.cancel` 능동 체인 필수

---

## 레퍼런스 코드 위치

```
_ref/Neuro/
├── main.py                     # 엔트리 + 스레드 기동
├── signals.py                  # 전역 상태 허브 (setter → sio_queue)
├── stt.py                      # RealtimeSTT 래핑
├── tts.py                      # RealtimeTTS + CoquiEngine
├── prompter.py                 # 100ms 폴링, PATIENCE 자율 발화
├── socketioServer.py           # 포트 8080 이벤트 허브
├── constants.py                # 전 설정
├── llmWrappers/
│   ├── abstractLLMWrapper.py   # 인젝션 + 토큰 예산 + SSE 스트림
│   ├── textLLMWrapper.py
│   ├── imageLLMWrapper.py      # 스크린샷 + Vision LLM
│   └── llmState.py
├── modules/
│   ├── module.py               # 베이스 클래스
│   ├── injection.py            # priority 시스템
│   ├── memory.py               # ChromaDB + reflection
│   ├── twitchClient.py
│   ├── vtubeStudio.py          # pyvts, 액션 큐
│   ├── customPrompt.py
│   ├── multiModal.py
│   └── audioPlayer.py
├── memories/
│   ├── chroma.db               # 벡터 DB
│   └── memoryinit.json         # 부트스트랩 Q&A
└── voices/
    └── neuro.wav               # XTTS-v2 참조 음성
```
