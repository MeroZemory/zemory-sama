# Open-LLM-VTuber 대화 파이프라인 분석

> GitHub: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber (6.8k stars)
> 분석일: 2026-04-13

## 아키텍처 개요

웹 기반(FastAPI + WebSocket) 구조로, 프런트엔드(브라우저)가 마이크/스피커를 제어하고 백엔드가 VAD/ASR/LLM/TTS를 처리한다. 프런트엔드와 백엔드가 물리적으로 분리된 오디오 채널을 사용하므로 에코 문제가 구조적으로 방지된다.

---

## 1. VAD (Voice Activity Detection)

### 구현체

Silero VAD 단일 구현. PyTorch 기반 신경망 추론.

- 소스: `src/open_llm_vtuber/vad/silero.py`
- 인터페이스: `src/open_llm_vtuber/vad/vad_interface.py`
- 팩토리: `src/open_llm_vtuber/vad/vad_factory.py`
- 설정: `src/open_llm_vtuber/config_manager/vad.py`

### 상태 머신

```
IDLE ──(3연속 hit, 96ms)──→ ACTIVE ──(24연속 miss, 768ms)──→ INACTIVE
  ↑                                                             │
  │         ┌──(3연속 hit, 음성 재개)──────────────────────────────┘
  │         ↓
  │      ACTIVE (다시)
  │         │
  └─────────┴──(24연속 miss, 768ms)──→ IDLE
```

상태 열거 (`silero.py:78-82`):
```python
class State(Enum):
    IDLE = 1       # 대기: 음성 없음
    ACTIVE = 2     # 활성: 음성 감지 중
    INACTIVE = 3   # 비활성: 침묵 감지 중 (아직 종료 확정 아님)
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `orig_sr` | 16000 | 입력 샘플레이트 |
| `target_sr` | 16000 | 처리 샘플레이트 |
| `prob_threshold` | 0.4 | Silero 모델의 음성 확률 임계치 |
| `db_threshold` | 60 | 최소 음량 (dB) |
| `required_hits` | 3 | 발화 시작 확인에 필요한 연속 프레임 수 |
| `required_misses` | 24 | 침묵 확인에 필요한 연속 프레임 수 |
| `smoothing_window` | 5 | 확률/dB 롤링 평균 윈도우 크기 |

### 타이밍

- 프레임 크기: 512 samples @ 16kHz = **32ms/프레임**
- 발화 시작 감지: 3 × 32ms = **96ms**
- 침묵 확인 (ACTIVE→INACTIVE): 24 × 32ms = **768ms**
- 침묵 확인 (INACTIVE→IDLE): 24 × 32ms = **768ms**
- Pre-buffer: 20프레임 = **640ms** (발화 감지 전 오디오 보존)
- 최소 발화 길이: 30프레임 = **960ms** (이하는 무시)

### 거짓 양성 방지 (5단계)

1. **이중 임계치**: prob≥0.4 AND dB≥60 동시 충족 필요
2. **스무딩**: 5프레임 롤링 평균으로 순간 노이즈 스파이크 제거
3. **연속 프레임 요구**: 3연속 hit 필요 (단발성 노이즈 차단)
4. **연속 miss 요구**: 24연속 miss 필요 (짧은 묵음에 상태 전환 방지)
5. **최소 발화 길이**: 30프레임(960ms) 미만 음성은 폐기

### 시그널 프로토콜

- `<|PAUSE|>`: IDLE→ACTIVE 전환 시 발생. 프런트엔드에 "interrupt" 전송.
- `<|RESUME|>`: INACTIVE→IDLE 전환 시 발생. 수집된 오디오를 ASR로 전달.

WebSocket 핸들러에서의 시그널 처리 (`websocket_handler.py:489-511`):
```python
for audio_bytes in context.vad_engine.detect_speech(chunk):
    if audio_bytes == b"<|PAUSE|>":
        await websocket.send_text(
            json.dumps({"type": "control", "text": "interrupt"})
        )
    elif audio_bytes == b"<|RESUME|>":
        pass  # 별도 동작 없음
    elif len(audio_bytes) > 1024:
        # 실제 음성 데이터 → 버퍼에 추가
        self.received_data_buffers[client_uid] = np.append(...)
        await websocket.send_text(
            json.dumps({"type": "control", "text": "mic-audio-end"})
        )
```

---

## 2. 대화 흐름 (Conversation Flow)

### 전체 시퀀스

```
[1] 사용자 발화 → VAD 감지 → 오디오 버퍼링
[2] 발화 종료 → "mic-audio-end" → conversation trigger
[3] conversation-chain-start 시그널
[4] ASR 텍스트 변환
[5] LLM 스트리밍 응답 → 문장별 TTS 병렬 생성
[6] TTS 오디오를 시퀀스 순서로 프런트엔드 전송
[7] backend-synth-complete 시그널
[8] ⏸️ frontend-playback-complete 대기 (무기한 블로킹)
[9] force-new-message → conversation-chain-end 시그널
[10] 시스템 준비 완료 → 다음 턴
```

### 시그널 테이블

| 시그널 | 방향 | 시점 | 목적 |
|--------|------|------|------|
| `conversation-chain-start` | 백엔드→프런트 | 턴 시작 | UI에 대화 시작 알림 |
| `full-text: "Thinking..."` | 백엔드→프런트 | 턴 시작 | 처리 중 표시 |
| `user-input-transcription` | 백엔드→프런트 | ASR 완료 후 | 인식된 텍스트 표시 |
| `audio` (payload) | 백엔드→프런트 | TTS 완료 시 | 오디오 + 볼륨 데이터 |
| `backend-synth-complete` | 백엔드→프런트 | 모든 TTS 완료 | 재생 시작 허가 |
| `frontend-playback-complete` | 프런트→백엔드 | 오디오 재생 완료 | 다음 턴 허가 |
| `force-new-message` | 백엔드→프런트 | 턴 종료 전 | UI 초기화 |
| `conversation-chain-end` | 백엔드→프런트 | 턴 종료 | 대화 턴 완료 |
| `interrupt` | 백엔드→프런트 | VAD PAUSE | AI 음성 중단 |

### 프런트엔드 재생 완료 대기 (핵심 동기화)

`conversation_utils.py:162-190`:
```python
async def finalize_conversation_turn(...) -> None:
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        # 핵심: 프런트엔드 확인까지 무기한 대기
        response = await message_handler.wait_for_response(
            client_uid, "frontend-playback-complete"
        )

    await websocket_send(json.dumps({"type": "force-new-message"}))
```

`message_handler.py:16-54`:
```python
async def wait_for_response(self, client_uid, response_type, ...):
    event = asyncio.Event()
    self._response_events[client_uid][response_key] = event
    await event.wait()  # 타임아웃 없음 — 프런트엔드 응답까지 대기
    return self._response_data[client_uid].pop(response_key, None)
```

---

## 3. TTS 관리 (TTSTaskManager)

### 병렬 생성 + 순서 보장

소스: `src/open_llm_vtuber/conversations/tts_manager.py`

각 문장에 시퀀스 넘버를 부여하고, 병렬로 TTS를 생성하되 전달은 순서대로.

```python
class TTSTaskManager:
    def __init__(self):
        self.task_list: List[asyncio.Task] = []
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
```

시퀀스 부여 (`tts_manager.py:70-71`):
```python
current_sequence = self._sequence_counter
self._sequence_counter += 1
```

순서 보장 전송 (`tts_manager.py:92-114`):
```python
async def _process_payload_queue(self, websocket_send):
    buffered_payloads: Dict[int, Dict] = {}
    while True:
        payload, sequence_number = await self._payload_queue.get()
        buffered_payloads[sequence_number] = payload
        # 다음 시퀀스가 준비되면 순서대로 전송
        while self._next_sequence_to_send in buffered_payloads:
            next_payload = buffered_payloads.pop(self._next_sequence_to_send)
            await websocket_send(json.dumps(next_payload))
            self._next_sequence_to_send += 1
```

예시:
```
TTS #0 (3초 소요) → 세 번째로 완료
TTS #1 (1초 소요) → 첫 번째로 완료 → buffered_payloads[1]에 저장, 대기
TTS #2 (2초 소요) → 두 번째로 완료 → buffered_payloads[2]에 저장, 대기
TTS #0 완료 → 전송: #0 → #1 → #2 (연속)
```

---

## 4. 인터럽트 (Interruption)

사용자가 AI 발화 중 말을 시작하면:

`conversation_handler.py:112-143`:
```python
async def handle_individual_interrupt(client_uid, current_conversation_tasks, ...):
    task = current_conversation_tasks[client_uid]
    if task and not task.done():
        task.cancel()  # 전체 대화 태스크 취소

    context.agent_engine.handle_interrupt(heard_response)  # LLM에 알림

    # 히스토리에 인터럽트 기록
    store_message(..., role="ai", content=heard_response)
    store_message(..., role="system", content="[Interrupted by user]")
```

- `asyncio.Task.cancel()`로 대화 파이프라인 전체 취소
- `finally` 블록에서 TTS 정리 (`cleanup_conversation`)
- LLM 컨텍스트에 인터럽트 사실 전달 (히스토리 보존)

---

## 5. 에코 방지 전략

웹 아키텍처의 구조적 장점:
- 마이크 입력: 프런트엔드 → WebSocket → 백엔드 (업스트림)
- 스피커 출력: 백엔드 → WebSocket → 프런트엔드 (다운스트림)
- 두 채널이 물리적으로 분리되어 서버에서 에코가 발생하지 않음
- 프런트엔드(브라우저)의 내장 에코 캔슬레이션 활용 가능
- "voice interruption without headphones" 기능 — AI가 자신의 음성을 듣지 않음
