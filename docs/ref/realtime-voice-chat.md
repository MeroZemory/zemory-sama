# RealtimeVoiceChat 대화 파이프라인 분석

> GitHub: https://github.com/KoljaB/RealtimeVoiceChat
> 저자: KoljaB (RealtimeTTS/RealtimeSTT 개발자)
> 분석일: 2026-04-13

## 아키텍처 개요

FastAPI + WebSocket 기반. 클라이언트가 48kHz PCM을 전송하면 서버에서 16kHz로 리샘플링 후 STT → LLM → TTS 파이프라인을 실행한다. 반이중(half-duplex) 방식으로 한 번에 한 쪽만 발화한다.

---

## 1. 반이중 상태 머신

### `interrupted` 플래그

마이크 차단의 핵심 메커니즘. 단일 boolean 플래그로 전체 시스템의 오디오 입력을 게이트한다.

```
MIC OPEN (interrupted=False)
    │
    ├─ on_before_final() 호출 (사용자 턴 종료)
    │     ↓
    │  MIC BLOCKED (interrupted=True, interruption_time=now)
    │     │
    │     ├─ LLM 처리 + TTS 생성
    │     ├─ 첫 TTS 청크 WebSocket 전송
    │     │     ↓
    │     │  _reset_interrupt_flag_async() 태스크 생성
    │     │     ↓
    │     │  1초 대기 후 → MIC OPEN (interrupted=False)
    │     │
    │     └─ 백업: 2초 초과 시 강제 해제 (send_tts_chunks 내)
    │
    └─ on_recording_start() (사용자가 TTS 재생 중 발화)
          ↓
       TTS 중단 + 생성 abort → MIC OPEN
```

### 플래그 설정/해제 위치

**설정 (True)** — `server.py:685-688`:
```python
if not self.app.state.AudioInputProcessor.interrupted:
    self.app.state.AudioInputProcessor.interrupted = True
    self.interruption_time = time.time()
```

**해제 (False)** — `server.py:367-371`:
```python
async def _reset_interrupt_flag_async(app, callbacks):
    await asyncio.sleep(1)  # 1초 대기
    if app.state.AudioInputProcessor.interrupted:
        app.state.AudioInputProcessor.interrupted = False
        callbacks.interruption_time = 0
```

**백업 해제** — `server.py:400-403`:
```python
if (app.state.AudioInputProcessor.interrupted
    and callbacks.interruption_time
    and time.time() - callbacks.interruption_time > 2.0):
    app.state.AudioInputProcessor.interrupted = False
```

### 오디오 게이팅

`audio_in.py:200-205`:
```python
if not self.interrupted:
    if not self._transcription_failed:
        self.transcriber.feed_audio(processed.tobytes(), audio_data)
# interrupted=True일 때: 오디오는 처리(리샘플링 등)되지만 STT에 전달되지 않음
```

---

## 2. ML 기반 턴 감지 (TurnDetection)

소스: `turndetect.py`

### DistilBERT 문장 완성 분류기

모델: `KoljaB/SentenceFinishedClassification`

```python
# turndetect.py:218-221
self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(model_dir)
self.classification_model = transformers.DistilBertForSequenceClassification.from_pretrained(model_dir)
self.classification_model.eval()

# 추론 (turndetect.py:322-374)
inputs = self.tokenizer(sentence, return_tensors="pt", max_length=128)
with torch.no_grad():
    outputs = self.classification_model(**inputs)
probabilities = F.softmax(outputs.logits, dim=1).squeeze().tolist()
prob_complete = probabilities[1]  # 0.0~1.0 문장 완성 확률
```

LRU 캐시 (256 엔트리)로 동일 텍스트 재추론 방지.

### 동적 Pause 계산

최종 공식:
```
final_pause = (0.65 × punctuation_pause + 0.35 × model_pause) × speed_factor
if 말줄임표: final_pause += 0.2
final_pause = max(final_pause, pipeline_latency + overhead)
```

#### 구두점 기반 Pause (`turndetect.py:376-401`)

| 구두점 | 기본 pause | 설명 |
|--------|-----------|------|
| `...` | 2.3초 | 말줄임표 (사용자가 더 말할 가능성) |
| `.` | 0.39초 | 마침표 |
| `!` | 0.35초 | 느낌표 |
| `?` | 0.33초 | 물음표 |
| 없음 | 1.25초 | 구두점 없이 끊긴 경우 |

#### ML 모델 Pause (선형 보간)

```python
# turndetect.py:129-167
anchor_points = [
    (0.0, 1.0),  # 완성 확률 0% → 1.0초 대기
    (1.0, 0.0),  # 완성 확률 100% → 0.0초 대기
]
```

#### Speed Factor 설정 (`turndetect.py:255-296`)

| 파라미터 | fast (0.0) | very_slow (1.0) |
|---------|-----------|----------------|
| detection_speed | 0.5 | 1.7 |
| ellipsis_pause | 2.3 | 3.0 |
| punctuation_pause | 0.39 | 0.9 |
| exclamation_pause | 0.35 | 0.8 |
| question_pause | 0.33 | 0.8 |
| unknown_pause | 1.25 | 1.9 |

클라이언트에서 speed_factor (0-100) 슬라이더로 실시간 조절 가능.

### 침묵 모니터 (Silence Monitor)

`transcribe.py:235-318` — 백그라운드 데몬 스레드

3단계 트리거:
1. **잠재 문장 종료** (`time_since_silence > potential_sentence_end_time`): 문장 끝 감지 + LLM 준비 시작
2. **TTS 합성 허가** (`time_since_silence > silence_duration - 0.25s`): TTS 워커에 시작 시그널
3. **"HOT" 상태** (`time_since_silence > silence_duration - 0.35s`): 최종 전사 임박 알림

---

## 3. Quick Answer + Final Answer 파이프라인

소스: `speech_pipeline_manager.py`

### 2단계 TTS 전략

사용자 체감 지연을 최소화하기 위해 LLM 응답을 두 단계로 나누어 처리:

**Quick Answer** (`speech_pipeline_manager.py:534-641`):
- LLM 출력에서 첫 문장 경계를 감지
- 8바이트 청크로 작게 분할하여 최소 지연 전송
- `tts_quick_allowed_event`로 시작 타이밍 제어

**Final Answer** (`speech_pipeline_manager.py:643-768`):
- Quick Answer 이후 나머지 텍스트
- 30바이트 청크로 처리량 최적화
- Quick Answer의 overhang(잘린 나머지)를 먼저 처리

```
LLM 스트리밍: "안녕하세요. 저는 제모리입니다. 오늘 날씨가 좋네요."
                    ↓
Quick Answer: "안녕하세요." → TTS (8byte 청크, 즉시 전송)
                    ↓
Final Answer: "저는 제모리입니다. 오늘 날씨가 좋네요." → TTS (30byte 청크)
                    ↓
같은 audio_chunks 큐로 병합 → 끊김 없는 재생
```

### 생성 상태 관리 (RunningGeneration)

`speech_pipeline_manager.py:66-111`:
```python
class RunningGeneration:
    # LLM 상태
    llm_generator = None
    llm_finished: bool = False
    llm_finished_event = threading.Event()

    # Quick Answer 상태
    quick_answer: str = ""
    quick_answer_provided: bool = False
    quick_answer_first_chunk_ready: bool = False
    tts_quick_started: bool = False
    tts_quick_allowed_event = threading.Event()

    # Final Answer 상태
    tts_final_started: bool = False
    final_answer: str = ""

    # 오디오 큐 (Quick + Final 공유)
    audio_chunks = Queue()

    # 중단 플래그
    audio_quick_aborted: bool = False
    audio_final_aborted: bool = False
    completed: bool = False
```

---

## 4. 오디오 전송 파이프라인

### TTS → WebSocket → 클라이언트

`server.py:376-511` (`send_tts_chunks`):

```
audio_module.synthesize()
    ↓ on_audio_chunk 콜백
audio_chunks Queue (공유)
    ↓ get_nowait()
Upsampler.get_base64_chunk(chunk)  # 24kHz→48kHz 업샘플링 + base64
    ↓
message_queue.put_nowait({"type": "tts_chunk", "content": base64_audio})
    ↓
send_text_messages() → ws.send_json()
```

### 오디오 버퍼링 로직 (`audio_module.py:273-364`)

```python
def on_audio_chunk(chunk):
    buffer.append(chunk)
    buf_dur += play_duration

    if buffering:
        # 초기 버퍼링: 2연속 "good" 청크 또는 0.5초 축적 시 플러시
        if good_streak >= 2 or buf_dur >= 0.5:
            for c in buffer:
                audio_chunks.put_nowait(c)
            buffering = False
    else:
        audio_chunks.put_nowait(chunk)  # 직접 큐잉
```

---

## 5. 인터럽트 처리

### 사용자가 TTS 재생 중 발화 시

`server.py:774-814` (`on_recording_start`):
```python
def on_recording_start(self):
    if self.tts_client_playing:
        self.tts_to_client = False         # TTS 스트리밍 중단
        self.user_interrupted = True
        self.send_final_assistant_answer(forced=True)
        self.tts_chunk_sent = False

        self.message_queue.put_nowait({"type": "stop_tts", "content": ""})
        self.abort_generations("user interrupts")
        self.message_queue.put_nowait({"type": "tts_interruption", "content": ""})
```

### Abort 체인

```
abort_generations() 호출
    ↓
SpeechPipelineManager.abort_generation(wait_for_completion=True)
    ↓
process_abort_generation():
    ├─ generation.abortion_started = True
    ├─ stop_llm_request_event.set()     → LLM 워커 중단
    ├─ stop_tts_quick_request_event.set() → Quick TTS 중단
    ├─ stop_tts_final_request_event.set() → Final TTS 중단
    ├─ llm_generator.close()
    └─ running_generation = None
```

---

## 6. 핵심 타이밍 파라미터

### STT 설정 (`transcribe.py:25-52`)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `post_speech_silence_duration` | 0.7초 | 발화 후 침묵 대기 시간 (동적 조절됨) |
| `min_length_of_recording` | 0.5초 | 최소 녹음 길이 |
| `silero_sensitivity` | 0.05 | Silero VAD 민감도 (낮을수록 민감) |
| `webrtc_sensitivity` | 3 | WebRTC VAD 레벨 |
| `realtime_processing_pause` | 0.03초 | 실시간 전사 업데이트 주기 |

### 침묵 모니터 상수 (`transcribe.py:79-92`)

| 상수 | 값 | 설명 |
|------|-----|------|
| `_PIPELINE_RESERVE_TIME_MS` | 0.02초 | 파이프라인 안전 버퍼 |
| `_HOT_THRESHOLD_OFFSET_S` | 0.35초 | HOT 상태 진입 오프셋 |
| `_TTS_ALLOWANCE_OFFSET_S` | 0.25초 | TTS 시작 허가 오프셋 |
| `_MIN_HOT_CONDITION_DURATION_S` | 0.15초 | 최소 HOT 지속 시간 |

### 오디오 청크 크기 (`audio_module.py`)

| 용도 | 청크 크기 | 목적 |
|------|----------|------|
| Quick Answer | 8 bytes | 최소 지연 |
| Final Answer | 30 bytes | 처리량 최적화 |

---

## 7. 콜백 체인 (전체 흐름)

```
클라이언트 PCM (48kHz)
    ↓ WebSocket binary
process_incoming_data()
    ↓ resample 48kHz→16kHz
AudioInputProcessor.process_chunk_queue()
    ↓ interrupted 체크
transcriber.feed_audio()
    ↓
Silence Monitor Thread
    ├→ detect_potential_sentence_end()
    │     ↓
    │   TurnDetection._text_worker() [DistilBERT 추론]
    │     ↓
    │   on_new_waiting_time() → post_speech_silence_duration 동적 갱신
    │
    ├→ on_tts_allowed_to_synthesize()
    │
    └→ potential_full_transcription_callback() ["HOT" 상태]
          ↓
on_before_final()
    ├→ interrupted = True [마이크 차단]
    ├→ tts_to_client = True [TTS 스트리밍 허가]
    └→ tts_quick_allowed_event.set()
          ↓
_tts_quick_inference_worker() → audio.synthesize() → audio_chunks
          ↓
send_tts_chunks() → Upsampler → message_queue
          ↓
send_text_messages() → ws.send_json({"type": "tts_chunk", ...})
          ↓
_reset_interrupt_flag_async() → 1초 후 interrupted = False
```
