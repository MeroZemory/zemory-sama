# AIRIS-VtuberAI (neurokitti) 대화 파이프라인 분석

> GitHub: https://github.com/neurokitti/AIRIS-VtuberAI (145 stars)
> 마지막 커밋: 2026-01 (활발, v2 개발 중)
> 분석일: 2026-04-15

## 아키텍처 개요

**완전 로컬** 추론 지향 Python 시스템. 각 관심사를 단일 `*_API.py` 파일에 분리하는 플랫한 모듈 컨벤션이 특징. 외부 생성 API(OpenAI/Anthropic 등) 호출은 **없다** — 모든 LLM/STT/TTS 추론이 로컬 NVIDIA GPU 에서 실행된다. (단, Twitch/YouTube 채팅 수신은 네트워크 필요.)

**엔트리** (`main.py:1-33`): 3개 모드 메뉴 → `startup_scripts.py` 의 파이프라인 오케스트레이터로 위임.

```
1. Chat Twitch     → main_chat_twitch_non_legacy()
2. Chat YouTube    → main_chat_youtube_non_legacy()
3. Interview (mic) → main_interview_non_legacy()
```

---

## 1. 녹음 (record_API.py)

**엔진**: `record_engine` (lines 6-35)
- PyAudio 44.1 kHz·mono·16-bit, 1024-sample chunk
- **트리거**: `keyboard.is_pressed("space")` — push-to-talk
- WAV 파일로 출력
- **VAD 없음**. Open-LLM-VTuber 의 상태 머신, airi 의 브라우저 Silero 와 달리 순수 수동 제어.

---

## 2. STT (Faster-Whisper)

**엔진**: `transcription_API.py:6-29`
- `faster-whisper` + CUDA
- 기본 모델 `distil-large-v3`, `compute_type="int8_float16"` 양자화
- **비스트리밍**: 파일 완료 후 일괄 전사 (segment 연결)
- 호출부: `startup_scripts.py:126` `transcription_instance.whisper(output_file)`

---

## 3. LLM (transformers + TextIteratorStreamer)

**엔진**: `chat_API.py:13-101` (`neo_chat_engine`)

- `transformers.AutoModelForCausalLM.from_pretrained()` 직접 사용 (Ollama/llama.cpp 아님)
- 기본 데모 모델: `meta-llama/Meta-Llama-3-8B-Instruct`
- 양자화 옵션: `load_in_4bit=True` / `load_in_8bit=True` (bitsandbytes)
- **컨텍스트 관리** (`chat_API.py:40`): 최근 N 턴 절단만 (`live_mem = transcript[-self.mem_length:]`) — 슬라이딩 윈도우·요약 없음. 모드별 `mem_length` 기본 3–5
- 시스템 프롬프트: `system_message.txt` 또는 `system_message_interview.txt` 파일 로드, 없으면 하드코딩 fallback
- 토크나이저의 `apply_chat_template()` 사용 (`chat_API.py:46`)

### 토큰 스트리밍 (`chat_API.py:49-74`)

```python
def thread_generate(self, input_ids, max_new_tokens):
    self.model.generate(..., streamer=self.streamer)
```

`TextIteratorStreamer` 를 데몬 스레드에서 소비 → `streamer_collector()` 제너레이터로 토큰을 yield. 호출부 (`startup_scripts.py:43-44`) 는 토큰을 누적하다가 구두점(`['?','!','.',':']`) 발견 시 문장 단위로 TTS 합성을 **즉시** 트리거 → 지연 체감 감소.

**미구현**: 프롬프트 캐시 / speculative decoding / KV 캐시 최적화.

---

## 4. TTS (OpenVoice, 블로킹)

**엔진**: `speach_API.py:6-29` (`speach_engine`)

- **OpenVoice** 를 `requirements.txt` 밖의 git clone 으로 별도 설치. 하드코딩 경로 `OpenVoice/checkpoints/base_speakers/EN` + `converter`
- 구조: Base speaker TTS + Tone Color Converter (2단)
- 보이스 클로닝: `.mp3` 참조에서 speaker embedding 추출 (line 23 주석 처리)
- 출력: WAV 디스크 저장 → `winsound.PlaySound(TTS_OUTPUT, SND_FILENAME)` 재생 (**Windows 전용**, `startup_scripts.py:51, 100, 146`)

**결정적 제약**: 블로킹 합성 — 각 문장의 TTS 완료까지 대기 후 다음 문장 진행. 병렬/스트리밍 TTS 없음.

---

## 5. 턴테이킹·에코

### Chat 모드 (Twitch/YouTube)
카운터 기반:

```python
while True:
    chat_messages = twitch_instance.get_messages()
    if len(chat_messages) > responded_messages:
        # LLM 스트림 → 구두점마다 TTS → 재생
        responded_messages += 1
```

**에코 불필요** — 입력이 텍스트. 인터럽트도 없음.

### Interview 모드
완전 순차 half-duplex:

```
Space 누름 → 녹음 → 릴리즈 → Whisper → LLM → TTS → 재생 → 반복
```

VTuber 가 말하는 동안 사용자 음성은 감지 불가(VAD 없음). 자연스러운 대화보다 **인터뷰형 턴 제어** 에 적합.

---

## 6. 스트리밍 통합

### Twitch_API.py
- `twitchio.ext.commands.Bot` 상속 (`Bot` 클래스, 5-23 lines)
- `event_message()` 에서 `{role, name, text}` 를 `messages` 리스트에 append
- **레이트 리밋·중복 제거·큐 한도 없음** — 전체 메시지 무한 버퍼

### Youtube_API.py
- `pytchat` (비공식 YouTube chat scraper)
- 정규식으로 video ID 추출 (여러 URL 포맷 대응, lines 20-26)
- 백그라운드 스레드 (`chat_streamer_thread`) 폴링
- async 아님

### OBS_API.py
- `obs-websocket-py` — `update_browser_source()`, `update_text()` (GDI+ 텍스트 캡션)
- `@OBS_Wrapper` 데코레이터로 조용히 예외 삼킴 (fire-and-forget)
- **현재 startup_scripts.py 에서 호출 주석 처리됨** — 미활성

---

## 7. 시스템 프롬프트

### `system_message.txt` (Chat)
"Hannah" — 활기찬 걸넥스트도어 VTuber 페르소나. 게임/요리/그림/노래 관심사. 대략 450 토큰.

### `system_message_interview.txt` (Interview)
Chat 버전 + 첫 줄 강조: `"DO NOT SAY YOU ARE A AI CHAT BOT."` — 몰입 유지용 safeguard.

모드별로 톤 제어·현재 스트림 그라운딩·역할 분리 없음 — 단일 프롬프트를 전체에 적용.

---

## 8. 프로파니티 필터 (utils.py)

`utils.censor()` (lines 29-61) 다단 필터:
1. 따옴표·`*...*` 패턴 제거
2. 소문자/Capitalized/UPPERCASE/앞공백 변형 생성
3. `better_profanity` 에 커스텀 단어목록 로드
4. 매치 시 `[Censored]` 치환

난독화 우회(`f u c k`, `Fuck`, `FUCK`) 에 대한 기본적 방어.

---

## 9. 성숙도

**장점**:
- **진정한 로컬**: OpenAI 등 외부 API 호출 부재
- 관심사별 단일 파일 → 읽기 쉬움
- 모델/양자화/메모리 길이/시스템 프롬프트가 파라미터화됨
- 문장 단위 TTS 조기 트리거로 체감 지연 감소
- Twitch + YouTube + OBS 배선 존재

**한계**:
- 타이포(`evryone`, `synthisise`), 백슬래시 경로(`OpenVoice\`), 주석 처리된 죽은 코드 — 전반적 코드 품질 저
- VAD 미구현: interview 는 push-to-talk
- 블로킹 TTS: 파이프라인 stall
- 동시성 없음: 채팅 폴링 외에는 스레드 미활용
- OpenVoice 경로 하드코딩 → 설정 불가
- `winsound` → Windows 전용 재생
- 무한 `messages` 버퍼 → 장시간 실행 시 메모리 누수
- README 의 레이턴시 벤치마크 대부분 "tbd"
- v2 에서 interrupt·tool calling·Ollama 지원 예고 → 현재 v1 은 부재

---

## 10. "완전 로컬" 주장 검증

| 항목 | 로컬? |
|------|-------|
| LLM 추론 | ✅ (HF 초기 다운로드 후) |
| TTS (OpenVoice) | ✅ |
| STT (Faster-Whisper) | ✅ |
| Twitch/YouTube 채팅 수신 | ❌ (네트워크 필요) |
| 외부 생성 API (OpenAI 등) | ✅ 사용 안 함 |

**결론**: 채팅 수신을 제외하면 주장대로 완전 로컬. 모델 캐시된 상태에서 interview 모드는 완전 오프라인 가능.

---

## 11. zemory-sama 가 차용할 패턴

1. **문장 단위 조기 TTS 트리거** — LLM 토큰을 누적하다 구두점 발견 시 즉시 합성 시작. 현재 zemory-sama 순차 큐에 직접 적용 가능 (구현 난이도 낮음).
2. **시스템 프롬프트 파일 분리** (`system_message.txt`) — constants 대신 외부 텍스트로 핫스왑
3. **관심사별 단일 모듈 컨벤션** — 현재 `zemory_vad/` 도 유사한 방향이지만 더 일관되게
4. **프로파니티 변형 생성 필터** — 한국어 방송 시 커스텀 블랙리스트 적용 가능

**반면교사**: 하드코딩 경로, 블로킹 TTS, `winsound`, 무한 버퍼 — 모두 zemory-sama 가 이미 피한 설계.

---

## 레퍼런스 코드 위치

```
_ref/AIRIS-VtuberAI/
├── main.py                      # 모드 선택 메뉴 (33 lines)
├── startup_scripts.py           # 3개 모드 오케스트레이션 (153 lines)
├── chat_API.py                  # transformers LLM, TextIteratorStreamer (102)
├── speach_API.py                # OpenVoice TTS (29)
├── transcription_API.py         # Faster-Whisper (30)
├── record_API.py                # PyAudio, 스페이스바 트리거 (36)
├── Twitch_API.py                # twitchio (24)
├── Youtube_API.py               # pytchat (41)
├── OBS_API.py                   # obs-websocket-py (34, 현재 미호출)
├── utils.py                     # 프로파니티 필터 + 파일 I/O (62)
├── system_message.txt           # Chat 페르소나
├── system_message_interview.txt # Interview 페르소나 + AI 부정 지시
└── requirements.txt             # faster-whisper, transformers, bitsandbytes 등 26개
```
