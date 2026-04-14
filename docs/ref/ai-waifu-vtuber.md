# AI-Waifu-Vtuber (ardha27) 대화 파이프라인 분석

> GitHub: https://github.com/ardha27/AI-Waifu-Vtuber (1.1k stars)
> 마지막 커밋: 2023-11-12 (약 5개월 이상 정체)
> 분석일: 2026-04-15

## 아키텍처 개요

Python 단일 파일 스크립트(`run.py`, 267 lines) + `utils/` 헬퍼. **채팅-주도** 파이프라인으로, 음성 품질보다 **Twitch/YouTube 라이브챗 응답**에 최적화되어 있다. VAD·인터럽트·비동기 I/O 가 모두 없으며, `is_Speaking` 전역 플래그와 `time.sleep(1)` 폴링으로 턴을 조절한다.

**모드 선택** (`run.py:238-266`):

```
Mode 1 - Mic:       RIGHT_SHIFT push-to-talk → Whisper API
Mode 2 - Youtube:   pytchat → LLM
Mode 3 - Twitch:    IRC socket → LLM
```

모든 모드가 동일한 `preparation` 루프(`run.py:225-236`)로 수렴 — `is_Speaking==False && chat != chat_prev` 시 `openai_answer()` 호출.

---

## 1. STT (OpenAI Whisper API)

모드 1 에서만 동작. PyAudio 로 44.1 kHz·mono 녹음(`input.wav`) 후 `openai.Audio.transcribe("whisper-1", ...)` 호출. **VAD 없음** — 사용자가 RIGHT_SHIFT 를 누르는 동안의 전체 오디오가 한 파일로 캡처된다. 노이즈 필터링·프레임 버퍼 없음.

---

## 2. LLM (GPT-3.5-turbo)

```python
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    max_tokens=128, temperature=1, top_p=0.9)
```

- `promptMaker.py:13-48` 에서 `characterConfig/Pina/identity.txt` 아이덴티티 + 히스토리(총 4000자 이하로 pruning) + "20자 이내 응답" 제약 조립
- 로컬 LLM / 다른 provider 라우팅 없음
- `conversation.json` 에 평문 히스토리 디스크 저장 (매 턴 전체 로드/세이브)

---

## 3. TTS (VoiceVox + Silero)

### VoiceVox (일본어, 기본)
- `utils/TTS.py:27-41`
- Docker/Colab 로컬 서비스 `http://localhost:50021`
- 2단계: `POST /audio_query` → `POST /synthesis?speaker=46&enable_interrogative_upspeak=True`
- 옵션 `katakana_converter()` (MeCab) 로 영단어를 가타카나 변환 후 합성
- 50+ 스피커 정의 `speaker.json`
- `winsound.PlaySound()` 로 재생 (**Windows 전용**)

### Silero (다국어, 선택)
- 기본 주석 처리 (line 206), PyTorch 자체 호스팅 모델
- 영어·러시아어·프랑스어·스페인어·독일어·인도어 지원

병렬 합성 없음 — 단일 `winsound.PlaySound()` 블로킹. `is_Speaking` 플래그로 중첩 차단.

**아바타 립싱크**: VTube Studio API 직접 호출이 아니라, 데스크톱 오디오 → **VB-Cable 가상 케이블** → VTube Studio 마이크 입력으로 루프백 → 볼륨 기반 입 파라미터 (README:100). 프래질하지만 구현이 간단.

---

## 4. VAD·에코·인터럽트

- **VAD 없음**: push-to-talk 만 지원
- **에코 방지 불필요**: 모드 2/3 은 텍스트 입력, 모드 1 은 모드 전환으로 시분할
- **인터럽트 없음**: `is_Speaking==True` 동안 모든 신규 메시지 블락

---

## 5. 라이브챗 통합

### YouTube (`run.py:122-145`)
`pytchat` 라이브러리. 블랙리스트(`["Nightbot", "streamelements"]`) 필터, `!` 접두어 명령 무시, `emoji.demojize()` 로 이모지 제거. 포맷: `"[username] berkata [message]"` (인도네시아어 *berkata*="said").

### Twitch (`run.py:147-180`)
**원시 IRC 소켓** — 라이브러리 없이 `socket.socket()` 직접 구동. OAuth 토큰은 `utils/twitch_config.py` 평문. PING/PONG 수동 처리, `PRIVMSG` regex 파싱.

```python
regex = r":(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)"
```

### 메시지 선택 로직
양쪽 모두 **전역 변수 `chat` 에 최신 메시지만 덮어씌우는 방식**. 1초 내 여러 메시지 도착 시 중간 것들은 **유실**. 큐/버퍼 없음.

---

## 6. 캐릭터 설정 — 경량

디렉터리별 한 캐릭터:

```
characterConfig/Pina/identity.txt  # 2줄 수준의 페르소나 텍스트
```

`promptMaker.py:16` 에서 매 프롬프트 첫 메시지로 주입. VoiceVox 스피커 ID (`speaker=46`) 와 번역 타깃 언어(`translate_google(text, detect, "JA")`) 는 **모두 하드코딩** — 런타임 교체 불가.

---

## 7. 스마트홈 / 자율 기능

**코드 레벨 미존재**. MQTT/Home Assistant/IoT API 호출 없음. README 도 특별히 주장하지 않음.

---

## 8. 턴테이킹

```python
while True:
    chat_now = chat
    if is_Speaking == False and chat_now != chat_prev:
        # process
    time.sleep(1)
```

- **1초 폴링 간격** → 구조적 지연 1초
- TTS 완료 후 다음 메시지 처리 (시리얼)
- 음성 바지인 없음 (모드 1 에서도 push-to-talk 이라 불가)

---

## 9. 번역·자막

- `utils/translate.py`: Google Translate (googletrans==4.0.0rc1) + DeepLX
- `utils/subtitle.py`: OBS 용 자막 파일 생성
- 언어 감지 후 2개 타깃(JA 음성용 + EN 자막용) 동시 번역 (`run.py:190-191`)

---

## 10. 성숙도·한계

**완결된 것** ✓
- 멀티플랫폼 (YouTube/Twitch/로컬 mic)
- TTS 백엔드 2종
- 번역 파이프라인
- OBS 자막
- 캐릭터 identity 파일

**미완/취약** ✗
- async/await 전무 (블로킹 `winsound.PlaySound`)
- 예외 핸들러 없음 → 소켓/API 오류 시 크래시
- `winsound`, `keyboard`, PyAudio 등 Windows 전용 의존
- 단일 메시지 버퍼 → 유실
- API 키 평문 저장
- 로깅·모니터링·메트릭 없음

---

## 11. zemory-sama 가 참고할 패턴

**채택 후보**:
1. **Raw IRC 소켓 Twitch 통합** — 의존성 최소화. 추후 방송 확장 시 가장 간단한 진입점.
2. **캐릭터 디렉터리 패턴** (`characterConfig/<name>/identity.txt`) — 다중 페르소나 핫스왑 시
3. **언어 감지 → 번역 타깃 분리** — 한국어 방송 중 영어 댓글 처리용

**반면교사**:
1. 전역 변수 폴링 → `asyncio.Queue` 로 대체
2. `winsound` → `sounddevice` 등 크로스플랫폼으로
3. `is_Speaking` 플래그 → 명시적 state machine
4. 평문 API 키 → `.env` + `pydantic-settings`

---

## 레퍼런스 코드 위치

```
_ref/AI-Waifu-Vtuber/
├── run.py                      # 267줄 엔트리, 모드 분기, 메인 루프
├── utils/
│   ├── TTS.py                  # VoiceVox + Silero
│   ├── translate.py            # Google / DeepLX
│   ├── promptMaker.py          # identity + 히스토리 조립
│   ├── subtitle.py             # OBS 자막
│   ├── twitch_config.py        # 평문 OAuth (경계선)
│   └── katakana.py             # MeCab 가타카나 변환
├── characterConfig/
│   └── Pina/identity.txt
├── speaker.json                # VoiceVox 50+ 스피커
├── conversation.json           # 히스토리 디스크 저장
└── requirements.txt            # openai, pytchat, torch, PyAudio 등 13종
```
