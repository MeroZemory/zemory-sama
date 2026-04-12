# 레퍼런스 프로젝트 비교 및 zemory-sama 적용 방향

> 분석일: 2026-04-13

## 1. 핵심 패턴 비교

| 항목 | Open-LLM-VTuber | RealtimeVoiceChat | zemory-sama (현재) |
|------|----------------|-------------------|-------------------|
| **아키텍처** | 웹 (FastAPI+WebSocket) | 웹 (FastAPI+WebSocket) | CLI (asyncio+sounddevice) |
| **VAD** | Silero VAD (자체 구현) | Silero VAD (RealtimeSTT 내장) | OpenAI Realtime API server_vad |
| **STT** | 다양한 ASR 엔진 | Faster-Whisper | OpenAI Realtime API 내장 |
| **LLM** | 다양한 LLM | Ollama 로컬 | OpenAI Realtime API 내장 |
| **TTS** | 다양한 TTS 엔진 | RealtimeTTS (다양한 엔진) | ElevenLabs Flash v2.5 |
| **에코 방지** | 분리 채널 (웹 구조) | `interrupted` 플래그 게이트 | `speaking` 이벤트 + 버퍼 클리어 |
| **턴 테이킹** | VAD 상태머신 + 시그널 프로토콜 | ML 턴 감지 + 동적 pause | server_vad + speech_stopped |
| **마이크 차단 시점** | VAD PAUSE 시그널 즉시 | 사용자 턴 종료 즉시 | speech_stopped 이벤트 |
| **마이크 해제 시점** | 프런트엔드 "재생완료" 확인 후 | 첫 TTS 청크 후 1초 | speaker 버퍼 비움 후 |
| **인터럽트** | asyncio.Task.cancel() | abort 이벤트 체인 | 비활성화 (half-duplex) |
| **TTS 순서 보장** | 시퀀스 넘버 리오더링 | Quick/Final 2단계 | 순차 처리 (단일 큐) |
| **TTS 병렬 생성** | O (TTSTaskManager) | O (Quick+Final 워커) | X (순차) |

## 2. 에코 방지 전략 비교

### Open-LLM-VTuber: 구조적 분리
- 웹 브라우저가 마이크/스피커를 제어
- 서버는 오디오를 수신만 하고 스피커 출력과 물리적으로 분리
- 브라우저 내장 에코 캔슬레이션 활용 가능
- **장점**: 가장 확실한 에코 방지
- **단점**: 웹 UI 필수

### RealtimeVoiceChat: 소프트웨어 게이팅
- `interrupted` 플래그로 STT 입력 차단
- 오디오는 계속 캡처하되 STT에 전달하지 않음
- 첫 TTS 청크 전송 후 1초 타이머로 해제
- **장점**: 단순, 효과적
- **단점**: 1초 해제가 스피커 재생 완료와 정확히 맞지 않을 수 있음

### zemory-sama: 이벤트 기반 게이팅
- `speaking` asyncio.Event로 마이크 전송 차단
- `speech_stopped` 이벤트에서 즉시 차단 시작
- `speaker.wait_until_done()` 후 해제 + `input_audio_buffer.clear()`
- **장점**: 스피커 버퍼 상태 기반으로 정확한 타이밍
- **주의**: PortAudio 내부 버퍼에 잔여 오디오 가능성

## 3. 마이크 해제 타이밍 비교

가장 중요한 차이점:

```
Open-LLM-VTuber:
  TTS 완료 → 프런트엔드 재생 → "playback-complete" 전송 → 마이크 해제
  (가장 확실하지만 웹 아키텍처 필요)

RealtimeVoiceChat:
  첫 TTS 청크 전송 → 1초 타이머 → 마이크 해제
  (단순하지만 긴 응답에서는 스피커가 아직 재생 중일 수 있음)

zemory-sama:
  마지막 TTS 청크 수신 → speaker 큐+버퍼 비움 대기 → 마이크 해제
  (PortAudio 내부 버퍼 완전 비움은 보장 불가)
```

## 4. zemory-sama 적용 가능한 개선 사항

### 즉시 적용 가능

**A. 스피커 재생 완료 후 안전 딜레이**
- RealtimeVoiceChat처럼 `speaker.wait_until_done()` 후 추가 0.5~1초 대기
- PortAudio 내부 버퍼 잔여분이 실제로 재생될 시간 확보
- 구현 난이도: 낮음 (1줄 추가)

**B. 서버 버퍼 이중 클리어**
- `speech_stopped` 시점에도 `input_audio_buffer.clear()` 호출
- 마이크 해제 시점에도 `input_audio_buffer.clear()` 호출
- 구현 난이도: 낮음

### 중기 개선

**C. TTS 병렬 생성 + 순서 보장**
- Open-LLM-VTuber의 TTSTaskManager 패턴 도입
- 여러 문장의 TTS를 동시에 요청하되 재생은 순서대로
- 체감 지연 감소 (특히 긴 응답에서)
- 구현 난이도: 중간

**D. Quick Answer 패턴**
- RealtimeVoiceChat처럼 첫 문장을 빠르게 처리
- 첫 문장은 작은 청크, 나머지는 큰 청크
- 구현 난이도: 중간

### 장기 개선

**E. 로컬 VAD (Silero) 도입**
- 현재 server_vad는 OpenAI에 오디오를 보낸 후 서버에서 판단
- 로컬 VAD로 전환하면 마이크 차단 타이밍을 완전히 제어 가능
- Silero VAD의 3-hit/24-miss 패턴으로 안정적인 턴 감지
- 구현 난이도: 높음 (아키텍처 변경)

**F. 웹 UI 전환**
- Open-LLM-VTuber처럼 웹 기반으로 전환
- 에코 문제 구조적 해결 + VTuber 아바타 표시
- 향후 VTuber 기능 확장의 기반
- 구현 난이도: 높음

## 5. 레퍼런스 코드 위치

### Open-LLM-VTuber
```
_ref/Open-LLM-VTuber/
├── src/open_llm_vtuber/
│   ├── vad/
│   │   ├── silero.py              # VAD 상태 머신
│   │   ├── vad_interface.py       # 추상 인터페이스
│   │   └── vad_factory.py         # 팩토리
│   ├── conversations/
│   │   ├── conversation_handler.py # 대화 트리거 + 인터럽트
│   │   ├── single_conversation.py  # 단일 대화 흐름
│   │   ├── conversation_utils.py   # 시그널 + 동기화
│   │   └── tts_manager.py         # TTS 병렬 생성 + 순서 보장
│   ├── websocket_handler.py       # WebSocket 라우팅
│   └── message_handler.py         # 응답 대기 메커니즘
└── config_templates/
    └── conf.default.yaml          # VAD 기본 설정
```

### RealtimeVoiceChat
```
_ref/RealtimeVoiceChat/
└── code/
    ├── server.py                  # 메인 서버 + 콜백 + TTS 전송
    ├── audio_in.py                # 오디오 입력 + interrupted 게이팅
    ├── transcribe.py              # STT + 침묵 모니터
    ├── turndetect.py              # ML 턴 감지 (DistilBERT)
    ├── speech_pipeline_manager.py # LLM+TTS 파이프라인 관리
    └── audio_module.py            # TTS 합성 + 오디오 버퍼링
```
