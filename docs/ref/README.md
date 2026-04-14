# Reference 분석 인덱스

> zemory-sama 설계에 참고한 7개 오픈소스 프로젝트 분석 문서. 모든 레포는 `_ref/` 하위에 shallow clone 되어 있음.

## 시작점

- **[comparison.md](./comparison.md)** — 7개 레퍼런스 비교표 + zemory-sama 적용 로드맵 (Tier A/B/C)

## 레퍼런스별 심층 분석

| 문서 | 프로젝트 | 한 줄 요약 |
|------|---------|-----------|
| [open-llm-vtuber.md](./open-llm-vtuber.md) | Open-LLM-VTuber/Open-LLM-VTuber | 웹 + Live2D + 교과서적 Silero 상태머신 |
| [realtime-voice-chat.md](./realtime-voice-chat.md) | KoljaB/RealtimeVoiceChat | 저지연 full-duplex, ML 턴 감지 |
| [airi.md](./airi.md) | moeru-ai/airi | Vue 모노레포, xsai 40+ provider, VRM/Kokoro |
| [neuro.md](./neuro.md) | kimjammer/Neuro | Chroma reflection 메모리, Patience 자율발화 |
| [ai-waifu-vtuber.md](./ai-waifu-vtuber.md) | ardha27/AI-Waifu-Vtuber | Twitch/YouTube 채팅 응답 (Raw IRC) |
| [airis-vtuber-ai.md](./airis-vtuber-ai.md) | neurokitti/AIRIS-VtuberAI | 완전 로컬 추론, 문장 단위 조기 TTS |
| [awesome-ai-vtubers.md](./awesome-ai-vtubers.md) | proj-airi/awesome-ai-vtubers | 추가 발굴 카탈로그 |

## zemory-sama 에 당장 채용할 후보 (comparison.md Tier A 요약)

1. **문장 단위 조기 TTS 트리거** — AIRIS 패턴. 구두점 감지 시 즉시 TTS 큐잉. ≤ 1일.
2. **시스템 프롬프트 파일 분리** — `prompts/*.txt` 로 핫스왑. ≤ 0.5일.
3. **`input_audio_buffer.clear()` 이중 호출** — 에코 잔여 대비. ≤ 0.5일.
4. **스피커 종료 후 안전 딜레이 0.5-1 s** — RVC 패턴. ≤ 0.5일.

## 향후 클론 후보 (awesome-ai-vtubers 에서 발굴)

1. `KroMiose/nekro-agent` — 샌드박스 에이전트, 다중 플랫폼
2. `elizaOS/eliza` — 범용 자율 에이전트 프레임워크
3. `shinshin86/aituber-onair` — TypeScript 모노레포, 관계성 시스템
4. `PeterH0323/Streamer-Sales` — 판매 시나리오 LLM
5. `shinyflvre/Mate-Engine` — VRM 데스크톱 Companion
