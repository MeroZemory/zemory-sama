# awesome-ai-vtubers (proj-airi) 카탈로그 요약

> GitHub: https://github.com/proj-airi/awesome-ai-vtubers (393 stars)
> 분석일: 2026-04-15

## 성격

**큐레이션 리스트**(단일 `README.md`) — 코드베이스가 아님. 명시적 카테고리 분할은 없고 알파벳 순이지만, 내용상 다음 군집으로 자연히 나뉜다.

## 분류 (자연 군집)

1. **Core VTuber 프레임워크** — AIRI, Open-LLM-VTuber, AI-Vtuber, aituber-onair
2. **Agent/LLM core** — elizaOS/eliza, nekro-agent, Neuro
3. **Chat·상호작용 레이어** — LingChat, Muice-Chatbot, amica
4. **데스크톱 Companion** — Mate-Engine, ZcChat
5. **스트리밍·판매** — Streamer-Sales, Riko Project
6. **TTS·보이스 컴포넌트** — edge-tts, VITS, ElevenLabs, Bark, Bert-VITS2
7. **아바타 시스템** — Live2D, VRM, UE 산발 언급

## Tier 1 (포괄 스택)

| 프로젝트 | 요지 |
|---------|------|
| **AIRI** (moeru-ai) | Web-first, WebGPU/WASM, realtime voice + 게임 연동 (이미 `_ref/` 에 클론) |
| **Open-LLM-VTuber** (t41372) | 핸즈프리 음성 + Live2D, 크로스플랫폼 (이미 `_ref/`) |
| **AI-Vtuber** (Ikaros-521) | 중화권 생태계 리더. Bilibili/抖音/YouTube/Twitch 9종 LLM 백엔드 |
| **aituber-onair** (shinshin86) | TypeScript 모노레포, "kizuna" 관계성 시스템, 다중 TTS |

## Tier 2 (독특한 각도)

| 프로젝트 | 특징 |
|---------|------|
| **nekro-agent** (KroMiose) | 샌드박스 에이전트, OneBot v11(QQ)/Discord/Minecraft/B站 live 플러그인 생태계 |
| **elizaOS/eliza** | 범용 자율 에이전트 프레임워크 (Python/TS) |
| **Mate-Engine** (shinyflvre) | VRM 지원 데스크톱 Companion |

## 주목할 변종

- **Bella** — Grok Companion 재현 (챗봇보다 Companion 우선)
- **Streamer-Sales** (PeterH0323) — "판매원으로서의 LLM" (AGPL)
- **J.A.I.son** — ML 비의존 설정형 응답 서버
- **Vixevia** (IRedDragonICY) — Google Gemini 특화
- **Xiao8 (Lanlan)** — "Audio-native" 3분 셋업 접근성 강조
- **z-waif** — 완전 로컬, "자기 AI waifu 만들기" 경로

## 동반 툴링 (자주 언급)

| 카테고리 | 항목 |
|---------|------|
| LLM | OpenAI, Claude, ChatGPT, LangChain, ChatGLM, text-gen-webui, Qwen, Kimi, Ollama, Grok, Gemini |
| TTS | edge-tts, VITS, ElevenLabs, Bark, Bert-VITS2, 睿声 |
| 음성변환 | so-vits-svc, DDSP-SVC |
| 아바타 | Live2D, UE, VRM, Galgame-style |
| 플랫폼 | Bilibili, 抖音, 快手, YouTube, Twitch, TikTok, Discord, OneBot v11(QQ), Minecraft, B站直播 |

## 추가 클론 후보 (우선순위)

zemory-sama 에 보완할만한 다음 후보들:

1. **nekro-agent** (KroMiose) — 샌드박스 + 다중 플랫폼 이벤트 아키텍처 + 플러그인 생태계. `_ref/airi`, `_ref/Open-LLM-VTuber` 에 없는 중화권 플랫폼 커버리지.
2. **elizaOS/eliza** — 범용 에이전트 프레임워크. 최신 VTuber 스택에 영향력 큼.
3. **aituber-onair** (shinshin86) — "kizuna"(인연) 관계성 시스템 + 다중 TTS. TypeScript 모노레포.
4. **Streamer-Sales** (PeterH0323) — 판매 시나리오 LLM. Vue + Python 풀스택 성숙도.
5. **Mate-Engine** (shinyflvre) — VRM 데스크톱 Companion. 아바타 렌더링 갭 보완.

## 주목할 공백

리스트 전반이 **중화권 프로젝트(Bilibili 생태계) 편중** — 한국어/일본어 VTuber 제작 특화 도구(Live2D 스튜디오, 애니메이션 프레임워크)는 부족. 의도적 범위 제한으로 보임 (상업 도구 제외 정책).

---

## 참고

- 리포지토리 위치: `_ref/awesome-ai-vtubers/README.md`
- 리스트는 자주 업데이트되므로 새 후보 탐색 시 원본 README 를 재확인할 것
