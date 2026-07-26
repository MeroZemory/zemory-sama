# Current-State Research Snapshot

> 조사일: 2026-06-27
> 목적: zemory-sama 저지연 실시간 음성 에이전트 설계 개정을 위한 최신 근거 정리.

> [!NOTE]
> **Historical evidence snapshot with a 2026-07-26 implementation correction.**
> 아래 외부 자료와 당시 판단은 조사일 기준으로 보존한다. 다만 실험적으로 추가됐던
> `TranscriptLedger`는 prompt, compaction, persistence consumer가 없어 현재 코드에서
> 제거됐다. 현재 conversation source는 provider state와 generation-correlated runtime
> handling이며, SQLite memory/context scheduler는 `local_cascade`만 실제로 소비한다.
> 현재 운영 상태는 [2026-07-26 adversarial audit](adversarial-audit-2026-07-26.md)을
> 기준으로 한다.

## Evidence Boundaries

- **Sourced facts**: 공식 OpenAI 문서, arXiv 논문, GitHub repository/README/release/API에서 확인한 내용이다.
- **Local facts**: 현재 repository 의 구현과 문서에서 확인한 내용이다.
- **Inference**: 여러 근거를 대조해 zemory-sama 설계에 적용한 판단이다.
- 이 문서는 특정 날짜의 스냅샷이다. GitHub stars, release, docs 내용은 이후 바뀔 수 있다.

## 1. OpenAI Realtime Facts

| Source | 확인 내용 | 설계 영향 |
| --- | --- | --- |
| [Realtime and audio](https://developers.openai.com/api/docs/guides/realtime) | 저지연 voice agent 시작점은 `gpt-realtime-2`이며, live audio session 과 chained voice pipeline 을 구분한다. GA migration 은 beta header 제거, 새 session/event shape, output audio config 사용을 요구한다. | 기본 fast path 를 `gpt-realtime-2` audio-in/audio-out으로 둔다. 기존 preview/text-only 전제는 낡았다. |
| [Voice agents](https://developers.openai.com/api/docs/guides/voice-agents) | 자연스러운 저지연 대화, barge-in, realtime tool use 는 speech-to-speech live audio session 이 적합하다. chained pipeline 은 중간 텍스트 제어가 중요할 때 적합하다. | audio-native profile 을 기본값, text+TTS 체인은 선택 프로파일로 둔다. |
| [Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations) | Realtime session 예시는 `model: "gpt-realtime-2"`, `output_modalities: ["audio"]`, `semantic_vad`, output voice `marin`을 사용한다. 세션 최대 길이는 60분이다. | config/schema/event adapter 를 GA 기준으로 업데이트한다. 장기 세션 compaction 이 필요하다. |
| [Voice activity detection](https://developers.openai.com/api/docs/guides/realtime-vad) | `server_vad`는 침묵 기반, `semantic_vad`는 사용자가 말을 끝냈는지 의미 기반으로 판단한다. `semantic_vad`는 `eagerness`로 조절한다. | 한국어/영어 턴 종료는 `semantic_vad` 우선, `server_vad`/Silero fallback 으로 둔다. |
| [Realtime with tools](https://developers.openai.com/api/docs/guides/realtime-mcp) | Realtime session 에 function tool, remote MCP server, built-in connector 를 붙일 수 있다. tool 은 session 또는 response 단위로 설정 가능하다. | memory/RAG/tool 은 sideband/async layer 로 설계하고, pending result 처리를 명시한다. |
| [Developer notes on Realtime API](https://developers.openai.com/blog/realtime-api) | GA Realtime 은 image input, async function calling, MCP, audio token to text, long context, SIP, idle timeout 등을 포함한다. 세션은 최대 60분, `gpt-realtime` 계열 context 는 32,768 tokens, response max 는 4,096 tokens, instructions+tools max 는 16,384 tokens 이다. | 조사 당시에는 `TranscriptLedger`/`ContextCompactor`를 미래 설계로 제안했다. 현재 구현은 둘을 제공하지 않고 Realtime native retention-ratio truncation을 사용한다. MCP와 async function calling 은 GA 경로에서 다룬다. |

## 2. Speech Research Signals

| Paper / project | Date | Sourced fact | Design implication |
| --- | --- | --- | --- |
| [Moshi](https://arxiv.org/abs/2410.00037) / [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | 2024-09, updated 2024-10 | full-duplex speech-text foundation model. Cascaded VAD/ASR/text/TTS pipelines lose overlap, interruption, and backchannels; Moshi reports about 200 ms practical latency. | Cascaded path is not the future default. Keep as fallback, monitor full-duplex SLMs. |
| [MoshiRAG](https://arxiv.org/abs/2604.12928) / [kyutai-labs/moshi-rag](https://github.com/kyutai-labs/moshi-rag) | 2026-04, updated 2026-05 | full-duplex SLM에 asynchronous knowledge retrieval 을 붙여 대화 흐름을 유지하면서 factuality 를 높인다. | 검색은 턴 종료 후 blocking 이 아니라 async/pending result 로 설계한다. |
| [Endpoint Anticipation](https://arxiv.org/abs/2606.13450) | 2026-06 | 발화 종료를 최대 2.56초 먼저 예측해 speculative LLM/TTS를 실행하고 평균 latency 505 ms 감소를 보고한다. | partial transcript 기반 speculative response 는 실험 옵션으로 가치가 있다. 기본값은 아니다. |
| [WavRAG](https://arxiv.org/abs/2502.14727) | 2025-02 | ASR을 거치지 않는 audio-integrated RAG로 spoken dialogue retrieval 을 가속한다. | audio-native RAG는 research profile 에 둔다. 당장 구현은 text transcript 기반 async recall 로 시작한다. |
| [Stream RAG](https://arxiv.org/abs/2510.02044) | 2025-10 | 사용자 발화가 끝나기 전에 tool query 를 예측해 tool latency 를 줄인다. | user turn 완료 전 async tool/RAG scheduling 의 근거가 된다. |
| [BayLing-Duplex](https://arxiv.org/abs/2606.14528) | 2026-06 | 단일 autoregressive LLM으로 listen/speak/stop 을 결정하는 native full-duplex speech dialogue 를 제안한다. | 장기 방향은 auxiliary turn-taking 축소다. 현재는 실험 프로파일로만 둔다. |
| [TurnGuide](https://arxiv.org/abs/2508.07375) | 2025-08, updated 2026-06 | turn-level text-speech interleaving 으로 full-duplex speech model 의 coherence 와 turn-taking 을 개선한다. | text/speech interleaving 평가는 research benchmark 에 포함한다. |
| [Raon-Speech](https://arxiv.org/abs/2605.23912) | 2026-04 | English/Korean 9B speech LM과 full-duplex Raon-SpeechChat pipeline 을 공개한다. | bilingual full-duplex 후보로 중요하다. hardware/cost 검증 전 production 기본값은 아니다. |

## 3. Reference Project Snapshot

| Project | 2026-06-27 status snapshot | Design takeaway |
| --- | --- | --- |
| [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) | GitHub API 기준 updated_at 2026-06-27, latest release v1.2.1 published 2025-08-26. README 는 v2.0 complete rewrite 가 early discussion/planning 이라고 명시한다. Long-term memory 는 temporarily removed 상태다. | v1 구현을 그대로 따라가기보다 v2 방향을 관찰한다. interrupt, provider, agent interface 패턴만 선별한다. |
| [AIRI](https://github.com/moeru-ai/airi) | GitHub API 기준 updated_at/pushed_at 2026-06-27, latest release v0.10.2 published 2026-05-07. README 는 browser/desktop/mobile, Discord/Telegram, Minecraft/Factorio, VRM, multi-provider voice synthesis, DuckDB WASM/pglite memory work 를 보여준다. | 가장 활발한 reference. web/native 확장, memory portability, provider layering 방향성을 반영한다. |
| [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat) | GitHub API 기준 updated_at 2026-06-27, pushed_at 2025-07-11. README 는 project status 를 Community-Driven 으로 두고 원 maintainer가 새 기능/지원에서 물러났다고 말한다. | low-latency cascade, interrupt, Docker 패턴은 참고하되 새 core dependency 로 삼지 않는다. |
| [Neuro](https://github.com/kimjammer/Neuro) | GitHub API 기준 updated_at 2026-06-27, pushed_at 2025-01-17. README 는 memory/RAG, vision, module injection, Twitch integration, prompt priority 를 강조한다. | memory, module injection, priority prompt 패턴은 유용하다. latency core reference 는 아니다. |
| [AI-Waifu-Vtuber](https://github.com/ardha27/AI-Waifu-Vtuber) | GitHub API 기준 updated_at 2026-06-26, pushed_at 2026-05-31. README 는 legacy `openai==0.28.1`, config.py key handling, Twitch/YouTube streaming 을 포함한다. | legacy streaming reference. 새 설계의 API/보안/latency 기준으로는 낮은 우선순위다. |
| [AIRIS-VtuberAI](https://github.com/StudioMovieGirl/AIRIS-VtuberAI) | 원 repo 는 이동됨. GitHub API 기준 moved target updated_at 2026-06-22, pushed_at 2026-01-03. README 는 새 버전에서 interrupting model, tool calling, Ollama API 를 계획한다고 말한다. | planned local/GPU pipeline 관찰용. 현 core 설계 근거는 아니다. |
| [awesome-ai-vtubers](https://github.com/proj-airi/awesome-ai-vtubers) | GitHub API 기준 updated_at 2026-06-20. README 에 [projectBEA](https://github.com/emqnuele/projectBEA), [nekro-agent](https://github.com/KroMiose/nekro-agent), [Xiao8](https://github.com/wehos/Xiao8) 등 새 후보가 보인다. | reference set 을 7개 고정으로 보지 않는다. 새 후보는 watch list 로 둔다. |

## 4. Local Repository Snapshot

| Area | Local fact | Design implication |
| --- | --- | --- |
| Current config | 구현 전 스냅샷에서는 `gpt-4o-mini-realtime-preview` + text/external TTS 전제였으나, 2026-07-26 현재 구현은 `realtime_audio` 기본 프로파일, `gpt-realtime-2.1`, GA `session.audio.*`, app-owned response creation을 위한 `server_vad`로 갱신됐다. | 기본 fast path 는 GA audio-native 로 전환됐고, non-empty transcript를 확인한 뒤에만 응답을 만든다. |
| Provider shape | `zemory/providers/base.py`에는 provider abstraction 이 이미 있다. | 전면 rewrite 보다 profile adapter 교체가 맞다. |
| Interrupt | `InterruptBus`의 partial callback은 현재 내용 자체를 저장하지 않고 길이만 기록한다. Realtime audio는 실제 재생 cursor까지만 provider item을 truncate하고, external-TTS는 전체 assistant item을 delete한 뒤 generic interruption note를 남긴다. 두 mutation은 ACK 전 다음 response를 만들지 않는다. | 들리지 않은 전체 답변을 local correction history나 provider context에 완료 발화처럼 보존하지 않는다. |
| Memory | 2026-06-27 iteration에는 `SQLiteMemoryStore`, `TranscriptLedger`, `AsyncContextScheduler`가 추가됐었다. 2026-07-26 현재 consumer가 없던 `TranscriptLedger`와 optional Chroma dependency는 제거됐고, SQLite provider와 scheduler 결과는 `local_cascade`만 소비한다. Realtime profile은 null memory provider를 사용해 시작 시 DB를 만들지 않는다. | blocking Chroma-first 대신 deadline 기반 local-cascade recall은 유지하되, canonical history/compaction은 별도 미래 설계로 남긴다. |
| Tests | 기존 tests 에 GA Realtime profile/event tests, fake SDK adapter tests, async memory/tool deadline tests, SQLite memory tests, latency report/CLI tests, speech-start interrupt tests, local VAD fallback tests 가 추가됐다. `uv run pytest tests/`는 core coverage >= 80% gate 를 포함한다. | 사용자 인터랙션 없이 핵심 설계 경로를 검증한다. |

## 5. Design Decisions

1. **Default to audio-native Realtime**: 조사 당시 OpenAI 공식 문서가 저지연 voice agent 에 `gpt-realtime-2` live audio session 을 직접 제시했고, 현재 구현은 이를 `gpt-realtime-2.1`로 올렸다. external TTS 체인은 선택 프로파일로 유지한다.
2. **Keep chained voice as a profile**: 외부 TTS 목소리와 중간 텍스트 제어가 필요할 때는 chained pipeline 이 여전히 맞다.
3. **Use server VAD for app-owned response creation**: semantic VAD의 의미 기반 turn completion은 연구 후보로 남기되, 현재 기본값은 `server_vad`이다. 서버 자동 응답은 끄고 non-empty, item-correlated transcript를 받은 뒤 앱이 정확히 한 번 `response.create`를 보낸다.
4. **Make RAG asynchronous**: 2025-2026 연구들은 retrieval/tool latency 를 대화 흐름과 분리하는 방향을 반복해서 보여준다.
5. **Treat full-duplex SLMs as watch/research**: Moshi/Raon/BayLing 계열은 방향성이 강하지만, hardware, model quality, multilingual behavior, integration cost 검증 전에는 production default 가 아니다.
6. **Use active references conservatively**: AIRI는 active architecture reference, Open-LLM-VTuber는 v2 rewrite watch, RealtimeVoiceChat/Neuro는 pattern-only다.
