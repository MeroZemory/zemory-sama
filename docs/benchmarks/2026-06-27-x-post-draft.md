# X.com Launch Draft - 2026-06-27

## Korean

제가 만든 `zemory-sama`를 공개합니다.

OpenAI Realtime GA 기반의 멀티링구얼 음성 에이전트 런타임입니다. UI/아바타보다 먼저, 마이크 입력부터 첫 오디오 응답, 인터럽트, 장기 세션 상태까지 음성 코어를 제대로 만드는 데 집중했습니다.

측정값:

- 실제 수동 세션 28턴: p50 816 ms, p95 1.70 s
- 인터럽트 p95: 1.5 ms
- macOS `say` 합성 음성 6개 fixture: input commit -> first audio p50 920 ms, p95 1.20 s

Open-LLM-VTuber, AIRI, RealtimeVoiceChat, Neuro, AI-Waifu-Vtuber, AIRIS도 최신화해서 로컬 의존성 세팅까지 비교했습니다. 공통 fixture가 없는 프로젝트에는 지연시간 숫자를 지어내지 않았고, 세팅 결과와 벤치 산출물을 README에 공개했습니다.

https://github.com/MeroZemory/zemory-sama

## English

I built `zemory-sama`: a low-latency multilingual voice-agent runtime on OpenAI Realtime GA.

It focuses on the voice core before UI/avatar work: mic streaming, first audible response, barge-in, and long-session state.

Benchmarks from 2026-06-27:

- Manual live session, 28 turns: p50 816 ms, p95 1.70 s
- Interrupt p95: 1.5 ms
- Controlled macOS `say` audio fixtures: input commit -> first audio p50 920 ms, p95 1.20 s

I also updated and setup-tested Open-LLM-VTuber, AIRI, RealtimeVoiceChat, Neuro, AI-Waifu-Vtuber, and AIRIS. I did not invent latency numbers where a common non-interactive fixture was unavailable; the README includes the artifacts, caveats, and comparison charts.

https://github.com/MeroZemory/zemory-sama
