# Turn Detection Optimization - 2026-06-27

Goal: reduce source-audio-end to first response audio without treating random extreme max values as representative latency.

## Evidence Used

- [OpenAI Realtime VAD documentation](https://developers.openai.com/api/docs/guides/realtime-vad): Realtime sessions support `server_vad` and `semantic_vad`; turn detection is configurable through `session.audio.input.turn_detection`.
- [OpenAI latency optimization guidance](https://developers.openai.com/api/docs/guides/latency-optimization): stream outputs, keep responses short, and measure first-token/first-audio latency separately from full completion latency.
- [FastTurn](https://arxiv.org/html/2604.01897v1): recent turn-detection research frames the target as low latency while preserving robust end-of-turn decisions by combining acoustic and streaming semantic cues.
- [FD-Bench](https://arxiv.org/html/2503.04721v3): recent full-duplex dialogue benchmarking measures response latency from the end of user speech to the start of model response and avoids mixing in unrelated silence periods.
- Reference repo review:
  - RealtimeVoiceChat uses dynamic pause and quick first-audio strategies.
  - Neuro uses a short `post_speech_silence_duration` of 0.4 seconds.
  - AIRI documents first streaming audio packet latency as the relevant UX metric for streaming TTS.

## Local Experiments

All rows use macOS `say` fixtures streamed as realtime 24 kHz PCM. Metric is final source-audio chunk sent to first response audio delta. Raw transcripts are not recorded.

| Candidate | Samples | p50 | Representative max | Result |
| --- | ---: | ---: | ---: | --- |
| `semantic_vad`, eagerness `high` | 2 | 1885.7 ms | 2660.2 ms | Stable but slower VAD wait |
| `semantic_vad`, eagerness `medium` | 1 attempted | n/a | n/a | Timed out on the first fixture |
| `server_vad`, silence 300 ms | 2 | 1462.7 ms | 2318.1 ms | Faster than semantic VAD |
| `server_vad`, silence 200 ms | 2 | 1270.0 ms | 1284.0 ms | Fastest stable two-sample probe |
| `server_vad`, silence 150 ms | 2 | n/a | n/a | Rejected: speech stop fired before source audio ended |
| `server_vad`, silence 100 ms | 2 | n/a | n/a | Rejected: speech stop fired before source audio ended |
| `server_vad`, silence 200 ms | 4 | 1051.5 ms | 1350.5 ms | Chosen default |

![Turn detection comparison](turn-detection-comparison.svg)

## Decision

`realtime_audio` now defaults to `server_vad` with `silence_duration_ms=200`.
`semantic_vad` remains configurable for conversations where avoiding early turn endings is more important than minimum latency.

The speaker output callback block size was also lowered from 40 ms to 20 ms so first decoded audio can reach the output device sooner. This is a playback-path improvement and is not reflected in the API first-audio benchmark above.
