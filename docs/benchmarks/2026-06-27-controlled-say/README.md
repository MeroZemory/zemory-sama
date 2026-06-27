# zemory-sama controlled TTS-audio benchmark

Six synthetic Korean/English utterances generated with macOS say, streamed directly as 24 kHz PCM into the OpenAI Realtime adapter. Metric is input-stream end/commit to first response audio delta because synthetic TTS clips do not always trigger semantic_vad speech_stopped. Numeric-only export; no private transcripts.

| Metric | Value |
| --- | ---: |
| turn count | 6 |
| turn min | 819.1 ms |
| turn mean | 960.9 ms |
| turn p50 | 920.4 ms |
| turn p95 | 1202.8 ms |
| turn max | 1202.8 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
