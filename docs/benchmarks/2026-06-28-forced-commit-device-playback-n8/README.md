# zemory-sama forced commit device playback benchmark n8

8 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=forced_commit; turn_detection=none; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=300 ms. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 8 |
| total events | 8 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 838.2 ms |
| turn mean | 1005.6 ms |
| turn p50 | 957.0 ms |
| turn p90 | 1354.2 ms |
| turn p95 | 1354.2 ms |
| representative max | 1354.2 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1354.2 ms |
| api first audio p50 | 947.1 ms |
| api first audio representative max | 1344.6 ms |
| api to playback p50 | 9.6 ms |
| api to playback representative max | 11.0 ms |
| speaker buffer p50 | 9.0 ms |
| speaker buffer representative max | 10.2 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
