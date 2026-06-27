# zemory-sama live realtime device playback benchmark n8

8 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=200 ms. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 8 |
| total events | 8 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1105.1 ms |
| turn mean | 1282.4 ms |
| turn p50 | 1275.0 ms |
| turn p90 | 1403.3 ms |
| turn p95 | 1403.3 ms |
| representative max | 1403.3 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1403.3 ms |
| api first audio p50 | 1267.9 ms |
| api first audio representative max | 1393.5 ms |
| api to playback p50 | 6.5 ms |
| api to playback representative max | 9.8 ms |
| speaker buffer p50 | 6.3 ms |
| speaker buffer representative max | 9.8 ms |
| vad wait p50 | 512.3 ms |
| vad wait representative max | 548.1 ms |
| first audio after speech stopped p50 | 728.6 ms |
| first audio after speech stopped representative max | 880.4 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
