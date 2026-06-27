# zemory-sama live realtime device playback benchmark

4 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=200 ms. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 4 |
| total events | 4 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1092.4 ms |
| turn mean | 1407.2 ms |
| turn p50 | 1228.6 ms |
| turn p90 | 1947.8 ms |
| turn p95 | 1947.8 ms |
| representative max | 1947.8 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1947.8 ms |
| api first audio p50 | 1223.7 ms |
| api first audio representative max | 1943.6 ms |
| api to playback p50 | 4.8 ms |
| api to playback representative max | 10.0 ms |
| speaker buffer p50 | 4.8 ms |
| speaker buffer representative max | 9.9 ms |
| vad wait p50 | 452.4 ms |
| vad wait representative max | 494.8 ms |
| first audio after speech stopped p50 | 771.4 ms |
| first audio after speech stopped representative max | 1490.2 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
