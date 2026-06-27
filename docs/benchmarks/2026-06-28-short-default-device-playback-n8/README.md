# zemory-sama short default device playback benchmark n8

8 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=200 ms. Input chunk=20 ms. response length=one short sentence. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 8 |
| total events | 8 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1144.9 ms |
| turn mean | 1299.7 ms |
| turn p50 | 1261.3 ms |
| turn p90 | 1498.7 ms |
| turn p95 | 1498.7 ms |
| representative max | 1498.7 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1498.7 ms |
| api first audio p50 | 1259.8 ms |
| api first audio representative max | 1489.4 ms |
| api to playback p50 | 5.3 ms |
| api to playback representative max | 9.4 ms |
| speaker buffer p50 | 4.6 ms |
| speaker buffer representative max | 7.7 ms |
| vad wait p50 | 504.4 ms |
| vad wait representative max | 692.6 ms |
| first audio after speech stopped p50 | 733.2 ms |
| first audio after speech stopped representative max | 918.6 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
