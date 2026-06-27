# zemory-sama local endpoint miss10 device playback benchmark n4

4 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=local_endpoint_commit; turn_detection=none; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=300 ms. local endpoint misses=10. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 2 |
| total events | 4 |
| invalid latency samples | 2 |
| early cutoffs | 2 |
| turn min | 1309.2 ms |
| turn mean | 1349.6 ms |
| turn p50 | 1309.2 ms |
| turn p90 | 1389.9 ms |
| turn p95 | 1389.9 ms |
| representative max | 1389.9 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1389.9 ms |
| api first audio p50 | 1025.6 ms |
| api first audio representative max | 1380.5 ms |
| api to playback p50 | 1.8 ms |
| api to playback representative max | 9.4 ms |
| speaker buffer p50 | 1.8 ms |
| speaker buffer representative max | 9.3 ms |
| vad wait p50 | -1239.7 ms |
| vad wait representative max | 367.0 ms |
| first audio after speech stopped p50 | 1023.7 ms |
| first audio after speech stopped representative max | 2983.3 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
