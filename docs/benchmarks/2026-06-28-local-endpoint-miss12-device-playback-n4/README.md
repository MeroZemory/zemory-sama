# zemory-sama local endpoint miss12 device playback benchmark n4

4 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=local_endpoint_commit; turn_detection=none; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=300 ms. local endpoint misses=12. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 4 |
| total events | 4 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1481.7 ms |
| turn mean | 1768.8 ms |
| turn p50 | 1832.1 ms |
| turn p90 | 1906.2 ms |
| turn p95 | 1906.2 ms |
| representative max | 1906.2 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1906.2 ms |
| api first audio p50 | 1824.9 ms |
| api first audio representative max | 1900.2 ms |
| api to playback p50 | 5.2 ms |
| api to playback representative max | 7.3 ms |
| speaker buffer p50 | 4.9 ms |
| speaker buffer representative max | 6.6 ms |
| vad wait p50 | 592.5 ms |
| vad wait representative max | 894.6 ms |
| first audio after speech stopped p50 | 976.4 ms |
| first audio after speech stopped representative max | 1232.4 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
