# zemory-sama local endpoint miss14 device playback benchmark n8

8 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=local_endpoint_commit; turn_detection=none; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=300 ms. local endpoint misses=14. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 8 |
| total events | 8 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1327.1 ms |
| turn mean | 1633.6 ms |
| turn p50 | 1654.2 ms |
| turn p90 | 2043.5 ms |
| turn p95 | 2043.5 ms |
| representative max | 2043.5 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 2043.5 ms |
| api first audio p50 | 1646.8 ms |
| api first audio representative max | 2037.2 ms |
| api to playback p50 | 5.3 ms |
| api to playback representative max | 7.4 ms |
| speaker buffer p50 | 5.2 ms |
| speaker buffer representative max | 7.3 ms |
| vad wait p50 | 508.6 ms |
| vad wait representative max | 722.8 ms |
| first audio after speech stopped p50 | 1020.0 ms |
| first audio after speech stopped representative max | 1422.9 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
