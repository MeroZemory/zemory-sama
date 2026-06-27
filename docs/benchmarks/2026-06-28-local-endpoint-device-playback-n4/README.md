# zemory-sama local endpoint device playback benchmark n4

4 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=local_endpoint_commit; turn_detection=none; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=300 ms. Input chunk=20 ms. play_output=True. Metric is final source-audio chunk sent to first local speaker playback callback; api_first_audio_ms retains the API first-audio delta timestamp. raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 4 |
| total events | 4 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 2853.7 ms |
| turn mean | 3029.9 ms |
| turn p50 | 2940.3 ms |
| turn p90 | 3255.7 ms |
| turn p95 | 3255.7 ms |
| representative max | 3255.7 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 3255.7 ms |
| api first audio p50 | 2932.7 ms |
| api first audio representative max | 3248.4 ms |
| api to playback p50 | 7.3 ms |
| api to playback representative max | 10.1 ms |
| speaker buffer p50 | 6.7 ms |
| speaker buffer representative max | 9.8 ms |
| vad wait p50 | 2034.6 ms |
| vad wait representative max | 2082.6 ms |
| first audio after speech stopped p50 | 889.5 ms |
| first audio after speech stopped representative max | 1165.8 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
