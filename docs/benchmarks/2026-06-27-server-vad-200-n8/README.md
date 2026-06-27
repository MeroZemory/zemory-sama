# zemory-sama server_vad 200ms n8 benchmark

8 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=200 ms. Metric is final source-audio chunk sent to first response audio delta; raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 8 |
| total events | 8 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1036.6 ms |
| turn mean | 1727.7 ms |
| turn p50 | 1240.0 ms |
| turn p90 | 5000.5 ms |
| turn p95 | 5000.5 ms |
| representative max | 1509.0 ms |
| extreme outliers | 1 |
| observed max, diagnostic | 5000.5 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
