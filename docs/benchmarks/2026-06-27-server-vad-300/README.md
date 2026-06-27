# zemory-sama server_vad 300ms live benchmark

2 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad silence=300 ms. Metric is final source-audio chunk sent to first response audio delta; raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 2 |
| total events | 2 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1462.7 ms |
| turn mean | 1890.4 ms |
| turn p50 | 1462.7 ms |
| turn p90 | 2318.1 ms |
| turn p95 | 2318.1 ms |
| representative max | 2318.1 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 2318.1 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
