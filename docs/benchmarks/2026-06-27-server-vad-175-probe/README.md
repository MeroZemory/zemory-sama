# zemory-sama server_vad 175ms probe

2 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=175 ms. Metric is final source-audio chunk sent to first response audio delta; raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 1 |
| total events | 2 |
| invalid latency samples | 1 |
| early cutoffs | 1 |
| turn min | 1290.0 ms |
| turn mean | 1290.0 ms |
| turn p50 | 1290.0 ms |
| turn p90 | 1290.0 ms |
| turn p95 | 1290.0 ms |
| representative max | 1290.0 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1290.0 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
