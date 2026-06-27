# zemory-sama input chunk 10ms n8 benchmark

8 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=200 ms. Input chunk=10 ms. Metric is final source-audio chunk sent to first response audio delta; raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 8 |
| total events | 8 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1107.6 ms |
| turn mean | 1326.3 ms |
| turn p50 | 1205.6 ms |
| turn p90 | 1810.5 ms |
| turn p95 | 1810.5 ms |
| representative max | 1810.5 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1810.5 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
