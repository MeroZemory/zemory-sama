# zemory-sama input chunk 10ms n4 benchmark

4 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad threshold=0.5; server_vad silence=200 ms. Input chunk=10 ms. Metric is final source-audio chunk sent to first response audio delta; raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 4 |
| total events | 4 |
| invalid latency samples | 0 |
| early cutoffs | 0 |
| turn min | 1142.2 ms |
| turn mean | 1197.3 ms |
| turn p50 | 1150.4 ms |
| turn p90 | 1334.5 ms |
| turn p95 | 1334.5 ms |
| representative max | 1334.5 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1334.5 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
