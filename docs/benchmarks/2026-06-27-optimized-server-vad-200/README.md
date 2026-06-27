# zemory-sama optimized server_vad 200ms live benchmark

4 macOS say fixtures streamed as realtime 24 kHz PCM. Mode=semantic_vad; turn_detection=server_vad; semantic_vad eagerness=high; server_vad silence=200 ms. Metric is final source-audio chunk sent to first response audio delta; raw transcripts are not recorded.

| Metric | Value |
| --- | ---: |
| turn count | 4 |
| turn min | 1022.2 ms |
| turn mean | 1143.5 ms |
| turn p50 | 1051.5 ms |
| turn p90 | 1350.5 ms |
| turn p95 | 1350.5 ms |
| representative max | 1350.5 ms |
| extreme outliers | 0 |
| observed max, diagnostic | 1350.5 ms |
| interrupt count | 0 |
| interrupt p95 | n/a |

![Latency chart](latency.svg)
