# zemory-sama speaker output callback benchmark

24 local SpeakerStream output samples. Output callback block=10 ms; sounddevice latency=low. Metric is queue insertion to first output callback that consumes non-zero PCM. Numeric-only export; no transcripts are recorded.

| Metric | Value |
| --- | ---: |
| sample count | 24 |
| output callback block | 10 ms |
| queue to first playback callback min | 0.87 ms |
| queue to first playback callback mean | 7.19 ms |
| queue to first playback callback p50 | 7.40 ms |
| queue to first playback callback p90 | 8.29 ms |
| queue to first playback callback representative max | 8.33 ms |
| queue to first playback callback observed max, diagnostic | 8.33 ms |
| queue to first playback callback outliers | 0 |
| queue to speaker buffer p50 | 0.06 ms |
| speaker buffer p50 | 7.29 ms |
| speaker buffer representative max | 8.25 ms |

![Latency chart](latency.svg)
