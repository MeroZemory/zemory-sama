# Realtime 2.1 migration benchmark evidence

This directory preserves the aggregate results and all 24 numeric samples used
for the 2026-07-26 model decision. Raw transcripts and generated audio were not
stored.

| Cohort | Boundary | n | Invalid / early | p50 | p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-realtime-2.1` app-owned server VAD | final source chunk → API first audio | 8 | 0 / 0 | 1574.5 ms | 2013.7 ms |
| `gpt-realtime-2.1` forced commit | final source chunk → device callback | 8 | 0 / 0 | 748.9 ms | 1307.4 ms |
| `gpt-realtime-2.1-mini` forced commit | final source chunk → device callback | 8 | 0 / 0 | 946.9 ms | 1161.2 ms |

Percentiles use nearest-rank (`ceil(p/100*n)`, one-indexed). With eight
samples, p95 is therefore the observed maximum; it is not an interpolated
estimate of a broader population tail.

The automatic-VAD cohort used `server_vad`, 200 ms silence, 20 ms chunks,
`create_response=false`, and an app-owned `response.create` only after a
non-empty transcription. Its `speech_stopped` → first-audio p50/p95 was
1099.7/1545.7 ms and VAD wait p50/p95 was 475.2/627.8 ms.

The forced-commit cohorts used real local playback (`play_output=true`) and
`turn_detection=none`; they are compatibility/latency probes, not response
quality tests. The historical `gpt-realtime-2` result was produced in an older
run and is intentionally not mixed into this contemporaneous artifact.

These three migration cohorts were collected before the 512-token server cap
was added. They therefore support the model compatibility/latency comparison,
not the cap itself. A later one-sample live smoke accepted
`max_output_tokens=512` and returned first audio in 687.8 ms. Its original safe
event and explicit single-sample/provenance limitations are preserved in the
[compatibility-smoke artifact](max-output-smoke/).

All three runs used runtime 0.2.0, OpenAI Python SDK 2.31.0, four macOS `say`
fixtures over two trials each, 24 kHz PCM, and eight samples. See `summary.json`
for run IDs, configuration hashes, and each measured value. The producer computed those
hashes over its expanded configuration, but the then-current public-event
sanitizer accidentally dropped the expanded object. That historical detail
cannot be recovered from the hash and is recorded as a provenance limitation;
new benchmark artifacts retain a recursively prompt-redacted effective
configuration alongside the hash, including hashed endpoint, source-tree, diff,
and rendered-PCM identities plus the trial and timeout protocol.
