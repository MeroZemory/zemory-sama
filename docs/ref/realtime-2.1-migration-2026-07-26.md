# GPT-Realtime-2.1 migration

## Decision

The default Realtime model is now `gpt-realtime-2.1`. It is a direct
`v1/realtime` replacement for `gpt-realtime-2`: OpenAI documents the same
text/image/audio modalities and Realtime endpoint, while explicitly listing
improved alphanumeric recognition, silence/noise handling, and interruption
behavior. Its published token prices are unchanged from `gpt-realtime-2`.

The default continues to use `reasoning.effort="low"`. The 2.1 model supports
configurable reasoning effort, and higher effort can increase latency and
output-token usage, so raising it is not part of this compatibility migration.

The source default does not override deployment configuration. If an existing
repository-root `.env` or exported process environment sets
`ZEMORY_REALTIME__MODEL`, that value wins; an exported value also wins over the
same key in `.env`. Set `ZEMORY_REALTIME__MODEL=gpt-realtime-2.1` explicitly and
restart the process when migrating an existing installation pinned to an older
model.

Sources:

- [GPT-Realtime-2.1 model](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [Realtime cost and caching guide](https://developers.openai.com/api/docs/guides/realtime-costs)

## Full model and mini comparison

| Model | Text input / cached / output (per 1M) | Audio input / cached / output (per 1M) | Official positioning | Decision |
| --- | --- | --- | --- | --- |
| `gpt-realtime-2.1` | $4.00 / $0.40 / $24.00 | $32.00 / $0.40 / $64.00 | Improved 2-series voice handling; configurable reasoning and tool use | Default |
| `gpt-realtime-2.1-mini` | $0.60 / $0.06 / $2.40 | $10.00 / $0.30 / $20.00 | Distilled reasoning model for faster, lower-cost voice interactions | Supported opt-in |

The mini is substantially cheaper (for example, 68.75% lower audio input and
output unit prices than full 2.1), and it exposes the same Realtime endpoint,
audio/text inputs and outputs, image input, and function calling. It is not
made the default because the local fixture does not grade response quality and
the product default prioritizes voice quality and interruption robustness. Set
`ZEMORY_REALTIME__MODEL=gpt-realtime-2.1-mini` to opt in without editing
configuration files.

For long sessions, the default session also applies the cost guide's
`truncation.retention_ratio=0.8` and
`truncation.token_limits.post_instructions=8000` example. This leaves cache
headroom by dropping a larger old-history segment only when truncation occurs;
it trades away some distant conversation memory for lower token growth and
fewer repeated cache invalidations.

Sources:

- [GPT-Realtime-2.1 model and pricing](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [GPT-Realtime-2.1 mini model and pricing](https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini)

## Live compatibility and latency evidence

The aggregate results and all 24 numeric samples are preserved in the
[migration benchmark artifact](../benchmarks/2026-07-26-realtime-model-migration/).

Both models completed the same four macOS `say` fixtures over two trials each
(eight samples per model) at 24 kHz PCM, with Realtime audio output routed to
the local speaker callback. This is a
forced-commit measurement (`turn_detection=none`), so it isolates model/API
and playback latency; it does not measure automatic VAD behavior or semantic
response quality.

| Model | n | invalid / early cutoff | Device-playback p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| `gpt-realtime-2` historical baseline | 8 | 0 / 0 | 957.0 ms | 1354.2 ms |
| `gpt-realtime-2.1` | 8 | 0 / 0 | 748.9 ms | 1307.4 ms |
| `gpt-realtime-2.1-mini` | 8 | 0 / 0 | 946.9 ms | 1161.2 ms |

These p50/p95 values use the repository's nearest-rank method. At n=8, p95
is the observed maximum, so the table is a small-run compatibility signal and
not a population-tail estimate.

The full 2.1 sample was 208.1 ms lower at p50 and 46.8 ms lower at p95 than
the historical full-2 baseline. The mini was available and stable in this
fixture; it had a similar p50 but lower measured tail. These are small,
network-sensitive samples, not a quality comparison or a broad performance
claim.

The first automatic server-VAD attempt timed out after the runtime changed VAD
to `create_response=false` to stop empty-transcript self-response loops: the
old harness still waited for server-owned response creation. The harness was
then changed to match production by calling `response.create` only after a
non-empty transcription, with a regression test for that gate.

The repaired app-owned server-VAD path completed the same four fixtures over
two trials (8/8 samples) with zero invalid samples or early cutoffs. It measured
API first audio rather than local
playback: end-of-source-audio p50/p95 was 1574.5/2013.7 ms; measured from the
server's `speech_stopped` event it was 1099.7/1545.7 ms, with VAD wait
p50/p95 475.2/627.8 ms. This correctness-preserving path therefore misses the
repository's aspirational 700/1200 ms end-to-first-audio gate and is not
presented as a latency win. It is also not directly comparable with the
forced-commit device-playback table because the endpoint and timing boundary
differ.

A separate
[one-sample compatibility smoke](../benchmarks/2026-07-26-realtime-model-migration/max-output-smoke/)
accepted a session configured with `max_output_tokens=512` and returned API
first audio in 687.8 ms. It is not release-gate evidence: the historical safe
event retained its run ID and configuration hash but not the expanded config,
so the cap cannot be independently derived or the hash recomputed from the
repository artifact.
