# Transcript corrector migration to GPT-5.6 Luna (2026-07-26)

The optional transcript-correction hop now defaults to `gpt-5.6-luna` with
`reasoning_effort="none"`. It remains disabled unless
`ZEMORY_TRANSCRIPT_CORRECTION_ENABLED=1` is configured.

## Compatibility and cost

Both `gpt-5.4-mini` and `gpt-5.6-luna` were exercised through Chat
Completions with non-reasoning mode, temperature-0 sampling, and a 512-token
output cap. Temperature 0 does not guarantee identical outputs across calls.
Keeping Chat Completions avoids an unrelated API-surface migration because
both target models support it.

Published per-million-token prices at the time of evaluation:

| Model | Input | Cached input | Output |
| --- | ---: | ---: | ---: |
| `gpt-5.4-mini` | $0.75 | $0.075 | $4.50 |
| `gpt-5.6-luna` | $1.00 | $0.10 | $6.00 |

Luna is therefore 33.3% more expensive for the same mix of uncached input,
cached reads, and output. That comparison excludes GPT-5.6 cache-write tokens,
which OpenAI bills separately at 1.25x the model's standard input price; the
actual ratio can differ when cache-write charges occur. The old source default
was actually `gpt-5-mini`, not `gpt-5.4-mini`; compared with that older default,
Luna is materially more expensive.

The source default does not override deployment configuration. If an existing
repository-root `.env` or exported process environment sets
`ZEMORY_TRANSCRIPT_CORRECTION_MODEL`, that value wins; an exported value also
wins over the same key in `.env`. Set
`ZEMORY_TRANSCRIPT_CORRECTION_MODEL=gpt-5.6-luna` explicitly and restart the
process when migrating an installation pinned to an older model.

## Synthetic Korean ASR A/B

The adopted prompt was tested once per fixture on nine synthetic corrections,
including a prompt-injection case. It treats transcript/history as untrusted
user data and asks the model to copy known names with exact spelling and case.

| Model | Exact results | p50 | Initial common-8 estimated cost |
| --- | ---: | ---: | ---: |
| `gpt-5.4-mini` | 8/9 | 882.8 ms | $0.001159 |
| `gpt-5.6-luna` | 9/9 | 653.6 ms | $0.001534 |

The inexpensive legacy `gpt-5-mini` over-corrected by inserting context words
that were not spoken. A second, more restrictive prompt improved 5.4 mini to
9/9 but reduced Luna from 9/9 to 8/9, so that prompt variant was rejected.

This is a one-call-per-fixture functional sample, not a general model-quality
benchmark; outputs can vary on rerun. It supports the requested migration for
this narrow correction task while making the cost increase explicit.

The aggregate is preserved in the
[machine-readable migration artifact](../benchmarks/2026-07-26-transcript-corrector-luna/).
The original run did not retain per-call token rows, model outputs, or a fixture
hash, so the artifact declares that reproducibility limitation rather than
fabricating missing provenance.

Official references:

- <https://developers.openai.com/api/docs/models/gpt-5.6-luna>
- <https://developers.openai.com/api/docs/models/gpt-5.4-mini>
- <https://developers.openai.com/api/docs/guides/prompt-caching>
- <https://developers.openai.com/api/docs/guides/latest-model>
