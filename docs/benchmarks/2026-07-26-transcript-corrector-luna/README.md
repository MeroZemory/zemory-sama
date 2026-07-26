# Transcript-corrector Luna migration evidence

This directory preserves the machine-readable aggregate used for the
2026-07-26 transcript-corrector model decision. The adopted prompt produced
8/9 exact fixture results with `gpt-5.4-mini` and 9/9 with `gpt-5.6-luna`;
observed p50 latency was 882.8 ms and 653.6 ms respectively.

Each fixture was called once with temperature-0 sampling, which does not make
model output deterministic; a rerun may produce different exact-match counts.

The original run did not persist per-call outputs, usage-token rows, or a
fixture-corpus hash. `summary.json` therefore records that provenance gap
explicitly. The aggregate is useful as narrow decision evidence, but it is not
a fully reproducible model-quality benchmark and should not be used to infer a
general quality ranking.
