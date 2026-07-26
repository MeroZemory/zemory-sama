# GPT-Realtime-2.1 512-token compatibility smoke

This historical one-sample run was collected after configuring the live session
with `max_output_tokens=512`. The session was accepted and returned first audio
687.8 ms after the final source-audio chunk.

It is compatibility-only evidence, not a latency or release-gate benchmark:

- one `ko_short` fixture was run once;
- no device-playback or audible-interrupt boundary was measured;
- the historical sanitizer retained the run ID and configuration hash but
  dropped the expanded configuration, so this repository cannot recompute the
  hash or independently derive the 512-token setting from the event;
- the JSONL below is the original numeric-only safe event and has not been
  rewritten to claim the stronger provenance emitted by the current harness.

| Metric | Value |
| --- | ---: |
| run ID | `f7959d71d418443196b153df717df88a` |
| configuration hash | `21a2e4f2f47ceef9d88cf70f9651da21d781e35c34cc4aef194dafe54ed8ec3a` |
| model | `gpt-realtime-2.1` |
| fixture / trial | `ko_short` / 1 |
| metric boundary | final source chunk → API first audio |
| valid / early cutoff | 1 / 0 |
| observed latency | 687.8 ms |

See [summary.json](summary.json) for the preserved aggregate and
[latency-events.jsonl](latency-events.jsonl) for the original safe event.
