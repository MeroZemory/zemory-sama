# Codex compaction compatibility decision (2026-07-26)

## Source pinned and reviewed

The comparison used the official `openai/codex` repository at commit
`62fd410384cca008446c2d64a4f2b3f915f4906e` (the repository HEAD when this
review was refreshed). The relevant Rust sources were:

- `codex-rs/core/src/compact.rs`
- `codex-rs/core/src/compact_remote.rs`
- `codex-rs/core/src/compact_remote_v2.rs`
- `codex-rs/core/src/compact_token_budget.rs`
- `codex-rs/core/src/context_manager/history.rs`
- `codex-rs/core/src/session/context_window.rs`
- `codex-rs/core/src/session/turn.rs`
- `codex-rs/prompts/templates/compact/prompt.md`
- `codex-rs/prompts/templates/compact/summary_prefix.md`

## What exact parity would require

Codex triggers automatic compaction at its scoped token limit (normally 90%
of the model context window), keeps the full context limit as a hard cap,
distinguishes pre-turn from mid-turn context injection, preserves recent user
messages under a 20k-token local budget, and atomically replaces its local
history. Remote v2 requires exactly one opaque compaction output item. It
considers recent system/developer/user messages, applies the canonical filter
(including removal of stale developer and non-user-content wrappers), keeps the
remaining eligible messages under a separate 64k-token budget, then installs
those messages plus the compaction item as one replacement. It also handles
repeated windows, model/context downshifts, comp-hash changes, and
previous-model to current-model fallback.

## Why the literal port was rejected

Zemory uses the Realtime WebSocket conversation, not Codex's Responses history
manager. Realtime exposes item-by-item `conversation.item.delete` and
`conversation.item.create`; it does not expose Codex's atomic
`replacement_history` installation or a model-visible encrypted remote-v2
compaction item. Applying a multi-item replacement in place could leave a
partially deleted conversation after cancellation or transport failure, and
editing the prefix repeatedly also reduces prompt-cache hits.

A throwaway dependency-free prototype (about 1.3k lines, with 33 focused
tests) was built to validate the mapping. It intentionally had no production
caller. Keeping that dead parallel history implementation would increase
maintenance and create a false claim of runtime compaction, so it was removed.
No patch artifact was retained; these counts are work-log metadata, not
independently reproducible release evidence.

## Runtime decision

Use Realtime's native retention-ratio truncation at the session boundary and
do not delete one oldest item on every turn. This is the transport's supported
cost/cache mechanism: dropping a larger batch creates headroom and avoids
busting the prompt prefix on every subsequent response. Only the optional
transcript-corrector history and per-turn memory recall count are separately
bounded; there is no canonical local transcript window. Durable SQLite memory
storage itself is not pruned yet and can grow without a configured retention
cap.

This is intentionally **not described as 100% Codex compaction parity**. A
future parity-oriented implementation needs a turn-boundary session rollover:
build a Codex-shaped textual checkpoint off-session, open a replacement
Realtime session, inject the complete replacement history, verify it, and only
then retire the old session. That design makes the replacement atomic from the
orchestrator's point of view and can be benchmarked for latency, cost, and
summary fidelity before adoption.
