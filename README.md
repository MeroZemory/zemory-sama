# realtime-voice-runtime

Low-latency realtime voice runtime for AI agents and VTuber systems.

`realtime-voice-runtime` is a Python CLI voice agent core. It focuses on the
hard part of a voice-first assistant before UI, avatar rendering, or streaming
integrations: capturing microphone audio, detecting turns, producing the first
audible response quickly, handling interruptions, and keeping the conversation
state coherent over long sessions.

The runtime module is still invoked as `python -m zemory`; `zemory` is the
internal character/runtime namespace inherited from the original prototype.

The default runtime uses OpenAI Realtime GA audio-in/audio-out with
`gpt-realtime-2.1`, low-latency `server_vad` at a 200 ms silence window,
one-short-sentence responses, and direct PCM playback. `semantic_vad`,
external TTS, and a cascade with local Silero turn detection plus OpenAI
transcription remain available as explicit choices rather than the default
fast path.

To control long-session cost without repeatedly invalidating the prompt cache,
the Realtime session uses server-side `retention_ratio=0.8` and a
post-instruction window of 8,000 tokens. This follows OpenAI's
[Realtime cost guidance](https://developers.openai.com/api/docs/guides/realtime-costs);
the tradeoff is that older conversation items can be dropped sooner.
Realtime prompt caching itself is automatic and best-effort rather than an
application toggle. `response.usage` now reports total and text/audio/image
input, cached, and output token splits; separate transcription usage is logged
under `input.transcription_usage` so ASR cost is not confused with model cost.

Startup makes no external TTS warmup request. The removed warmup synthesized a
`.` through ElevenLabs before any user turn, which could incur cost and external
text/audio generation merely by launching the app. External-TTS profiles now
pay any cold-start latency only on the first real response. The optional Luna
transcript corrector also has deterministic character budgets: 2,000 per
stored history entry, 8,000 across rendered history, 4,000 for the current raw
transcript, and 12,000 for the complete prompt. A raw transcript over 4,000
characters bypasses the correction API and is returned unchanged. These are
privacy/cost ceilings in characters, not token limits.

## One-Prompt Setup For Coding Agents

Paste this into Claude Code, Codex, or another repository-aware coding agent:

```text
Set up and validate this repository for local manual testing.

Rules:
- Read the repo instructions and inspect the current tree before changing files.
- Do not read, print, commit, or infer secrets from .env.
- Use built-in agent tools when available: plan for multi-step work, ask_user/request_user_input only when a missing secret or hardware decision blocks progress, and use repository search instead of guessing.
- Keep changes surgical. If setup requires edits, explain why and verify them.

Steps:
1. Inspect README.md, pyproject.toml, config.toml, zemory/config.py, zemory/orchestrator.py, and tests/.
2. Run uv sync --frozen.
3. If .env is missing, create it from .env.example without inventing API keys. Tell the user to set OPENAI_API_KEY; ELEVENLABS_API_KEY is only needed for external TTS profiles.
4. Run uv run pytest tests/, uv run ruff check zemory tests scripts, and uv run python -m compileall zemory tests scripts.
5. If checks pass and OPENAI_API_KEY is present, start a manual voice session with uv run python -m zemory. For a detachable session, use tmux new-session -s realtime-voice-runtime-test 'uv run python -m zemory'.
6. Report the active profile, whether Realtime reached session.configured, where logs are stored, and the stop command.
```

## What It Does

- Streams microphone PCM to OpenAI Realtime and plays audio deltas directly.
- Forwards `server_vad` input through one ordered, bounded 32-frame sender so a
  blocked WebSocket append cannot silently block the microphone pump. Overflow
  is terminal: continuing after an audio hole would corrupt turn semantics.
- Treats inactive, finished, or callback-stalled microphone/speaker devices as
  terminal runtime failures. Silent PCM callbacks remain healthy; acoustic
  silence is not mistaken for a dead device.
- Uses a 200 ms `server_vad` default as the measured stability/tail-latency
  tradeoff; it is not a general fastest-path claim. `semantic_vad` remains
  configurable for more conservative turns.
- Mirrors the user's language by default, so Korean and English conversation
  both stay natural without switching profiles.
- Supports barge-in: clear speaker output, cancel pending TTS work, cancel the
  active Realtime response, then ACK-gate provider history synchronization.
  Realtime audio retains only the heard prefix via truncation; external-TTS
  removes the full assistant item. Interrupted partial text is discarded
  locally; after either remote mutation, only a generic interruption note is
  added without copying generated content.
- Keeps provider boundaries small: turn detection, STT, LLM, and TTS can be
  swapped by profile.
- Includes deadline-bounded memory/tool scheduling primitives for
  `local_cascade`. Retrieval can consume the configured pre-response budget;
  late results are deferred. Realtime profiles do not yet inject this context.
- Records latency metrics and ships a JSONL benchmark checker.
- Runs hardware-free regression tests for the core orchestration contracts.

## Performance Snapshot

Benchmarks were refreshed on 2026-06-27 and 2026-06-28, with model-migration
probes added on 2026-07-26, on macOS Apple Silicon using the `realtime_audio`
profile. Artifacts are numeric-only; raw runtime logs are not committed because
they can contain private transcript text.

Final HTML + SVG report:
[docs/reports/2026-06-28-final-comparison](docs/reports/2026-06-28-final-comparison).

The report is not a cross-repository latency leaderboard. Public voice-agent
repos do not expose the same non-interactive fixture, model, endpoint definition,
and output callback boundary. Cross-repo latency numbers would therefore be
misleading. The fair latency evidence is this repo's internal ablation series:
which local/runtime choices reduced, failed to reduce, or destabilized response
latency under the same harness.

| Question | Evidence | Result |
| --- | --- | --- |
| What is the current default path? | `gpt-realtime-2.1`, audio-in/audio-out, `server_vad` 200 ms, 20 ms input chunks, one-short-sentence prompt plus 512-token server cap, 10 ms speaker callback | [2026-07-26 migration probes](docs/benchmarks/2026-07-26-realtime-model-migration): automatic VAD 8/8 valid but above the aspirational gate; forced commit p50/p95 748.9/1307.4 ms. Those migration samples predate the 512-token cap; a separate one-sample live smoke verified that the capped session is accepted. |
| Is local playback the bottleneck? | Full API-to-device rerun and speaker callback microbench | API first-audio to playback p50 5.3-6.5 ms; local speaker output alone p50 7.404 ms |
| Can exact endpoint commit be faster? | `turn_detection=none` with fixture-known endpoint | [forced commit n8](docs/benchmarks/2026-06-28-forced-commit-device-playback-n8): p50 957.0 ms, upper-bound only |
| Did local endpoint detection beat server VAD? | One-stage Silero endpoint miss10/miss12/miss14 probes | miss10 caused 2/4 early cutoffs; stable miss14 was slower at p50 1654.2 ms and rejected |
| Did smaller input chunks help? | 20 ms vs 10 ms live sweep | 10 ms increased p50 and representative tail; 20 ms kept |
| Why is the server-VAD sender queue 32 frames rather than 64? | 20 ms-paced hung-send injection, n=3 each; 0.5 ms append producer-boundary microbench, n=5 | Queue 32 rejected the first discontinuous frame at median 717.199 ms versus 1417.110 ms for 64 (about 49.4% sooner), so 64 was rejected. Enqueuing 1,000 frames reached the producer boundary in median 20.751 ms versus 671.328 ms with direct await, but full drain was 667.267 ms: this is isolation/failure-detection evidence, not an end-to-end speedup. |
| Does a silent input/output device trip the new health checks? | Local PortAudio probes with payloads discarded immediately | Mic: 123 callback updates over 2.5 s, first callback 166.8 ms, max gap 20.4 ms, zero health failures. Speaker: 53 zero-output samples over 0.6 s, zero inactive/health failures and clean stop. No audible content or API call was used. |
| How quickly does terminal speaker loss reach the task owner? | Deterministic active-to-inactive, no-PCM failure injection, n=20 | 20/20 detected; task-group exit p50/p95/max 52.110/52.396/52.415 ms. The prior implementation was still running at the 200 ms test deadline. |
| Did external repo comparison prove speed superiority? | Setup/readiness/source review of public reference repos | No direct speed ranking. Comparison is used to justify architecture choices and show which claims are measured locally. |

See the numeric artifacts in [docs/benchmarks](docs/benchmarks) and the rendered
report for charts. README intentionally links charts instead of embedding every
SVG, because dense benchmark SVGs are difficult to read at GitHub README widths.

## Status

This is an active research/runtime repo, not a packaged end-user app yet.

Implemented:

- Realtime GA audio-native default profile.
- Realtime text + external TTS profile.
- Local cascade profile with Silero VAD and OpenAI `gpt-4o-transcribe` STT.
- Interrupt bus with cancel/truncate/delete ACK barriers. Interrupted text is
  not retained in local correction history.
- SQLite store/recall primitives and a deadline-bound context scheduler for
  `local_cascade`. Realtime profiles use a null memory provider and do not
  create the configured SQLite file as a startup side effect.
- Bounded input/output queues, callback-health monitoring, an ordered
  server-VAD sender, and deadline-bound per-resource cleanup.
- Latency report utility and benchmark CLI.
- Regression suite with coverage gate.

Not implemented yet:

- Web UI, browser WebRTC, Live2D/VRM, OBS integration.
- Twitch/YouTube/Discord chat adapters.
- Hosted deployment, installable package, or CI workflow.
- Codex-equivalent atomic long-session compaction and vector memory. Realtime
  native retention-ratio truncation is implemented as the current cache/cost
  safeguard.
- Automatic reflection writes, memory pruning, runtime tool registration, and
  scheduler-context injection into the default Realtime profiles. A fresh
  memory DB is not populated by ordinary conversation yet, so an arbitrary
  automatic deletion policy was not added.

## Runtime Profiles

| Profile | Default | Pipeline | Use case |
| --- | --- | --- | --- |
| `realtime_audio` | Yes | OpenAI Realtime GA audio input/output, `server_vad` 200 ms, `NullTTS` | Lowest-latency voice conversation |
| `realtime_text_external_tts` | No | Realtime text output, sentence chunking, external TTS | Character voice quality over minimum latency |
| `local_cascade` | No | Local Silero VAD, OpenAI `gpt-4o-transcribe` STT, Realtime text LLM, external TTS | Local turn-detection fallback and development |
| `research_full_duplex` | No | Placeholder | Future full-duplex research track |

Legacy aliases `realtime` and `local` are still accepted and normalized.
`enable_barge_in` is intentionally limited to the two Realtime profiles:
`local_cascade` has neither retained responding PCM nor echo cancellation, so
enabling it would create an unsafe false-interrupt path and is rejected at
configuration load time.

## Architecture

```text
MicrophoneStream (bounded queue + callback health)
  -> TurnDetector
     -> Realtime: ordered server-VAD sender -> OpenAI Realtime session
     -> local_cascade: Silero -> STT -> AsyncContextScheduler
        (deadline-bound SQLite recall/tools)
  -> Orchestrator (phase + generation/response ownership)
     -> InterruptBus
     -> ResponseStream
        -> audio-native: response.output_audio.delta -> SpeakerStream
        -> external TTS: text delta -> SentenceChunker -> TTSTaskManager
  -> RuntimeCleanup (independent per-resource deadlines)
```

The context scheduler is constructed for every profile as a uniform lifecycle
boundary, but only `local_cascade` receives a real SQLite memory provider and
consumes its result before response creation. The default Realtime profiles
receive a null provider; they neither inject scheduler context nor create a
memory database. There is no parallel transcript ledger: provider conversation
state plus generation-correlated completed-output handling is authoritative;
partial text is transient and is discarded when output is interrupted.

### Provider-State Safety Contracts

- A scoped response cancel is authoritative only when `response.done` names
  that exact response ID; its status alone is not the correlation key. An
  unscoped cancel accepts no `response.done` as its ACK, even a cancelled one.
  Only `response_cancel_not_active` tied to the original cancel `event_id` can
  authorize the single active-conflict retry. Cancel rejection or ACK timeout
  terminates the unsafe session instead of guessing that it is reusable.
- Barge-in and failed, incomplete, or unheard output may require a remote
  truncate/delete. A new `response.create` cannot cross that mutation barrier;
  rejection or timeout fails the session closed. For normal completion with
  unheard output, the runtime also stays `RESPONDING` until delete ACK arrives.
- A speculative correction uses a strict transaction order: cancel terminal
  ACK, assistant truncate/delete ACK, raw user-item delete ACK, then corrected
  replacement request. Any ambiguous send or unacknowledged mutation ends the
  session rather than leaving mixed raw/corrected provider history.
- In manual Realtime turn mode, a local commit timeout/send exception is
  ambiguous and terminal. A later correlated commit error permits reuse only
  after `input_audio_buffer.clear` receives its ACK; clear rejection/timeout is
  terminal, and a stale old-generation error cannot clear a new input buffer.
- Automatic STT/TTS retries are limited to failures that prove request
  connection establishment did not complete. ElevenLabs retries once for a
  pre-audio `ConnectError` or `ConnectTimeout`; Whisper retries once only for
  an SDK `APIConnectionError` directly caused by `ConnectError`. Read/write
  failures, other timeouts, and HTTP 429/5xx are single-attempt failures to
  avoid duplicate transcription, synthesis, audio, and billing. Provider SDK
  retries are disabled where the adapter owns this policy.

Important modules:

- `zemory/orchestrator.py` wires the runtime tasks.
- `zemory/config.py` builds profile-specific Realtime session config.
- `zemory/providers/llm/openai_realtime.py` adapts GA Realtime events.
- `zemory/pipeline/interrupt_bus.py` owns the abort chain.
- `zemory/pipeline/context.py` owns memory and async context scheduling.
- `zemory/observability/latency_report.py` parses latency JSONL.

## Requirements

- Python 3.12+
- macOS, Linux, or another environment supported by `sounddevice`
- Working microphone and speaker device
- OpenAI API key
- ElevenLabs API key only for external TTS profiles

The default `realtime_audio` profile does not require ElevenLabs.

## Quick Start

Install dependencies:

```bash
uv sync --frozen
```

Create local secrets:

```bash
test -f .env || cp .env.example .env
```

The repository-root `.env` file is loaded automatically; it does not need to
be sourced in the shell. The command above preserves an existing `.env`, which
may contain secrets or model overrides.

An existing `.env` can therefore keep an older model even after the source
default is migrated. Check the startup log's effective profile/model, and use a
one-command process override when you need to force the migrated defaults:

```bash
ZEMORY_REALTIME__MODEL=gpt-realtime-2.1 \
ZEMORY_TRANSCRIPT_CORRECTION_MODEL=gpt-5.6-luna \
uv run python -m zemory
```

On a shared Unix/macOS machine, restrict locally stored secrets and memory
after creating them (when the memory database exists):

```bash
chmod 600 .env
test ! -f .zemory/memory.sqlite3 || chmod 600 .zemory/memory.sqlite3
```

Set at least:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Run the voice agent:

```bash
uv run python -m zemory
```

For long-running manual testing:

```bash
tmux new-session -s realtime-voice-runtime-test 'uv run python -m zemory'
```

Stop with `Ctrl+C` or:

```bash
tmux kill-session -t realtime-voice-runtime-test
```

## Configuration

The user-facing config load order is:

1. Variables already exported in the process environment win over the same
   variable name loaded from `.env`.
2. Values from the automatically loaded repository-root `.env` for variable
   names that were not already exported. If both provider-key aliases exist,
   the `ZEMORY_*` alias wins over the legacy unprefixed name.
3. `config.toml`.
4. Defaults in `zemory/config.py`.

Configuration variables use the `ZEMORY_` prefix; nested settings use `__`
(for example, `ZEMORY_REALTIME__MODEL`). The provider keys also accept the
legacy names `OPENAI_API_KEY` and `ELEVENLABS_API_KEY`. Existing environment or
`.env` model overrides therefore take precedence over a newer source default.
The SDK's ambient `OPENAI_BASE_URL` variable is deliberately ignored so it
cannot silently redirect credentials; use the validated
`ZEMORY_OPENAI_BASE_URL` setting for an explicit compatible endpoint.

Common values:

```bash
ZEMORY_PROFILE=realtime_audio
ZEMORY_OPENAI_BASE_URL=https://api.openai.com/v1
ZEMORY_REALTIME__MODEL=gpt-realtime-2.1
ZEMORY_ENABLE_BARGE_IN=0
ZEMORY_RESPONSE_LENGTH="one short sentence"
ZEMORY_MEMORY_ENABLED=1
ZEMORY_MEMORY_PATH=.zemory/memory.sqlite3
ZEMORY_MEMORY_RECALL_DEADLINE_MS=80
ZEMORY_CONTEXT_TOOL_DEADLINE_MS=200
ZEMORY_TRANSCRIPT_CORRECTION_MODEL=gpt-5.6-luna
ZEMORY_TRANSCRIPT_CORRECTION_TIMEOUT_S=5
```

Barge-in is off by default because laptop speakers can leak into the mic and
cause false interrupts without hardware echo cancellation. Enable it when using
headphones or an AEC-capable device.

## Testing

Run the full suite:

```bash
uv run pytest tests/
```

Run lint:

```bash
uv run ruff check zemory tests scripts
```

Compile-check all Python files:

```bash
uv run python -m compileall zemory tests scripts
```

Current local verification target:

- 436 tests passing in 9.50 s.
- 80% coverage gate.
- Whole-package coverage: 87.69%.

## Latency Benchmarks

Runtime logs emit `turn.complete` events with timing fields such as
`total_ms`, `first_tts_byte_ms`, `speaker_buffer_ms`, and `interrupted`.
In the CLI runtime, `total_ms` is measured to the first output callback that
actually consumes response audio, not just the moment audio enters the local
speaker buffer.
The microphone and speaker streams request the device's low-latency mode in
addition to the 10 ms output callback block.

Generate README-ready artifacts from a local runtime log:

```bash
uv run python scripts/build_benchmark_artifacts.py \
  --log .zemory/run.log \
  --out docs/benchmarks/local-run \
  --title "realtime-voice-runtime realtime_audio local benchmark"
```

The live benchmark harness is macOS-only (`say` plus `ffmpeg`); the runtime
itself can still run on the other supported platforms. These commands make
real, billable OpenAI calls. Run a live generated-audio smoke check only when
that cost is intended. It verifies fixture completion and API first-audio
events, but does not measure local playback or audible interruption and is not
release-gate evidence:

```bash
uv run python scripts/bench_realtime_audio_fixture.py \
  --out docs/benchmarks/local-live-smoke \
  --turn-detection server_vad \
  --server-vad-silence-ms 200
```

For a release candidate, collect eight device-playback turn samples and eight
separate audible barge-in samples, then run the strict gate. This is also a
billable live run; the barge-in fixtures deliberately request a longer response
so cancellation can be measured:

```bash
uv run python scripts/bench_realtime_audio_fixture.py \
  --out docs/benchmarks/local-live-release \
  --turn-detection server_vad \
  --server-vad-silence-ms 200 \
  --play-output \
  --measure-interrupt \
  --trials 2

uv run python scripts/bench_latency.py \
  docs/benchmarks/local-live-release/latency-events.jsonl
```

Use `--min-interrupt-samples 0` only for an explicitly turn-only gate. It makes
an absent interrupt target optional; any interrupt events that are present are
still validated.

Default thresholds:

- turn p50 <= 700 ms
- turn p95 <= 1200 ms
- interrupt p95 <= 150 ms

The controlled 2026-06-27 fixture reports input-stream end/commit to first
response audio because synthetic TTS clips do not always trigger
`semantic_vad` `speech_stopped` consistently.

## Repository Layout

```text
zemory/
  audio.py                  # microphone and speaker streams
  config.py                 # settings and Realtime session shape
  orchestrator.py           # runtime task wiring
  pipeline/                 # chunking, interrupts, memory, context, TTS manager
  providers/                # LLM, STT, TTS, turn detector providers
  observability/            # metrics, logging, latency reports
tests/                      # executable behavior contracts
scripts/                    # operational scripts
docs/
  plans/                    # architecture and migration plans
  ref/                      # reference project analysis
```

## Design Notes

The current design intentionally chooses a simple Realtime-first path:

- audio-native output before external TTS by default
- server-side turn detection before local heuristics
- deadline-bound local-cascade context retrieval before unbounded RAG
- profile-aware resources: default Realtime startup has no unused SQLite side
  effect
- explicit profiles before runtime magic
- no legacy preview Realtime path

See [docs/plans/optimal-low-latency-design.md](docs/plans/optimal-low-latency-design.md)
for the detailed design.

## Security And Privacy

- Do not commit `.env`.
- `.zemory/` is ignored and can contain local memory or runtime logs.
- Microphone audio is sent to OpenAI in every implemented profile: it is
  streamed continuously in Realtime profiles and uploaded once per completed
  utterance to `/audio/transcriptions` in `local_cascade`. Silero turn
  detection is local; the cascade STT is not.
- External TTS profiles send assistant text to the configured TTS provider.
- Review runtime logs before sharing them; they may contain conversation text.

## License

No license has been specified yet.
