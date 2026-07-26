# realtime-voice-runtime Project Intent

> Date: 2026-06-27
> Last validated: 2026-07-26
> Scope: tracked source, tests, configuration, and design documents. `.env` is
> intentionally excluded because it may contain secrets.

## One Sentence

`zemory-sama` is a Python runtime for multilingual, low-latency realtime voice
conversation. Its current purpose is to make the conversation core fast and
reliable before adding avatar rendering, web UI, or streaming platform
integrations.

## What The Project Optimizes For

The project is optimized for the moment after a user stops speaking:

- detect the end of the user turn naturally
- start audible assistant output quickly
- allow the user to interrupt the assistant
- mirror the user's spoken language, including Korean and English
- preserve enough transcript and memory context for coherent follow-up turns
- keep provider boundaries simple enough to replace model, VAD, STT, or TTS

The default fast path is now `realtime_audio`: OpenAI Realtime GA audio
input/output with `gpt-realtime-2.1`, `server_vad` at a 200 ms silence window,
one-short-sentence responses, and direct PCM playback. Microphone frames cross
an ordered 32-frame sender boundary; overflow or provider-send failure is
terminal because continuing after a dropped frame would make the server's
audio stream discontinuous.

## Current Runtime Profiles

| Profile | Purpose |
| --- | --- |
| `realtime_audio` | Default low-latency audio-native path. No external TTS required. |
| `realtime_text_external_tts` | Realtime text output plus external TTS for specific voice quality. |
| `local_cascade` | Local Silero VAD and Whisper STT fallback path. |
| `research_full_duplex` | Placeholder for future full-duplex research experiments. |

Legacy aliases `realtime` and `local` are accepted by configuration but the old
preview Realtime implementation has been removed.

## Implemented

- Unified `zemory.orchestrator.run()` runtime.
- Profile-based provider wiring for turn detection, STT, LLM, and TTS.
- OpenAI Realtime GA adapter using `client.realtime.connect`.
- Audio-native response routing via `response.output_audio.delta`.
- External TTS route with sentence chunking and ordered parallel playback.
- `InterruptBus` for speaker clear, TTS abort, LLM cancel, and partial capture.
- Generation/response-correlated provider history; there is no independent
  shadow transcript ledger.
- SQLite local memory store and deadline-bound async context scheduler for
  `local_cascade`. Realtime profiles use a null memory provider and do not
  create the SQLite file during startup.
- Bounded microphone/speaker queues, input/output callback-health monitoring,
  and fail-closed device-loss propagation.
- Per-resource deadline cleanup isolated from turn orchestration.
- Latency report utility and `scripts/bench_latency.py`.
- Hardware-free regression tests for core contracts.

## Deferred

- Browser WebRTC client.
- Live2D/VRM/OBS avatar output.
- Twitch, YouTube, Discord, and other side-channel chat sources.
- Production long-session compaction and vector memory.
- Acoustic echo cancellation and automatic device/transport reconnect.
- Hosted deployment and CI automation.
- Full-duplex local speech model experiments.

## Design Principles

- Prefer audio-native Realtime output for the default latency path.
- Keep external TTS as an explicit profile, not hidden in the fast path.
- Keep context retrieval deadline-bound so memory and tools cannot block first
  audio.
- Keep interruption state explicit and tested.
- Fail closed on discontinuous audio, dead device callbacks, or unknown
  provider ownership instead of silently continuing a corrupted session.
- Create persistent resources only in profiles that actually consume them.
- Do not preserve stale preview API paths in runtime code.

## Success Criteria

This stage is successful when:

- realtime audio first output stays near the configured benchmark targets
  (`p50 <= 700 ms`, `p95 <= 1200 ms`)
- interrupt-to-silence remains within the `p95 <= 150 ms` target on suitable
  hardware
- core behavior remains covered by tests without requiring live API calls or
  audio hardware
- real manual sessions can run with microphone and speaker devices without
  configuration drift from the tested contracts

The latency thresholds are aspirational gates, not a claim that every current
sample passes. The 2026-07-26 automatic server-VAD n=8 run was valid without
early cutoff but measured p50/p95 1574.5/2013.7 ms from final source audio to
first API audio. Correctness and liveness changes remain accepted; they are not
reported as an end-to-end latency win.

## Runtime Configuration Contract

The repository-root `.env` is loaded automatically and intentionally remains
outside this document's review scope. An exported process variable overrides
the same `.env` variable, and `.env` overrides `config.toml` and source
defaults. An old model override can therefore mask a source migration; the
startup log's effective profile/model is authoritative for a manual run.

## Primary References

- [docs/plans/optimal-low-latency-design.md](plans/optimal-low-latency-design.md)
- [docs/ref/current-state-2026-06-27.md](ref/current-state-2026-06-27.md)
- [docs/ref/comparison.md](ref/comparison.md)
- [README.md](../README.md)
