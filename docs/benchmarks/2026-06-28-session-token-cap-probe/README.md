# Session Token Cap Probe - 2026-06-28

Goal: test whether adding a Realtime session-level
`max_response_output_tokens=512` cap can reduce runaway voice response work
without changing the low-latency automatic `server_vad` response path.

## Result

Rejected. Do not send `session.max_response_output_tokens` in the current
Realtime GA session update payload.

The live probe failed before a latency sample could be recorded:

```text
RealtimeError(message="Unknown parameter: 'session.max_response_output_tokens'.", type='invalid_request_error', code='unknown_parameter', param='session.max_response_output_tokens')
```

After removing the field, the same server VAD 200 ms path was rechecked and
returned valid first-audio samples:

| Recheck | Events | Valid turns | Early cutoffs | p50 | Representative max |
| --- | ---: | ---: | ---: | ---: | ---: |
| no session token cap, 20 ms input chunks | 2 | 2 | 0 | 1102.7 ms | 1261.0 ms |

## Decision

Keep the automatic `server_vad` `create_response=true` path unchanged. A
response-level token cap could be tested only if response creation is moved out
of the server VAD auto-response path, but that would add a local event round
trip and needs a separate latency/correctness experiment.

## Artifact

- [session-cap-rejected-recheck](../2026-06-28-session-cap-rejected-recheck)
