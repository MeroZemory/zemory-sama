"""Legacy modules kept for one release after the v0.2 refactor.

``zemory/legacy/zemory_vad/`` is the original Phase-2 package. Its
contents have been absorbed into the unified ``zemory`` pipeline:

* VAD + state machine → :mod:`zemory.vad` + :class:`zemory.providers.turn.silero.SileroTurnDetector`
* Whisper STT         → :class:`zemory.providers.stt.openai_whisper.WhisperSTT`
* Half-duplex loop    → :func:`zemory.orchestrator.run` (local profile)

This directory will be deleted in the release after capability features
(memory, Twitch/YouTube, Patience) are ported. Do not add new code here.
"""

import warnings

warnings.warn(
    "zemory.legacy is retained for one release only; use zemory.orchestrator instead",
    DeprecationWarning,
    stacklevel=2,
)
