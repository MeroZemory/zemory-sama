"""Provider abstraction layer (xsai-inspired).

Four protocols — ``TurnDetector``, ``STTProvider``, ``LLMProvider``,
``TTSProvider`` — let the orchestrator compose a pipeline without knowing
the concrete backend. A profile selects one implementation per protocol.
"""

from zemory.providers.base import (
    Injection,
    LLMProvider,
    STTProvider,
    TTSProvider,
    TurnDetector,
    TurnEvent,
    build_pipeline,
)

__all__ = [
    "Injection",
    "LLMProvider",
    "STTProvider",
    "TTSProvider",
    "TurnDetector",
    "TurnEvent",
    "build_pipeline",
]
