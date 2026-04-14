"""OpenAI Realtime API LLMProvider adapter.

Normalizes Realtime events to the generic schema documented in
:class:`zemory.providers.base.LLMProvider`. Audio push / text inject /
response cancel are thin wrappers around the underlying SDK methods.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from zemory.config import build_session_config, settings
from zemory.observability import get_logger
from zemory.providers.base import Injection

_log = get_logger(__name__)


class OpenAIRealtimeLLM:
    """Adapter over :class:`openai.AsyncOpenAI` beta realtime connect."""

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._conn_cm = None
        self._conn = None
        self._opened = asyncio.Event()
        self._events_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pump_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def open_session(self) -> None:
        self._conn_cm = self._client.beta.realtime.connect(model=settings.realtime.model)
        self._conn = await self._conn_cm.__aenter__()
        await self._conn.session.update(session=build_session_config())
        self._pump_task = asyncio.create_task(self._pump_events())
        self._opened.set()
        _log.info("llm.session.opened", model=settings.realtime.model,
                  profile=settings.profile)

    async def close(self) -> None:
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
        if self._conn_cm is not None:
            try:
                await self._conn_cm.__aexit__(None, None, None)
            except Exception:  # pragma: no cover — best-effort teardown
                pass

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    async def push_audio(self, pcm_bytes: bytes) -> None:
        if not self._conn:
            return
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        await self._conn.input_audio_buffer.append(audio=encoded)

    async def clear_input_buffer(self) -> None:
        if self._conn:
            await self._conn.input_audio_buffer.clear()

    async def send_user_text(
        self, text: str, injections: list[Injection] | None = None
    ) -> None:
        """Inject text as a user-turn and request a response.

        Used in the local profile after Whisper STT completes. Priority-
        ordered injections are concatenated as system messages before the
        user turn (cheaper than a full PromptAssembler for now).
        """
        if not self._conn:
            return
        # Inject context items first (ascending priority → earliest)
        for inj in sorted(injections or [], key=lambda i: i.priority):
            await self._conn.conversation.item.create(
                item={
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": inj.text}],
                }
            )
        await self._conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        )
        await self._conn.response.create()

    async def cancel_current(self) -> None:
        """Ask Realtime to abort the in-flight response."""
        if not self._conn:
            return
        try:
            await self._conn.response.cancel()
        except Exception as e:  # pragma: no cover
            _log.warning("llm.cancel.failed", error=str(e))

    async def delete_item(self, item_id: str) -> None:
        if self._conn:
            try:
                await self._conn.conversation.item.delete(item_id=item_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    async def events(self) -> AsyncIterator[dict]:
        while True:
            event = await self._events_queue.get()
            if event is None:
                break
            yield event

    async def _pump_events(self) -> None:
        """Read raw Realtime events and normalize them to the Provider schema."""
        assert self._conn is not None
        try:
            async for event in self._conn:
                normalized = self._normalize(event)
                if normalized is not None:
                    await self._events_queue.put(normalized)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover
            _log.error("llm.event.pump_error", error=str(e))
            await self._events_queue.put({"type": "error", "error": str(e)})

    @staticmethod
    def _normalize(event: Any) -> dict | None:
        t = getattr(event, "type", None)
        if t == "session.created":
            return {"type": "session.created", "session_id": event.session.id}
        if t == "session.updated":
            return {"type": "session.updated"}
        if t == "input_audio_buffer.speech_started":
            return {"type": "input.speech_started"}
        if t == "input_audio_buffer.speech_stopped":
            return {"type": "input.speech_stopped"}
        if t == "conversation.item.input_audio_transcription.completed":
            return {"type": "input.transcript", "text": event.transcript}
        if t == "conversation.item.created":
            item = getattr(event, "item", None)
            item_id = getattr(item, "id", None) if item else None
            return {"type": "conversation.item.created", "item_id": item_id}
        if t == "response.text.delta":
            return {"type": "text.delta", "delta": event.delta}
        if t == "response.text.done":
            return {"type": "text.done"}
        if t == "response.done":
            usage = getattr(event.response, "usage", None)
            return {
                "type": "response.done",
                "usage": {"total_tokens": usage.total_tokens} if usage else None,
            }
        if t == "error":
            return {"type": "error", "error": getattr(event, "error", "unknown")}
        return None
