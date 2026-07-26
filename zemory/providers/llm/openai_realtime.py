"""OpenAI Realtime API LLMProvider adapter.

Normalizes Realtime events to the generic schema documented in
:class:`zemory.providers.base.LLMProvider`. Audio push / text inject /
response cancel are thin wrappers around the underlying SDK methods.
"""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from zemory.config import build_session_config, settings
from zemory.observability import get_logger
from zemory.providers.base import Injection

_log = get_logger(__name__)
_EVENTS_CLOSED = object()
_EVENT_QUEUE_MAXSIZE = 256
_MAX_CONTEXT_ITEMS = 32
_MAX_CONTEXT_CHARS = 16_000
_SESSION_UPDATE_TIMEOUT_S = 5.0
_PENDING_OPERATION_LIMIT = 512
_UNTRUSTED_CONTEXT_PREFIX = "[BEGIN UNTRUSTED CONTEXT DATA]\n"
_UNTRUSTED_CONTEXT_SUFFIX = "\n[END UNTRUSTED CONTEXT DATA]"


class OpenAIRealtimeLLM:
    """Adapter over :class:`openai.AsyncOpenAI` Realtime GA connect."""

    def __init__(self, api_key: str, client: Any | None = None) -> None:
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=settings.openai_base_url,
        )
        self._conn_cm = None
        self._conn = None
        self._opened = asyncio.Event()
        self._events_queue: asyncio.Queue[dict | object] = asyncio.Queue(
            maxsize=_EVENT_QUEUE_MAXSIZE
        )
        self._pump_task: asyncio.Task | None = None
        self._session_ready: asyncio.Future[None] | None = None
        self._operation_seq = 0
        self._pending_operations: OrderedDict[str, dict[str, Any]] = OrderedDict()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def open_session(self) -> None:
        self._conn_cm = self._client.realtime.connect(model=settings.realtime.model)
        self._conn = await self._conn_cm.__aenter__()
        self._session_ready = asyncio.get_running_loop().create_future()
        self._pump_task = asyncio.create_task(self._pump_events())
        try:
            event_id = self._track_operation("session.update")
            await self._conn.session.update(
                session=build_session_config(),
                event_id=event_id,
            )
            async with asyncio.timeout(_SESSION_UPDATE_TIMEOUT_S):
                await asyncio.shield(self._session_ready)
        except BaseException:
            await self.close()
            raise
        self._opened.set()
        _log.info("llm.session.opened", model=settings.realtime.model,
                  profile=settings.profile)

    async def close(self) -> None:
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
        if self._pump_task is not None:
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - pump reports via event stream
                pass
            self._pump_task = None
        if self._conn_cm is not None:
            try:
                await self._conn_cm.__aexit__(None, None, None)
            except Exception:  # pragma: no cover — best-effort teardown
                pass
            self._conn_cm = None
        self._conn = None
        self._session_ready = None
        self._pending_operations.clear()
        self._opened.clear()
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    async def push_audio(self, pcm_bytes: bytes) -> None:
        if not self._conn:
            return
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        await self._conn.input_audio_buffer.append(audio=encoded)

    async def clear_input_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        if self._conn:
            event_id = self._track_operation(
                "input_audio_buffer.clear",
                generation_id=generation_id,
            )
            try:
                await self._conn.input_audio_buffer.clear(event_id=event_id)
            except BaseException:
                self._pending_operations.pop(event_id, None)
                raise

    async def commit_input_audio_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        if self._conn:
            event_id = self._track_operation(
                "input_audio_buffer.commit",
                generation_id=generation_id,
            )
            try:
                await self._conn.input_audio_buffer.commit(event_id=event_id)
            except BaseException:
                self._pending_operations.pop(event_id, None)
                raise

    async def send_user_text(
        self,
        text: str,
        injections: list[Injection] | None = None,
        *,
        generation_id: int | None = None,
    ) -> None:
        """Inject text as a user-turn and request a response.

        Used in the local profile after Whisper STT completes. Priority-
        ordered injections are concatenated as system messages before the
        user turn (cheaper than a full PromptAssembler for now).
        """
        if not self._conn:
            return
        # Inject context items first (ascending priority → earliest). Dynamic
        # memory/tool output never inherits system authority by source name;
        # only an explicitly code-curated trusted_instruction may do so.
        remaining_chars = _MAX_CONTEXT_CHARS
        for inj in sorted(injections or [], key=lambda i: i.priority)[
            :_MAX_CONTEXT_ITEMS
        ]:
            bounded_text = inj.text[:remaining_chars]
            if not bounded_text:
                continue
            remaining_chars -= len(bounded_text)
            trusted = inj.trust == "trusted_instruction"
            context_text = (
                bounded_text
                if trusted
                else (
                    _UNTRUSTED_CONTEXT_PREFIX
                    + bounded_text
                    + _UNTRUSTED_CONTEXT_SUFFIX
                )
            )
            await self._create_conversation_item(
                {
                    "type": "message",
                    "role": "system" if trusted else "user",
                    "content": [{"type": "input_text", "text": context_text}],
                },
                generation_id=generation_id,
                purpose="user_input",
            )
            if remaining_chars <= 0:
                break
        await self._create_conversation_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
            generation_id=generation_id,
            purpose="user_input",
        )
        await self._create_response(generation_id=generation_id)

    async def record_system_note(self, text: str) -> None:
        """Add a non-response context note to the current Realtime conversation."""
        if not self._conn:
            return
        await self._create_conversation_item(
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": text}],
            },
            purpose="system_note",
        )

    async def cancel_current(self, response_id: str | None = None) -> None:
        """Ask Realtime to abort the in-flight response."""
        if not self._conn:
            return
        event_id = self._track_operation(
            "response.cancel",
            response_id=response_id,
        )
        if response_id:
            try:
                await self._conn.response.cancel(
                    response_id=response_id,
                    event_id=event_id,
                )
            except BaseException:
                self._pending_operations.pop(event_id, None)
                raise
        else:
            try:
                await self._conn.response.cancel(event_id=event_id)
            except BaseException:
                self._pending_operations.pop(event_id, None)
                raise

    async def trigger_response(self, *, generation_id: int | None = None) -> None:
        """Ask Realtime to generate a response for the current conversation.

        Used when ``create_response=false`` is set (e.g. when transcript
        correction takes ownership of response timing) and we want to
        generate a reply without injecting a new user item.
        """
        if not self._conn:
            return
        await self._create_response(generation_id=generation_id)

    async def _create_response(self, *, generation_id: int | None) -> None:
        """Create a response tagged with its local orchestration generation."""
        assert self._conn is not None
        event_id = self._track_operation(
            "response.create",
            generation_id=generation_id,
        )
        try:
            if generation_id is None:
                await self._conn.response.create(event_id=event_id)
                return
            await self._conn.response.create(
                response={"metadata": {"zemory_generation": str(generation_id)}},
                event_id=event_id,
            )
        except BaseException:
            self._pending_operations.pop(event_id, None)
            raise

    async def delete_item(self, item_id: str) -> bool:
        if self._conn:
            event_id = self._track_operation(
                "conversation.item.delete",
                item_id=item_id,
            )
            try:
                await self._conn.conversation.item.delete(
                    item_id=item_id,
                    event_id=event_id,
                )
                return True
            except Exception:
                self._pending_operations.pop(event_id, None)
                return False
        return False

    async def truncate_item(
        self,
        item_id: str,
        *,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Remove audio the user did not hear from the server conversation."""
        if not self._conn:
            return
        event_id = self._track_operation(
            "conversation.item.truncate",
            item_id=item_id,
        )
        try:
            await self._conn.conversation.item.truncate(
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=max(0, audio_end_ms),
                event_id=event_id,
            )
        except BaseException:
            self._pending_operations.pop(event_id, None)
            raise

    async def _create_conversation_item(
        self,
        item: dict[str, Any],
        *,
        generation_id: int | None = None,
        purpose: str,
    ) -> None:
        assert self._conn is not None
        event_id = self._track_operation(
            "conversation.item.create",
            generation_id=generation_id,
            item_create_purpose=purpose,
        )
        try:
            await self._conn.conversation.item.create(
                item=item,
                event_id=event_id,
            )
        except BaseException:
            self._pending_operations.pop(event_id, None)
            raise

    def _track_operation(self, operation: str, **metadata: Any) -> str:
        """Assign a bounded client event ID so delayed errors stay attributable."""
        self._operation_seq += 1
        event_id = f"zemory:{self._operation_seq}"
        self._pending_operations[event_id] = {
            "operation": operation,
            **metadata,
        }
        while len(self._pending_operations) > _PENDING_OPERATION_LIMIT:
            self._pending_operations.popitem(last=False)
        return event_id

    def _pop_matching_operation(
        self,
        operation: str,
        **metadata: Any,
    ) -> dict[str, Any] | None:
        for event_id, pending in tuple(self._pending_operations.items()):
            if pending.get("operation") != operation:
                continue
            if any(pending.get(key) != value for key, value in metadata.items()):
                continue
            self._pending_operations.pop(event_id, None)
            return pending
        return None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    async def events(self) -> AsyncIterator[dict]:
        while True:
            event = await self._events_queue.get()
            if event is _EVENTS_CLOSED:
                self._events_queue.task_done()
                break
            assert isinstance(event, dict)
            try:
                yield event
            finally:
                self._events_queue.task_done()

    def _put_terminal_event_nowait(
        self,
        event: dict | object,
        *,
        event_type: str,
    ) -> None:
        """Force a terminal event into the bounded queue without blocking."""
        try:
            self._events_queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            dropped = self._events_queue.get_nowait()
            self._events_queue.task_done()

        dropped_type = (
            dropped.get("type") if isinstance(dropped, dict) else "stream.closed"
        )
        _log.error(
            "llm.event.queue_full_terminal_eviction",
            inserted_type=event_type,
            dropped_type=dropped_type,
        )
        self._events_queue.put_nowait(event)

    async def _pump_events(self) -> None:
        """Read raw Realtime events and normalize them to the Provider schema."""
        assert self._conn is not None
        try:
            async for event in self._conn:
                normalized = self._normalize(event)
                if normalized is not None:
                    event_type = normalized.get("type")
                    if event_type == "error":
                        client_event_id = normalized.get("client_event_id")
                        operation = (
                            self._pending_operations.pop(client_event_id, None)
                            if isinstance(client_event_id, str)
                            else None
                        )
                        if operation is not None:
                            normalized.update(operation)
                    elif event_type == "session.updated":
                        self._pop_matching_operation("session.update")
                    elif event_type == "input.committed":
                        operation = self._pop_matching_operation(
                            "input_audio_buffer.commit"
                        )
                        if operation is not None:
                            normalized.update(operation)
                    elif event_type == "input.cleared":
                        operation = self._pop_matching_operation(
                            "input_audio_buffer.clear"
                        )
                        if operation is not None:
                            normalized.update(operation)
                    elif event_type == "conversation.item.deleted":
                        self._pop_matching_operation(
                            "conversation.item.delete",
                            item_id=normalized.get("item_id"),
                        )
                    elif event_type == "conversation.item.truncated":
                        self._pop_matching_operation(
                            "conversation.item.truncate",
                            item_id=normalized.get("item_id"),
                        )
                    elif event_type == "response.created":
                        self._pop_matching_operation(
                            "response.create",
                            generation_id=normalized.get("generation_id"),
                        )
                    elif event_type == "response.done":
                        matched_cancel = self._pop_matching_operation(
                            "response.cancel",
                            response_id=normalized.get("response_id"),
                        )
                        # An unscoped cancel has no response ID or client
                        # event ID on response.done, so even a cancelled
                        # terminal could belong to an older request. Never
                        # claim correlation from status alone. The only safe
                        # unscoped completion is the server error carrying the
                        # original cancel event_id (cancel-not-active).
                        if matched_cancel is not None:
                            normalized["cancel_acknowledged"] = True
                            normalized["cancel_acknowledged_scope"] = "scoped"
                    if normalized.get("type") == "session.updated":
                        if self._session_ready is not None and not self._session_ready.done():
                            self._session_ready.set_result(None)
                    elif normalized.get("type") == "error":
                        if self._session_ready is not None and not self._session_ready.done():
                            self._session_ready.set_exception(
                                RuntimeError("Realtime session configuration failed")
                            )
                    await self._events_queue.put(normalized)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.error("llm.event.pump_error", error_type=type(e).__name__)
            self._put_terminal_event_nowait(
                {
                    "type": "error",
                    "error_code": "event_stream_failed",
                    "error_type": type(e).__name__,
                },
                event_type="error",
            )
        finally:
            if self._session_ready is not None and not self._session_ready.done():
                self._session_ready.set_exception(
                    ConnectionError(
                        "Realtime event stream ended before session.updated"
                    )
                )
            # Stream closure is fail-closed: when the consumer has saturated
            # the queue, sacrifice the oldest payload instead of leaving an
            # unterminated iterator or deadlocking adapter shutdown.
            self._put_terminal_event_nowait(
                _EVENTS_CLOSED,
                event_type="stream.closed",
            )

    @staticmethod
    def _normalize(event: Any) -> dict | None:
        t = getattr(event, "type", None)
        if t == "session.created":
            session = getattr(event, "session", None)
            return {
                "type": "session.created",
                "session_id": getattr(session, "id", None),
            }
        if t == "session.updated":
            return {"type": "session.updated"}
        if t == "input_audio_buffer.speech_started":
            return {
                "type": "input.speech_started",
                "item_id": getattr(event, "item_id", None),
            }
        if t == "input_audio_buffer.speech_stopped":
            return {
                "type": "input.speech_stopped",
                "item_id": getattr(event, "item_id", None),
            }
        if t == "input_audio_buffer.committed":
            return {
                "type": "input.committed",
                "item_id": getattr(event, "item_id", None),
            }
        if t == "input_audio_buffer.cleared":
            return {"type": "input.cleared"}
        if t == "conversation.item.input_audio_transcription.completed":
            return {
                "type": "input.transcript",
                "text": event.transcript,
                "item_id": getattr(event, "item_id", None),
                "transcription_usage": OpenAIRealtimeLLM._normalize_transcription_usage(
                    getattr(event, "usage", None)
                ),
            }
        if t == "conversation.item.input_audio_transcription.failed":
            return {
                "type": "input.transcript.failed",
                "item_id": getattr(event, "item_id", None),
            }
        if t in {
            "conversation.item.created",
            "conversation.item.added",
            "conversation.item.done",
        }:
            item = getattr(event, "item", None)
            item_id = getattr(item, "id", None) if item else None
            return {
                "type": "conversation.item.created",
                "source_type": t,
                "item_id": item_id,
                "item_type": getattr(item, "type", None) if item else None,
                "role": getattr(item, "role", None) if item else None,
            }
        if t == "conversation.item.deleted":
            return {
                "type": "conversation.item.deleted",
                "item_id": getattr(event, "item_id", None),
            }
        if t == "conversation.item.truncated":
            return {
                "type": "conversation.item.truncated",
                "item_id": getattr(event, "item_id", None),
            }
        if t in {"response.output_item.added", "response.output_item.created"}:
            item = getattr(event, "item", None)
            return {
                "type": "response.output_item",
                "response_id": getattr(event, "response_id", None),
                "item_id": getattr(item, "id", None) if item else None,
                "item_type": getattr(item, "type", None) if item else None,
            }
        if t == "response.created":
            response = getattr(event, "response", None)
            metadata = getattr(response, "metadata", None)
            generation_id = None
            if isinstance(metadata, dict):
                raw_generation = metadata.get("zemory_generation")
                try:
                    generation_id = int(raw_generation)
                except (TypeError, ValueError):
                    pass
            return {
                "type": "response.created",
                "response_id": getattr(response, "id", None),
                "generation_id": generation_id,
            }
        if t in {"response.output_text.delta", "response.text.delta"}:
            normalized = {"type": "text.delta", "delta": event.delta}
            OpenAIRealtimeLLM._copy_event_ids(event, normalized)
            return normalized
        if t in {"response.output_text.done", "response.text.done"}:
            normalized = {"type": "text.done"}
            OpenAIRealtimeLLM._copy_event_ids(event, normalized)
            return normalized
        if t in {"response.output_audio.delta", "response.audio.delta"}:
            normalized = {
                "type": "audio.delta",
                "audio": base64.b64decode(event.delta),
            }
            OpenAIRealtimeLLM._copy_event_ids(event, normalized)
            return normalized
        if t in {"response.output_audio.done", "response.audio.done"}:
            normalized = {"type": "audio.done"}
            OpenAIRealtimeLLM._copy_event_ids(event, normalized)
            return normalized
        if t in {
            "response.output_audio_transcript.delta",
            "response.audio_transcript.delta",
        }:
            normalized = {"type": "audio.transcript.delta", "delta": event.delta}
            OpenAIRealtimeLLM._copy_event_ids(event, normalized)
            return normalized
        if t in {
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
        }:
            normalized = {"type": "audio.transcript.done"}
            OpenAIRealtimeLLM._copy_event_ids(event, normalized)
            return normalized
        if t == "response.done":
            response = getattr(event, "response", None)
            usage = getattr(response, "usage", None)
            status = getattr(response, "status", None)
            metadata = getattr(response, "metadata", None)
            generation_id = None
            if isinstance(metadata, dict):
                raw_generation = metadata.get("zemory_generation")
                try:
                    generation_id = int(raw_generation)
                except (TypeError, ValueError):
                    pass
            return {
                "type": "response.done",
                "response_id": getattr(response, "id", None),
                "generation_id": generation_id,
                "status": status,
                "usage": OpenAIRealtimeLLM._normalize_usage(usage),
            }
        if t == "error":
            error = getattr(event, "error", "unknown")
            client_event_id = (
                error.get("event_id")
                if isinstance(error, dict)
                else getattr(error, "event_id", None)
            )
            error_code = (
                error.get("code")
                if isinstance(error, dict)
                else getattr(error, "code", None)
            )
            return {
                "type": "error",
                # Provider messages can contain transcript/request material.
                # Preserve only fields needed for control-event correlation.
                "error_code": error_code,
                "error_type": type(error).__name__,
                "client_event_id": client_event_id,
                "server_event_id": getattr(event, "event_id", None),
            }
        return None

    @staticmethod
    def _copy_event_ids(event: Any, normalized: dict) -> None:
        """Preserve identifiers needed to reject stale response events."""
        for name in ("response_id", "item_id", "content_index"):
            value = getattr(event, name, None)
            if value is not None:
                normalized[name] = value

    @staticmethod
    def _normalize_usage(usage: Any) -> dict | None:
        if usage is None:
            return None

        def read(value: Any, name: str) -> Any:
            return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

        input_details = read(usage, "input_token_details")
        output_details = read(usage, "output_token_details")
        cached_details = read(input_details, "cached_tokens_details")
        return {
            "total_tokens": read(usage, "total_tokens"),
            "input_tokens": read(usage, "input_tokens"),
            "output_tokens": read(usage, "output_tokens"),
            "input_text_tokens": read(input_details, "text_tokens"),
            "input_audio_tokens": read(input_details, "audio_tokens"),
            "input_image_tokens": read(input_details, "image_tokens"),
            "cached_tokens": read(input_details, "cached_tokens"),
            "cached_text_tokens": read(cached_details, "text_tokens"),
            "cached_audio_tokens": read(cached_details, "audio_tokens"),
            "cached_image_tokens": read(cached_details, "image_tokens"),
            "output_text_tokens": read(output_details, "text_tokens"),
            "output_audio_tokens": read(output_details, "audio_tokens"),
        }

    @staticmethod
    def _normalize_transcription_usage(usage: Any) -> dict | None:
        if usage is None:
            return None

        def read(value: Any, name: str) -> Any:
            return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

        usage_type = read(usage, "type")
        if usage_type == "duration":
            return {
                "type": "duration",
                "seconds": read(usage, "seconds"),
            }
        input_details = read(usage, "input_token_details")
        return {
            "type": usage_type or "tokens",
            "total_tokens": read(usage, "total_tokens"),
            "input_tokens": read(usage, "input_tokens"),
            "output_tokens": read(usage, "output_tokens"),
            "input_text_tokens": read(input_details, "text_tokens"),
            "input_audio_tokens": read(input_details, "audio_tokens"),
        }
