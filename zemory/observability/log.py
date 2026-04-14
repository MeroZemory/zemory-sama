"""structlog configuration.

Emits human-readable console output by default. Set ``ZEMORY_LOG_JSON=1``
for line-delimited JSON suitable for log shipping / bench scripts.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    as_json = os.environ.get("ZEMORY_LOG_JSON") == "1"
    logging.basicConfig(
        stream=sys.stderr,
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if as_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
