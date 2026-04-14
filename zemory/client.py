"""Backwards-compatible shim.

The original ``zemory.client.run`` has moved to
:func:`zemory.orchestrator.run` after the v0.2 refactor that merged
``zemory/`` and ``zemory_vad/`` into one package with runtime-selectable
profiles. This module only re-exports so ``python -m zemory`` and any
legacy imports keep working.
"""

from __future__ import annotations

from zemory.orchestrator import run

__all__ = ["run"]
