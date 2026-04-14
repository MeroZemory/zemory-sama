"""In-process histogram metrics with p50/p95/p99 on demand.

Keeps the last ``_WINDOW`` samples per key. Export via
:meth:`Metrics.snapshot` (used by the optional Prometheus endpoint, not
shipped in this iteration).
"""

from __future__ import annotations

import statistics
import threading
from collections import deque

_WINDOW = 500


class Histogram:
    __slots__ = ("_samples", "_lock")

    def __init__(self) -> None:
        self._samples: deque[float] = deque(maxlen=_WINDOW)
        self._lock = threading.Lock()

    def observe(self, value_ms: float) -> None:
        with self._lock:
            self._samples.append(value_ms)

    def percentile(self, pct: float) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            data = sorted(self._samples)
        k = max(0, min(len(data) - 1, int(round(pct / 100.0 * (len(data) - 1)))))
        return data[k]

    def p50(self) -> float | None:
        return self.percentile(50)

    def p95(self) -> float | None:
        return self.percentile(95)

    def p99(self) -> float | None:
        return self.percentile(99)

    def count(self) -> int:
        with self._lock:
            return len(self._samples)

    def mean(self) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return statistics.fmean(self._samples)


class Metrics:
    def __init__(self) -> None:
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def _h(self, key: str) -> Histogram:
        with self._lock:
            h = self._histograms.get(key)
            if h is None:
                h = self._histograms[key] = Histogram()
            return h

    def observe(self, key: str, value_ms: float) -> None:
        self._h(key).observe(value_ms)

    def get(self, key: str) -> Histogram:
        return self._h(key)

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            keys = list(self._histograms.keys())
        return {
            k: {
                "count": self._histograms[k].count(),
                "p50": self._histograms[k].p50(),
                "p95": self._histograms[k].p95(),
                "p99": self._histograms[k].p99(),
                "mean": self._histograms[k].mean(),
            }
            for k in keys
        }


metrics = Metrics()
