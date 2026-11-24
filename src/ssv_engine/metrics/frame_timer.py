"""Frame timing and rolling FPS metrics."""

from __future__ import annotations

from collections import deque
from time import perf_counter


class FrameTimer:
    def __init__(self, sample_size: int = 120, history_size: int = 300):
        self._last = perf_counter()
        self._samples: deque[float] = deque(maxlen=sample_size)
        # Raw per-frame times in milliseconds, used by the on-screen graph.
        self.history_ms: deque[float] = deque(maxlen=history_size)
        self.delta_time = 0.0
        self.fps = 0.0
        self.frame_time_ms = 0.0
        self.frame_count = 0

    def tick(self) -> None:
        now = perf_counter()
        self.delta_time = now - self._last
        self._last = now
        self.frame_count += 1
        self._samples.append(self.delta_time)
        self.history_ms.append(self.delta_time * 1000.0)
        average = sum(self._samples) / len(self._samples)
        self.fps = 1.0 / average if average > 0 else 0.0
        self.frame_time_ms = average * 1000.0
