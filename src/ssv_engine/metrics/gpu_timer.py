"""Non-blocking GPU timing with GL_TIME_ELAPSED queries.

Results are read one or more frames later (only when GL reports them as
available), so the measurement never stalls the CPU waiting for the GPU.
If the driver does not support timer queries, the timer disables itself.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from OpenGL import GL


class GpuTimer:
    def __init__(self, ring_size: int = 6, average_window: int = 60) -> None:
        self.available = False
        self.last_ms = 0.0
        self._samples: deque[float] = deque(maxlen=average_window)
        self._free: list[int] = []
        self._pending: deque[int] = deque()
        self._active: int | None = None
        try:
            ids = GL.glGenQueries(ring_size)
            self._free = [int(v) for v in np.atleast_1d(ids)]
            self._ready = np.zeros(1, dtype=np.int32)
            self._result = np.zeros(1, dtype=np.uint32)
            self.available = len(self._free) > 0
        except Exception as error:  # pragma: no cover - depends on the driver
            print(f"GPU timer queries unavailable: {error}")

    @property
    def average_ms(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    def reset(self) -> None:
        self._samples.clear()
        self.last_ms = 0.0

    def begin(self) -> None:
        if not self.available or self._active is not None:
            return
        self._collect()
        if not self.available or not self._free:
            return
        query = self._free.pop()
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
        self._active = query

    def end(self) -> None:
        if self._active is None:
            return
        GL.glEndQuery(GL.GL_TIME_ELAPSED)
        self._pending.append(self._active)
        self._active = None

    def _collect(self) -> None:
        try:
            while self._pending:
                query = self._pending[0]
                GL.glGetQueryObjectiv(query, GL.GL_QUERY_RESULT_AVAILABLE, self._ready)
                if not int(self._ready[0]):
                    break
                # 32-bit result (ns) is enough for a pass shorter than ~4 s and
                # avoids PyOpenGL's broken 64-bit converter on some drivers.
                GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT, self._result)
                self._pending.popleft()
                self._free.append(query)
                self.last_ms = float(self._result[0]) / 1.0e6
                self._samples.append(self.last_ms)
        except Exception as error:  # pragma: no cover - depends on the driver
            print(f"GPU timer disabled: {error}")
            self.available = False

    def release(self) -> None:
        ids = self._free + list(self._pending) + ([self._active] if self._active is not None else [])
        if ids:
            try:
                GL.glDeleteQueries(len(ids), np.asarray(ids, dtype=np.uint32))
            except Exception:
                pass
        self._free, self._pending, self._active = [], deque(), None
