import time
from collections import defaultdict
from contextlib import contextmanager

class Timer:
    def __init__(self, sep="/"):
        self.total = defaultdict(float)
        self.count = defaultdict(int)
        self.stack = []
        self.sep = sep

    def __call__(self, name):
        if self.stack:
            name = self.sep.join(self.stack + [name])
        return _TimerBlock(self, name)

    @contextmanager
    def scope(self, name):
        self.stack.append(name)
        try:
            yield self
        finally:
            self.stack.pop()

    def add(self, name, dt):
        self.total[name] += dt
        self.count[name] += 1

    def summary(self):
        return self._format(self.total)

    def summary_level(self, level=0):
        grouped = defaultdict(float)
        grouped_count = defaultdict(int)

        for name, t in self.total.items():
            parts = name.split(self.sep)
            key = self.sep.join(parts[:level+1]) if len(parts) > level else name
            grouped[key] += t
            grouped_count[key] += self.count[name]

        return self._format(grouped, grouped_count)

    def summary_by_leaf(self):
        grouped = defaultdict(float)
        grouped_count = defaultdict(int)

        for name, t in self.total.items():
            leaf = name.split(self.sep)[-1]
            grouped[leaf] += t
            grouped_count[leaf] += self.count[name]

        return self._format(grouped, grouped_count)

    def _format(self, total, count=None):
        if count is None:
            count = self.count

        lines = []
        total_all = sum(total.values())

        for name, t in sorted(total.items(), key=lambda kv: -kv[1]):
            n = count[name]
            avg = t / n if n else 0.0
            pct = 100.0 * t / total_all if total_all else 0.0
            lines.append(
                f"{name:40s} {t:10.4f}s  {pct:6.2f}%  "
                f"{n:6d} calls  avg={avg:.4e}s"
            )

        return "\n".join(lines)

class _TimerBlock:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer.add(self.name, time.perf_counter() - self.t0)