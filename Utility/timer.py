import time
from collections import defaultdict

class Timer:
    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def __call__(self, name):
        return _TimerBlock(self, name)

    def add(self, name, dt):
        self.total[name] += dt
        self.count[name] += 1

    def summary(self):
        lines = []
        total_all = sum(self.total.values())

        for name, t in sorted(self.total.items(), key=lambda kv: -kv[1]):
            n = self.count[name]
            avg = t / n if n else 0.0
            pct = 100.0 * t / total_all if total_all else 0.0
            lines.append(
                f"{name:25s} {t:10.4f}s  {pct:6.2f}%  "
                f"{n:6d} calls  avg={avg:.4e}s"
            )

        return "\n".join(lines)


class _TimerBlock:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer.add(self.name, time.perf_counter() - self.t0)