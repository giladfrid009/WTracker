from time import perf_counter
from contextlib import contextmanager


class Timer:
    def __init__(self, silent=False) -> None:
        self._silent = silent
        self._start = None
        self.time = None

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self._start
        if not self._silent:
            print(f"Time: {self.time:.3f} seconds")


class TimeBenchmark:
    def __init__(self, silent=False) -> None:
        self._iters = 0
        self._time_total = 0
        self._silent = silent

    @contextmanager
    def measure(self, silent: bool = None) -> Timer:
        if silent is None:
            silent = self._silent

        with Timer(self._silent) as timer:
            yield timer

        self._iters += 1
        self._time_total += timer.time

    def reset(self):
        self._iters = 0
        self._time_total = 0

    def average(self) -> float:
        if self._iters == 0:
            return 0.0
        return self._time_total / self._iters

    def __str__(self) -> str:
        return f"Average time: {self.average()} seconds"
