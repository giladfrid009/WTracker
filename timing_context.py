from time import perf_counter


class TimingContext:
    def __init__(self, silent=False) -> None:
        self.silent = silent
        self.start = None
        self.time = None

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        if not self.silent:
            print(f"Time: {self.time:.3f} seconds")
