from time import perf_counter
from contextlib import contextmanager


class Timer:
    """
    A context manager for measuring the execution time of a code block.

    Attributes:
        time (float): The elapsed time of the code block execution.

    Methods:
        __enter__(): Enter the context and start measuring the time.
        __exit__(type, value, traceback): Exit the context and calculate the elapsed time.
    """

    def __init__(self, silent=False) -> None:
        """
        Initializes a TimingContext object.

        Args:
            silent (bool, optional): If True, suppresses the output. Defaults to False.
        """
        self._silent = silent
        self._start = -1
        self.time = -1

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self._start
        if not self._silent:
            print(f"Time: {self.time:.3f} seconds")


class TimeBenchmark:
    """
    A class for measuring the average time taken for a block of code to execute.

    Methods:
        __init__(self, silent=False): Initializes a TimeBenchmark instance.
        measure(self, silent: bool = None): Context manager for measuring the time of a code block.
        reset(self): Resets the measurement counters.
        average(self) -> float: Calculates the average time taken per iteration.
        __str__(self) -> str: Returns a string representation of the TimeBenchmark instance.
    """

    def __init__(self, silent=False) -> None:
        """
        Initializes a TimingContext object.

        Args:
            silent (bool, optional): If True, suppresses the output of timing information. Defaults to False.
        """
        self._iters = 0
        self._time_total = 0
        self._silent = silent

    @contextmanager
    def measure(self, silent: bool = None):
        """
        Measures the execution time of a code block using a Timer context manager.
        The Timer object is yielded to the caller, and the TimeBenchmark instance is updated with the measured time.

        Args:
            silent (bool, optional): If True, suppresses the output of the Timer. If None, uses the value of self._silent.

        Yields:
            Timer: A Timer object that measures the execution time of the code block.

        """
        if silent is None:
            silent = self._silent

        with Timer(self._silent) as timer:
            yield timer

        self._iters += 1
        self._time_total += timer.time

    def reset(self):
        """
        Resets the timing context by setting the iteration count and total time to zero.
        """
        self._iters = 0
        self._time_total = 0

    def average(self) -> float:
        """
        Calculate the average time per iteration.

        Returns:
            float: The average time per iteration.
        """
        if self._iters == 0:
            return 0.0
        return self._time_total / self._iters

    def __str__(self) -> str:
        return f"Average time: {self.average()} seconds"
