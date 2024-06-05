import queue
import threading
from typing import Callable
from tqdm.auto import tqdm


class TqdmQueue(queue.Queue):
    """
    A subclass of `queue.Queue` that provides progress tracking using `tqdm`.

    Attributes:
        pbar (tqdm.tqdm): The progress bar object.
        total (int): The total number of items processed.

    Methods:
        task_done(): Indicate that a previously enqueued task is complete.
        _put(item): Put an item into the queue.
        join(): Block until all items in the queue have been processed.

    Example:
        queue = ProgressQueue(maxsize=10)
        queue.put(item)
        queue.task_done()
        queue.join()
    """

    def __init__(self, maxsize: int = 0, **kwargs):
        """
        Initialize the TqdmQueue object.

        Args:
            maxsize (int): The maximum size of the queue (default: 0).
            **kwargs: Additional keyword arguments to be passed to the tqdm progress bar.
        """
        super().__init__(maxsize=maxsize)
        self.pbar = tqdm(total=1, **kwargs)
        self.total = 0  # Keep our own total tracker so we can update the Progressbar

    def task_done(self):
        """
        Mark the task as done and update the progress bar.
        This method should be called when a task is completed. It updates the progress bar to reflect the completion
        of the task.
        """
        super().task_done()
        self.pbar.update()
        self.pbar.refresh()  # Redraw the progressbar

    def _put(self, item):
        super()._put(item)
        self.total += 1
        processed = self.pbar.n  # Get current progress to re-apply
        self.pbar.reset(self.total)  # Reset and update total
        self.pbar.update(processed)  # Re-apply progress
        self.pbar.refresh()  # Redraw the progressbar

    def join(self):
        """
        Blocks until all items in the Queue have been gotten and processed.
        """
        super().join()
        self.pbar.close()


class TaskScheduler:
    def __init__(
        self,
        task_func: Callable,
        maxsize: int = 0,
        tqdm: bool = True,
        **tqdm_kwargs,
    ):
        """
        Initializes a TaskScheduler object. This class is used to schedule tasks to be executed by a worker thread.

        Args:
            task_func (Callable): The function to be executed by the worker thread.
            maxsize (int, optional): The maximum number of items that can be in the queue. Defaults to 0.
            tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
            **tqdm_kwargs: Additional keyword arguments to be passed to the TqdmQueue constructor.
        """
        self._queue = TqdmQueue(maxsize, **tqdm_kwargs) if tqdm else queue.Queue(maxsize)
        self._worker_thread = threading.Thread(target=self._worker, args=(self._queue,))
        self._task_func = task_func

    def start(self):
        """
        Starts the worker thread.
        """
        self._worker_thread.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def schedule_save(self, *params):
        """
        Schedules a task by putting task parameters into the queue.

        Args:
            *params: The parameters to be passed to the task function.
        """
        self._queue.put(item=params, block=True)

    def _worker(self, q: queue.Queue):
        while True:
            params = q.get(block=True)

            # exit if signaled
            if params is None:
                break

            self._task_func(params)
            q.task_done()

    def close(self):
        """
        Waits for the queue to empty and then closes the worker thread.
        """
        self._queue.join()
        self._queue.put(None)
        self._worker_thread.join()
