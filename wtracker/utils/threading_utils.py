import queue
import threading
import multiprocessing
from typing import Callable
from tqdm.auto import tqdm


def adjust_num_workers(num_tasks: int, chunk_size: int, num_workers: int = None) -> int:
    """
    Adjust the number of workers based on the number of tasks and chunk size.

    Args:
        num_tasks (int): The number of tasks to be processed.
        chunk_size (int): The size of each processing chunk.
        num_workers (int, optional): The number of workers to use for parallel processing. Defaults to None.
            If None, the number of workers is determined automatically.
    """
    if num_workers is None:  # if None then choose automatically
        num_workers = min(multiprocessing.cpu_count() / 2, num_tasks / (2 * chunk_size))
        num_workers = round(num_workers)

    num_workers = min(num_workers, num_tasks // chunk_size)  # no point having workers without tasks
    num_workers = min(num_workers, multiprocessing.cpu_count())  # no point having more workers than cpus

    if num_workers < 0:  # make sure value is valid
        num_workers = 0

    return num_workers


class TqdmQueue(queue.Queue):
    """
    A subclass of `queue.Queue` that provides progress tracking using `tqdm`.

    Args:
        maxsize (int): The maximum size of the queue (default: 0).
        **kwargs: Additional keyword arguments to be passed to the tqdm progress bar.

    Attributes:
        pbar (tqdm.tqdm): The progress bar object.
        total (int): The total number of items processed.

    Example:
        queue = ProgressQueue(maxsize=10)
        queue.put(item)
        queue.task_done()
        queue.join()
    """

    def __init__(self, maxsize: int = 0, **kwargs):
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
    """
    This class is used to schedule tasks to be executed by a worker thread.

    Args:
        task_func (Callable): The function to be executed by the worker thread.
        maxsize (int, optional): The maximum number of items that can be in the queue. Defaults to 0.
        tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
        **tqdm_kwargs: Additional keyword arguments to be passed to the TqdmQueue constructor.
    """

    def __init__(
        self,
        task_func: Callable,
        maxsize: int = 0,
        tqdm: bool = True,
        **tqdm_kwargs,
    ):

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
