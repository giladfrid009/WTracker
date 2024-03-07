import queue
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
