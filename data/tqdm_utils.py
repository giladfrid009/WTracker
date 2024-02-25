import queue
from tqdm.auto import tqdm
from functools import partial


class ProgressQueue(queue.Queue):
    def __init__(self, maxsize: int = 0, **kwargs):
        super().__init__(maxsize=maxsize)
        self.pbar = tqdm(total=1, **kwargs)
        self.total = 0  # Keep our own total tracker so we can update the Progressbar

    def task_done(self):
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
        super().join()
        self.pbar.close()
