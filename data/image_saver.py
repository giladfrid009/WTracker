import cv2 as cv
import threading
import queue

from data.file_utils import join_paths, create_directory
from data.frame_reader import FrameReader
from data.tqdm_utils import TqdmQueue


class ImageSaver:
    def __init__(self, frame_reader: FrameReader, save_folder: str, maxsize: int = 0, tqdm: bool = True, **tqdm_kwargs):
        self._frame_reader = frame_reader
        self._save_folder = save_folder

        create_directory(save_folder)

        self._queue = TqdmQueue(maxsize, **tqdm_kwargs) if tqdm else queue.Queue(maxsize)

        self._worker_thread = threading.Thread(target=self.save_worker, args=(self._queue,))
        self._worker_thread.start()

    def save_image(self, img_index: int, crop_dims: tuple[int, int, int, int], img_name: str):
        self._queue.put((img_index, crop_dims, img_name))

    def save_worker(self, param_queue: queue.Queue):
        while True:
            params = param_queue.get(block=True)

            # exit if signaled
            if params is None:
                break

            img_index, crop_dims, img_name = params
            x, y, w, h = crop_dims
            save_path = join_paths(self._save_folder, img_name)

            img = self._frame_reader[img_index]
            img = img[y : y + h, x : x + w]
            cv.imwrite(save_path, img)

            param_queue.task_done()

    def close(self):
        self._queue.join()
        self._queue.put(None)
        self._worker_thread.join()
