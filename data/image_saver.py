import cv2 as cv
import threading
import queue

from data.file_utils import join_paths, create_directory, create_parent_directory
from data.frame_reader import FrameReader
from data.tqdm_utils import TqdmQueue


class ImageSaver:
    """
    A class for saving images from a frame reader to a specified folder.
    This class utilizes a queue to save images in a separate thread, which allows for non-blocking image saving.

    Methods:
        save_image: Adds an image to the queue for saving.
        close: Waits for all images to be saved and closes the image saver.
    """

    def __init__(self, frame_reader: FrameReader, save_folder: str, maxsize: int = 0, tqdm: bool = True, **tqdm_kwargs):
        """
        Initializes an ImageSaver object.

        Args:
            frame_reader (FrameReader): The frame reader object from which images will be saved.
            save_folder (str): The folder path where the images will be saved.
            maxsize (int, optional): The maximum size of the queue. Defaults to 0.
            tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
            **tqdm_kwargs: Additional keyword arguments for tqdm.
        """

        self._frame_reader = frame_reader
        self._save_folder = save_folder
        create_directory(save_folder)

        self._queue = TqdmQueue(maxsize, **tqdm_kwargs) if tqdm else queue.Queue(maxsize)
        self._worker_thread = threading.Thread(target=self._save_worker, args=(self._queue,))

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

    def save_image(self, img_index: int, crop_dims: tuple[int, int, int, int], img_name: str):
        """
        Adds an image to the queue for saving.

        Args:
            img_index (int): The index of the image in the frame reader.
            crop_dims (tuple[int, int, int, int]): The crop dimensions (x, y, w, h) for the image.
            img_name (str): The name of the image file.
        """

        self._queue.put((img_index, crop_dims, img_name))

    def _save_worker(self, param_queue: queue.Queue):
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
            success = cv.imwrite(save_path, img)

            if not success:
                create_parent_directory(save_path)
                if not cv.imwrite(save_path, img):
                    raise ValueError(f"Failed to save image {save_path}")

            param_queue.task_done()

    def close(self):
        """
        Closes the image saver and waits for all images to be saved.

        """
        self._queue.join()
        self._queue.put(None)
        self._worker_thread.join()
