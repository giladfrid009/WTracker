import cv2 as cv
import numpy as np
import pickle

from utils.path_utils import join_paths, create_directory, create_parent_directory
from frame_reader import FrameReader
from utils.threading_utils import TaskScheduler


class FrameSaver(TaskScheduler):
    """
    A class for saving images from a frame reader to a specified folder.
    This class utilizes a queue to save images in a separate thread, which allows for non-blocking image saving.

    Methods:
        save_image: Adds an image to the queue for saving.
        close: Waits for all images to be saved and closes the image saver.
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        root_path: str = "",
        maxsize: int = 0,
        tqdm: bool = True,
        **tqdm_kwargs,
    ):
        """
        Args:
            frame_reader (FrameReader): The frame reader object from which images will be saved.
            root_path (str): The root folder path, relative to which all other paths are. Defaults to "".
            maxsize (int, optional): The maximum size of the queue. Defaults to 0.
            tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
            **tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        super().__init__(self._save_frame, maxsize, tqdm, **tqdm_kwargs)
        self._frame_reader = frame_reader
        self._root_path = root_path
        create_directory(root_path)

    def schedule(self, img_index: int, crop_dims: tuple[int, int, int, int], img_name: str):
        """
        Adds an image to the queue for saving.

        Args:
            img_index (int): The index of the image in the frame reader.
            crop_dims (tuple[int, int, int, int]): The crop dimensions (x, y, w, h) for the image.
            img_name (str): The name (path) of the image file relative to the root path.
        """
        super().schedule(img_index, crop_dims, img_name)

    def _save_frame(self, params: tuple[int, tuple[int, int, int, int], str]):
        img_index, crop_dims, img_name = params
        x, y, w, h = crop_dims
        save_path = join_paths(self._root_path, img_name)

        img = self._frame_reader[img_index]
        img = img[y : y + h, x : x + w]
        success = cv.imwrite(save_path, img)

        if not success:
            create_parent_directory(save_path)
            if not cv.imwrite(save_path, img):
                raise ValueError(f"Failed to save image {save_path}")


class ImageSaver(TaskScheduler):
    def __init__(
        self,
        root_path: str = "",
        maxsize: int = 0,
        tqdm: bool = True,
        **tqdm_kwargs,
    ):
        """
        Args:
            root_path (str): The root folder path, relative to which all other paths are. Defaults to "".
            maxsize (int, optional): The maximum size of the queue. Defaults to 0.
            tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
            **tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        super().__init__(self._save_image, maxsize, tqdm, **tqdm_kwargs)
        self._root_path = root_path
        create_directory(root_path)

    def schedule(self, img: np.ndarray, img_path: str):
        """
        Adds an image to the queue for saving.

        Args:
            img (np.ndarray): The image to save.
            img_name (str): The name (path) of the image file relative to the root path.
        """
        super().schedule(img, img_path)

    def _save_image(self, params: tuple[np.ndarray, str]):
        img, img_name = params
        save_path = join_paths(self._root_path, img_name)

        success = cv.imwrite(save_path, img)

        if not success:
            create_parent_directory(save_path)
            if not cv.imwrite(save_path, img):
                raise ValueError(f"Failed to save image {save_path}")


def pickle_load_object(file_path: str):
    """
    Load an object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error loading the object from the pickle file.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"file does not exist: {file_path}")
    except Exception as e:
        raise ValueError(f"error loading object from pickle file: {e}")


def pickle_save_object(obj, file_path: str):
    """
    Save an object to a pickle file.

    Args:
        obj: The object to be saved.
        file_path (str): The path to the pickle file.

    Raises:
        ValueError: If there is an error saving the object to the pickle file.
    """
    try:
        create_parent_directory(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(f"error saving object to pickle file: {e}")
