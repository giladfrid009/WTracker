from __future__ import annotations
import os
import glob
import numpy as np
import cv2 as cv
from data.file_utils import join_paths


class FrameReader:
    """
    A class for reading frames from a directory of frame files.
    """

    def __init__(
        self,
        root_folder: str,
        frame_files: list[str],
        read_format: int = cv.IMREAD_GRAYSCALE,
    ):
        """
        Initialize the FrameReader object.

        Args:
            root_folder (str): The root folder path where the frame files are located.
            frame_files (list[str]): A list of frame file names.
            read_format (int, optional): The format in which the frames should be read. Defaults to cv.IMREAD_GRAYSCALE.
        """
        assert os.path.exists(root_folder)
        assert len(frame_files) > 0

        self._root_folder = root_folder
        self._files = frame_files
        self._read_format = read_format
        self._frame_shape = None
        self._frame_shape = self.__getitem__(0).shape

    @staticmethod
    def create_from_template(root_folder: str, name_format: str) -> FrameReader:
        """
        Creates a FrameReader object from a file name template.

        Args:
            root_folder (str): The root folder where the frame files are located.
            name_format (str): The format of the frame file names.

        Returns:
            FrameReader: The created FrameReader object.
        """
        # get all files matching name format
        fmt = name_format.format("[0-9]*")
        frame_paths = glob.glob(fmt, root_dir=root_folder)
        frame_paths = [f for f in frame_paths if os.path.isfile(join_paths(root_folder, f))]
        frame_paths = sorted(frame_paths)
        return FrameReader(root_folder, frame_paths)

    @staticmethod
    def create_from_directory(root_folder: str) -> FrameReader:
        """
        Creates a FrameReader object from a directory.

        Args:
            root_folder (str): The root folder containing the frame files.

        Returns:
            FrameReader: The created FrameReader object.

        """
        # get all files in root
        frame_paths = glob.glob("*.*", root_dir=root_folder)
        frame_paths = [f for f in frame_paths if os.path.isfile(join_paths(root_folder, f))]
        frame_paths = sorted(frame_paths)
        return FrameReader(root_folder, frame_paths)

    @property
    def root_folder(self) -> str:
        """
        Returns the root folder path.

        Returns:
            str: The root folder path.
        """
        return self._root_folder

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the frame.

        Returns:
            tuple[int, ...]: The shape of the frame, in format (h, w, ...).
        """
        return self._frame_shape

    @property
    def files(self) -> list[str]:
        """
        Returns the list of files associated with the FrameReader object.

        Returns:
            list[str]: The list of file paths.
        """
        return self._files

    @property
    def read_format(self) -> int:
        """
        Returns the read format of the frame reader.

        Returns:
            int: The read format.
        """
        return self._read_format

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self._files):
            raise IndexError("index out of bounds")

        path = join_paths(self.root_folder, self.files[idx])
        frame = cv.imread(path, self._read_format)

        if self.frame_shape and frame.shape != self.frame_shape:
            raise Exception("shape mismatch")

        return frame.astype(np.uint8, copy=False)

    def __iter__(self):
        return FrameStream(self)

    def make_stream(self):
        """
        Creates and returns a FrameStream object using the current instance of FrameReader.

        Returns:
            FrameStream: A FrameStream object.
        """
        return FrameStream(self)


class FrameStream:
    """
    A class for streaming frames from a FrameReader object.
    This class serves as an iterator for the FrameReader object.
    """

    def __init__(self, frame_reader: FrameReader):
        """
        Initializes a new instance of the FrameReader class.

        Args:
            frame_reader (FrameReader): The frame reader object.
        """
        self._frame_reader = frame_reader
        self._idx = 0

    def __len__(self):
        return len(self._frame_reader)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._idx >= len(self._frame_reader):
            raise StopIteration()

        frame = self.read()
        self.progress(1)
        return frame

    def read(self) -> np.ndarray:
        """
        Read and return the frame at the current index.

        Raises:
            IndexError: If the index is out of bounds.

        Returns:
            np.ndarray: The frame at the current index.
        """
        if self._idx < 0 or self._idx >= len(self):
            raise IndexError("index out of bounds")

        return self._frame_reader[self._idx]

    def seek(self, idx: int) -> bool:
        """
        Move the index to the specified position.

        Args:
            idx (int): The index to seek to.

        Returns:
            bool: True if the index is within the valid range, False otherwise.
        """
        self._idx = idx
        return 0 <= self._idx < len(self._frame_reader)

    def progress(self, n: int = 1) -> bool:
        """
        Moves the current index forward by the specified number of steps.

        Args:
            n (int): The number of steps to move forward. Defaults to 1.

        Returns:
            bool: True if the index was successfully moved forward, False otherwise.
        """
        return self.seek(self._idx + n)

    def reset(self):
        """
        Resets the frame reader to the beginning of the steam.
        """
        self.seek(0)
