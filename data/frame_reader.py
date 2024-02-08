from __future__ import annotations
import os
import glob
import numpy as np
import cv2 as cv


class FrameReader:
    def __init__(
        self,
        root_folder: str,
        frame_files: list[str],
    ):
        assert os.path.exists(root_folder)
        assert len(frame_files) > 0

        self._root_folder = root_folder
        self._files: list[str] = frame_files
        self._frame_shape = None
        self._frame_shape = self.__getitem__(0).shape

    @staticmethod
    def create_from_template(root_folder: str, name_format: str) -> FrameReader:
        # get all files matching name format
        fmt = os.path.join(root_folder, name_format.format("[0-9]*"))
        frame_paths = glob.glob(fmt)
        frame_paths = [f for f in frame_paths if os.path.isfile(f)]
        frame_paths = sorted(frame_paths)
        return FrameReader(root_folder, frame_paths)

    @staticmethod
    def create_from_directory(root_folder) -> FrameReader:
        # get all files in root
        frame_paths = glob.glob(root_folder + "/*")
        frame_paths = [f for f in frame_paths if os.path.isfile(f)]
        frame_paths = sorted(frame_paths)
        return FrameReader(root_folder, frame_paths)

    @property
    def root_folder(self) -> str:
        return self._root_folder

    @property
    def frame_shape(self) -> tuple(int, int):
        return self._frame_shape

    @property
    def files(self):
        return self._files

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self._files):
            raise IndexError("index out of bounds")

        frame = cv.imread(self._files[idx], cv.IMREAD_GRAYSCALE)

        if self.frame_shape and frame.shape != self.frame_shape:
            raise Exception("shape mismatch")

        return frame.astype(np.uint8)

    def __iter__(self):
        return FrameStream(self)


class FrameStream:
    def __init__(self, frame_reader: FrameReader):
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
        if self._idx < 0 or self._idx >= len(self):
            raise IndexError("index out of bounds")

        return self._frame_reader[self._idx]

    def seek(self, idx: int) -> bool:
        self._idx = idx
        return 0 <= self._idx < len(self._frame_reader)

    def progress(self, n: int = 1) -> bool:
        return self.seek(self._idx + n)

    def reset(self):
        self.seek(0)
