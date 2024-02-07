from __future__ import annotations
import os
import glob
import numpy as np
import cv2 as cv


class FrameReader:
    def __init__(
        self,
        root_folder: str,
        frame_name_template: str,
    ):
        assert os.path.exists(root_folder)
        self._root_folder = root_folder
        self._frame_name_template = frame_name_template

        # get the files in the root folder which match the pattern
        # note that their order is determined by their string-comparison sorting
        file_fmt = os.path.join(root_folder, frame_name_template.format("[0-9]*"))
        self._files: list[str] = sorted(glob.glob(file_fmt))
        assert len(self._files) > 0

        self._frame_shape = self.__getitem__(0).shape

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
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of bounds")

        frame = cv.imread(self._files[idx], cv.IMREAD_GRAYSCALE)
        return frame

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

        frame = self._frame_reader[self._idx]
        self._idx += 1
        return frame

    def read(self) -> np.ndarray:
        if self._idx < 0 or self._idx >= len(self):
            raise IndexError("index out of bounds")

        return self._frame_reader[self._idx]

    def seek(self, idx: int):
        self._idx = idx

    def next(self, n: int) -> np.ndarray:
        self.seek(self._idx + n)
        return self.read()

    def reset(self):
        self.seek(0)
