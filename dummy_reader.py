import numpy as np

from frame_reader import FrameReader


class DummyReader(FrameReader):
    def __init__(self, num_frames: int, resolution: tuple[int, int], colored: bool = True):
        self.colored = colored
        self._resolution = resolution
        shape = (*resolution, 3) if colored else resolution
        self._frame = np.full(shape, fill_value=255, dtype=np.uint8)

        frames = [str(i) for i in range(num_frames)]
        super().__init__(".", frame_files=frames)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._frame.copy()

    def _extract_frame_shape(self) -> tuple[int, ...]:
        if self.colored:
            return (*self._resolution, 3)
        return self._resolution
