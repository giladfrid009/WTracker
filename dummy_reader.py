import numpy as np

from frame_reader import FrameReader


class DummyReader(FrameReader):
    def __init__(self, num_frames: int, resolution: tuple[int, int]):
        self._resolution = resolution
        frames = [str(i) for i in range(num_frames)]
        super().__init__(".", frame_files=frames)

    def _extract_frame_shape(self) -> tuple[int, ...]:
        return self._resolution

    def __getitem__(self, idx: int) -> np.ndarray:
        frame = np.zeros(self._resolution, dtype=np.uint8)
        return frame
