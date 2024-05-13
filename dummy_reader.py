import numpy as np

from frame_reader import FrameReader


class DummyReader(FrameReader):
    def __init__(self, num_frames: int, resolution: tuple[int, int], color:bool=True):
        self.color = color
        self._resolution = resolution
        frames = [str(i) for i in range(num_frames)]
        super().__init__(".", frame_files=frames)
        
        self._frame = np.zeros(self.frame_shape, dtype=np.uint8)
        self._frame[:,:] = 255

    def _extract_frame_shape(self) -> tuple[int, ...]:
        return self._resolution

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._frame.copy()
    
    def _extract_frame_shape(self) -> tuple[int, ...]:
        if self.color:
            return (*self._resolution, 3)
        return self._resolution
