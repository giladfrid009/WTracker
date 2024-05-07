import numpy as np

from frame_reader import FrameReader
from dataset.raw_dataset import ExperimentConfig


class DummyReader(FrameReader):
    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        frames = [str(i) for i in range(experiment_config.num_frames)]

        super().__init__(".", frame_files=frames)
        self._frame_shape = experiment_config.orig_resolution

    def __getitem__(self, idx: int) -> np.ndarray:
        frame = np.zeros(self.experiment_config.orig_resolution, dtype=np.uint8)
        return frame
