import numpy as np

from typing import Collection, Iterable
from dataset.bbox_utils import BoxUtils
from dataset.create.box_calculator import BoxCalculator
from utils.io_utils import FrameSaver


class SampleExtractor:
    def __init__(self, bbox_calculator: BoxCalculator):
        self._bbox_calculator = bbox_calculator
        self._frame_reader = bbox_calculator._frame_reader

    def move_bboxes_into_bounds(self, bboxes: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
        """
        frame_size in format (w, h)
        """
        max_w, max_h = frame_size
        x, y, w, h = BoxUtils.unpack(bboxes)

        x[x < 0] = 0

        mask = (x + w) > max_w
        x[mask] = max_w - w[mask]

        y[y < 0] = 0

        mask = (y + h) > max_h
        y[mask] = max_h - h[mask]

        if np.any(x < 0) or np.any(y < 0):
            raise ValueError()
        
        if np.any(x + w > frame_size[0]) or np.any(y + h > frame_size[1]):
            raise ValueError()

        return BoxUtils.pack(x, y, w, h)

    def create_specified_samples(
        self,
        frame_indices: Collection[int],
        target_size: tuple[int, int],
        save_folder: str,
        name_format: str = "img_{:09d}.png",
        num_workers: int = None,
        chunk_size: int = 50,
    ):
        bboxes = self._bbox_calculator.calc_specified_boxes(
            frame_indices=frame_indices,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

        x, y, w, h = BoxUtils.unpack(bboxes)

        x -= np.random.randint(0, target_size[0] - w + 1)
        y -= np.random.randint(0, target_size[1] - h + 1)
        w = np.full_like(x, target_size[0])
        h = np.full_like(x, target_size[1])

        bboxes = BoxUtils.pack(x, y, w, h)

        frame_size = tuple(reversed(self._frame_reader.frame_size)) # (h, w) -> (w, h)
        bboxes = self.move_bboxes_into_bounds(bboxes, frame_size)

        with FrameSaver(self._frame_reader, root_path=save_folder, desc="Saving samples", unit="fr") as saver:
            for i, bbox in enumerate(bboxes):
                saver.schedule_save(i, bbox, name_format.format(i))

    def create_samples(
        self,
        count: int,
        target_size: tuple[int, int],
        save_folder: str,
        name_format: str = "img_{:09d}.png",
        num_workers: int = None,
        chunk_size: int = 50,
    ):
        length = len(self._frame_reader)
        count = min(length, count)
        frame_indices = np.random.choice(length, size=count, replace=False)

        self.create_specified_samples(frame_indices, target_size, save_folder, name_format, num_workers, chunk_size)

    def create_all_samples(
        self,
        target_size: tuple[int, int],
        save_folder: str,
        name_format: str = "img_{:09d}.png",
        num_workers: int = None,
        chunk_size: int = 50,
    ):
        frame_indices = range(0, len(self._frame_reader))
        self.create_specified_samples(frame_indices, target_size, save_folder, name_format, num_workers, chunk_size)
