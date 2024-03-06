import numpy as np

from typing import Iterable
import multiprocessing
from dataset.bbox_utils import BoxFormat, BoxUtils
from data.box_calculator import BoxCalculator
from data.image_saver import ImageSaver


class SampleExtractor:
    def __init__(self, bbox_calculator: BoxCalculator):
        assert bbox_calculator.bbox_format == BoxFormat.XYWH
        self._bbox_calculator = bbox_calculator
        self._frame_reader = bbox_calculator.frame_reader

    def move_bboxes_into_bounds(self, bboxes: np.ndarray, frame_size: tuple[int, int]):
        max_w, max_h = frame_size
        x, y, w, h = BoxUtils.unpack(bboxes)

        print((x < 0).shape)
        print(((x + w) > max_w).shape)

        x[x < 0] = 0

        mask = (x + w) > max_w
        x[mask] = max_w - w[mask]

        y[y < 0] = 0

        mask = (y + h) > max_h
        y[mask] = max_h - h[mask]

        if np.any(x < 0) or np.any(y < 0):
            raise ValueError()

        return BoxUtils.pack(x, y, w, h)

    def create_specified_samples(
        self,
        frame_indices: Iterable[int],
        target_size: tuple[int, int],
        save_folder: str,
        name_format: str = "img_{:09d}.png",
        num_workers: int = multiprocessing.cpu_count() // 2,
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
        bboxes = self.move_bboxes_into_bounds(bboxes, self._frame_reader.frame_size)

        saver = ImageSaver(self._frame_reader, save_folder, desc="Saving samples", unit="fr")

        for i, bbox in enumerate(bboxes):
            saver.save_image(i, bbox, name_format.format(i))

        saver.close()

    def create_samples(
        self,
        count: int,
        target_size: tuple[int, int],
        save_folder: str,
        name_format: str = "img_{:09d}.png",
        num_workers: int = multiprocessing.cpu_count() // 2,
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
        num_workers: int = multiprocessing.cpu_count() // 2,
        chunk_size: int = 50,
    ):
        frame_indices = range(0, len(self._frame_reader))
        self.create_specified_samples(frame_indices, target_size, save_folder, name_format, num_workers, chunk_size)
