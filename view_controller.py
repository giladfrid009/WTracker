import cv2 as cv
import numpy as np

from data.frame_reader import FrameReader, FrameStream


class ViewController(FrameStream):
    def __init__(
        self,
        frame_reader: FrameReader,
        camera_size: tuple[int, int] = (251, 251),
        micro_size: tuple[int, int] = (45, 45),
        init_position: tuple[int, int] = (0, 0),
        padding_value: int = 255,
    ):
        super().__init__(frame_reader)

        assert camera_size[0] >= micro_size[0]
        assert camera_size[1] >= micro_size[1]

        self._padding_size: tuple[int, int] = (camera_size[0] // 2, camera_size[1] // 2)
        self._padding_value: tuple[int, int, int] = padding_value

        self._camera_size = camera_size
        self._micro_size = micro_size
        self._position = init_position
        self.set_position(*init_position)

    def read(self):
        frame = super().read()
        frame = cv.copyMakeBorder(
            src=frame,
            left=self._padding_size[0],
            right=self._padding_size[0],
            top=self._padding_size[1],
            bottom=self._padding_size[1],
            borderType=cv.BORDER_CONSTANT,
            value=self._padding_value,
        )
        return frame

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @property
    def camera_size(self) -> tuple[int, int]:
        return self._camera_size

    @property
    def micro_size(self) -> tuple[int, int]:
        return self._micro_size

    def set_position(self, x: int, y: int):
        assert x >= 0 and x < self._frame_reader.frame_shape[1]
        assert y >= 0 and y < self._frame_reader.frame_shape[0]
        self._position = (x, y)

    def move_position(self, dx: int, dy: int):
        self.set_position(self._position[0] + dx, self._position[1] + dy)

    def _calc_view_coords(self, w: int, h: int) -> tuple[int, int, int, int]:
        # calc upper left coord, take padding into account
        x = self._position[0] + self._padding_size[0] - w // 2
        y = self._position[1] + self._padding_size[1] - h // 2
        return x, y, w, h

    def _custom_view(self, w: int, h: int) -> np.ndarray:
        x, y, w, h = self._calc_view_coords(w, h)
        frame = self.read()
        slice = frame[y : y + w, x : x + h]
        return slice

    def camera_view(self) -> tuple[int, int]:
        return self._custom_view(*self.camera_size)

    def micro_view(self) -> tuple[int, int]:
        return self._custom_view(*self.micro_size)

    def visualize_world(self, line_width: int = 4):
        # Get positions and views of micro and camera
        x_mid, y_mid, _, _ = self._calc_view_coords(0, 0)
        x_cam, y_cam, w_cam, h_cam = self._calc_view_coords(*self.camera_size)
        x_mic, y_mic, w_mic, h_mic = self._calc_view_coords(*self.micro_size)

        world = self.read()
        world = cv.cvtColor(world, cv.COLOR_GRAY2BGR)

        # Draw bboxes of micro and camera
        cv.rectangle(world, (x_cam, y_cam), (x_cam + w_cam, y_cam + h_cam), (0, 0, 255), line_width)
        cv.rectangle(world, (x_mic, y_mic), (x_mic + w_mic, y_mic + h_mic), (0, 255, 0), line_width)
        cv.circle(world, (x_mid, y_mid), 1, (255, 0, 0), line_width)
        cv.imshow("world view", world)
