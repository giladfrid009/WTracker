import cv2 as cv
from cv2 import VideoCapture
import numpy as np


class VideoReader:
    def __init__(
        self,
        video_stream: VideoCapture,
        padding_size: tuple[int, int],
        padding_value: tuple[int, int, int] = [255, 255, 255],
        init_position: tuple[int, int] = (0, 0),
        color_conversion: int = None,
    ):
        self._video_stream: VideoCapture = video_stream
        self._padding_size: tuple[int, int] = padding_size
        self._padding_value: tuple[int, int, int] = padding_value
        frame_width = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._frame_size: tuple[int, int] = (frame_height, frame_width)
        self._color_conversion: int = color_conversion

        self.set_position(*init_position)

        self._finished: bool = False
        self._frame: np.ndarray = None
        self.restart()

    def next_frame(self, n: int = 1) -> bool:
        assert n >= 1
        assert self._finished is False

        for _ in range(n):
            res, frame = self._video_stream.read()
            if res is False:
                self._finished = True
                return False

        self._frame = cv.copyMakeBorder(
            src=frame,
            top=self._padding_size[0],
            bottom=self._padding_size[0],
            left=self._padding_size[1],
            right=self._padding_size[1],
            borderType=cv.BORDER_CONSTANT,
            value=self._padding_value,
        )

        if self._color_conversion is not None:
            frame = cv.cvtColor(frame, self._color_conversion)

        return True

    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def position(self) -> tuple[int, int]:
        return self._position

    def set_position(self, y: int, x: int):
        assert y >= 0 and y < self._frame_size[0]
        assert x >= 0 and x < self._frame_size[1]
        self._position = (y, x)

    def move_position(self, dy: int, dx: int):
        self.set_position(self._position[0] + dy, self._position[1] + dx)

    def world_view(self) -> np.ndarray:
        return self._frame.copy()

    def get_slice_coords(self, h: int, w: int) -> tuple[int, int, int, int]:
        y_mid, x_mid = self._position
        h_pad, w_pad = self._padding_size

        # Calc upper left coord, take padding into account
        y = y_mid - h // 2 + h_pad
        x = x_mid - w // 2 + w_pad

        return y, x, h, w

    def get_slice(self, h: int, w: int) -> np.ndarray:
        y, x, h, w = self.get_slice_coords(h, w)
        slice = self._frame[y : y + h, x : x + w]
        return slice

    def restart(self):
        self._video_stream.set(cv.CAP_PROP_POS_FRAMES, 0)
        self._finished = False

        res = self.next_frame()
        if res is False:
            raise Exception("couldn't read first frame")


class ViewController(VideoReader):
    def __init__(
        self,
        video_stream: VideoCapture,
        camera_size: tuple[int, int] = (251, 251),
        micro_size: tuple[int, int] = (45, 45),
        padding_value: tuple[int, int, int] = [255, 255, 255],
        init_position: tuple[int, int] = (0, 0),
        color_conversion: int = None,
    ):
        assert camera_size[0] >= micro_size[0]
        assert camera_size[1] >= micro_size[1]

        self._camera_size = camera_size
        self._micro_size = micro_size

        super().__init__(
            video_stream,
            (camera_size[0] // 2, camera_size[1] // 2),
            padding_value,
            init_position,
            color_conversion,
        )

    def camera_size(self) -> tuple[int, int]:
        return self._camera_size

    def micro_size(self) -> tuple[int, int]:
        return self._micro_size

    def camera_view(self) -> tuple[int, int]:
        return self.get_slice(*self.camera_size())

    def micro_view(self) -> tuple[int, int]:
        return self.get_slice(*self.micro_size())

    def visualize_world(self):
        # Get positions and views of micro and camera
        y_mid, x_mid = self.position()
        y_cam, x_cam, h_cam, w_cam = self.get_slice_coords(*self.camera_size())
        y_mic, x_mic, h_mic, w_mic = self.get_slice_coords(*self.micro_size())

        # Draw bboxes of micro and camera
        world = self.world_view()
        cv.rectangle(world, (x_cam, y_cam), (x_cam + w_cam, y_cam + h_cam), (0, 0, 255), 4)
        cv.rectangle(world, (x_mic, y_mic), (x_mic + w_mic, y_mic + h_mic), (0, 255, 0), 4)
        cv.circle(world, (x_mid, y_mid), 1, (255, 0, 0), 2)
        cv.imshow("world view", world)
