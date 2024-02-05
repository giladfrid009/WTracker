import cv2 as cv
from cv2 import VideoCapture
import numpy as np


class VideoReader:
    def __init__(
        self,
        video_path: str,
        padding_size: tuple[int, int] = (0, 0),
        padding_value: tuple[int, int, int] = [255, 255, 255],
        init_position: tuple[int, int] = (0, 0),
        color_conversion: int = None,
    ):
        video_stream = VideoCapture(video_path)
        
        assert video_stream.isOpened()

        self._path = video_path
        self._stream: VideoCapture = video_stream
        self._padding_size: tuple[int, int] = padding_size
        self._padding_value: tuple[int, int, int] = padding_value
        frame_width = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._frame_size: tuple[int, int] = (frame_width, frame_height)
        self._color_conversion: int = color_conversion

        self.set_position(*init_position)

        self._finished: bool = False
        self._frame: np.ndarray = None
        self.restart()

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._finished:
            raise StopIteration()

        frame = self.get_frame()
        self.next_frame()
        return frame

    def __len__(self) -> int:
        return int(self._stream.get(cv.CAP_PROP_FRAME_COUNT))
    
    def path(self) -> str:
        return self._path

    def next_frame(self, n: int = 1) -> bool:
        assert n >= 0
        assert self._finished is False

        if n == 0:
            return True

        for _ in range(n):
            res, frame = self._stream.read()
            if res is False:
                self._finished = True
                return False

        frame = cv.copyMakeBorder(
            src=frame,
            left=self._padding_size[0],
            right=self._padding_size[0],
            top=self._padding_size[1],
            bottom=self._padding_size[1],
            borderType=cv.BORDER_CONSTANT,
            value=self._padding_value,
        )

        if self._color_conversion is not None:
            frame = cv.cvtColor(frame, self._color_conversion)

        self._frame = frame
        return True

    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def position(self) -> tuple[int, int]:
        return self._position

    def set_position(self, x: int, y: int):
        assert x >= 0 and x < self._frame_size[0]
        assert y >= 0 and y < self._frame_size[1]
        self._position = (x, y)

    def move_position(self, dx: int, dy: int):
        self.set_position(self._position[0] + dx, self._position[1] + dy)

    def get_frame(self) -> np.ndarray:
        return self._frame.copy()

    def get_slice_coords(self, w: int, h: int) -> tuple[int, int, int, int]:
        # Calc upper left coord, take padding into account
        x = self._position[0] + self._padding_size[0] - w // 2
        y = self._position[1] + self._padding_size[1] - h // 2
        return x, y, w, h

    def get_slice(self, w: int, h: int) -> np.ndarray:
        x, y, w, h = self.get_slice_coords(w, h)
        slice = self._frame[y : y + w, x : x + h]
        return slice

    def restart(self) -> bool:
        self._stream.set(cv.CAP_PROP_POS_FRAMES, 0)
        self._finished = False

        res = self.next_frame()
        if res is False:
            raise Exception("couldn't read first frame")
        return True

    def seek(self, frame_number: int) -> bool:
        assert frame_number >= 0

        if frame_number == 0:
            return self.restart()

        self._finished = False
        self._stream.set(cv.CAP_PROP_POS_FRAMES, frame_number - 1)
        return self.next_frame()


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

    def visualize_world(self, line_width: int = 4):
        # Get positions and views of micro and camera
        x_mid, y_mid, _, _ = self.get_slice_coords(0, 0)
        x_cam, y_cam, w_cam, h_cam = self.get_slice_coords(*self.camera_size())
        x_mic, y_mic, w_mic, h_mic = self.get_slice_coords(*self.micro_size())

        # Draw bboxes of micro and camera
        world = self.get_frame()
        cv.rectangle(world, (x_cam, y_cam), (x_cam + w_cam, y_cam + h_cam), (0, 0, 255), line_width)
        cv.rectangle(world, (x_mic, y_mic), (x_mic + w_mic, y_mic + h_mic), (0, 255, 0), line_width)
        cv.circle(world, (x_mid, y_mid), 1, (255, 0, 0), line_width)
        cv.imshow("world view", world)
