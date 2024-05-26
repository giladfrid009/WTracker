import cv2 as cv
import numpy as np

from utils.frame_reader import FrameReader, FrameStream


class ViewController(FrameStream):
    """
    A class representing a view controller for a frame stream.

    Attributes:
        frame_reader (FrameReader): The frame reader object.
        camera_size (tuple[int, int]): The size of the camera view (width, height).
        micro_size (tuple[int, int]): The size of the micro view (width, height).
        position (tuple[int, int]): The current position of the center of the view (x, y).

    Methods:
        read(): Read a frame from the frame reader and apply padding.
        set_position(x: int, y: int): Set the position of the view controller.
        move_position(dx: int, dy: int): Move the position of the view controller by dx and dy.
        visualize_world(line_width: int = 4): Visualize the world view with bounding boxes.
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        camera_size: tuple[int, int] = (251, 251),
        micro_size: tuple[int, int] = (45, 45),
        init_position: tuple[int, int] = (0, 0),
    ):
        """
        Initializes the View Controller object.

        Args:
            frame_reader (FrameReader): The frame reader object.
            camera_size (tuple[int, int], optional): The size of the camera frame. Defaults to (251, 251).
            micro_size (tuple[int, int], optional): The size of the micro frame. Defaults to (45, 45).
            init_position (tuple[int, int], optional): The initial position of the view. Defaults to (0, 0).
        """
        super().__init__(frame_reader)

        assert camera_size[0] >= micro_size[0]
        assert camera_size[1] >= micro_size[1]

        self._padding_size: tuple[int, int] = (camera_size[0] // 2, camera_size[1] // 2)

        self._camera_size = camera_size
        self._micro_size = micro_size
        self._position = init_position
        self.set_position(*init_position)

    def read(self) -> np.ndarray:
        """
        Read a frame from the frame reader and apply padding.

        Returns:
            np.ndarray: The padded frame.
        """
        frame = super().read()
        frame = cv.copyMakeBorder(
            src=frame,
            left=self._padding_size[0],
            right=self._padding_size[0],
            top=self._padding_size[1],
            bottom=self._padding_size[1],
            borderType=cv.BORDER_REPLICATE,
        )
        return frame

    @property
    def position(self) -> tuple[int, int]:
        """
        Get the current position of the view controller.

        Returns:
            tuple[int, int]: The current position (x, y).
        """
        return self._position

    @property
    def camera_size(self) -> tuple[int, int]:
        """
        Get the size of the camera view.

        Returns:
            tuple[int, int]: The size of the camera view (w, h).
        """
        return self._camera_size

    @property
    def micro_size(self) -> tuple[int, int]:
        """
        Get the size of the micro view.

        Returns:
            tuple[int, int]: The size of the micro view (w, h).
        """
        return self._micro_size

    @property
    def camera_position(self) -> tuple[int, int, int, int]:
        """
        Get the position of the camera view.

        Returns:
            tuple[int, int, int, int]: The position of the camera view (x, y, w, h).
        """
        w, h = self.camera_size
        x = self._position[0] - w // 2
        y = self._position[1] - h // 2
        return x, y, w, h

    @property
    def micro_position(self) -> tuple[int, int, int, int]:
        """
        Get the position of the micro view.

        Returns:
            tuple[int, int, int, int]: The position of the micro view (x, y, w, h).
        """
        w, h = self.micro_size
        x = self._position[0] - w // 2
        y = self._position[1] - h // 2
        return x, y, w, h

    def set_position(self, x: int, y: int):
        """
        Set the position of the view controller.
        Note, that the position is clamped to the frame size.

        Args:
            x (int): The x-coordinate of the position.
            y (int): The y-coordinate of the position.
        """

        x = np.clip(x, 0, self._frame_reader.frame_shape[1] - 1)
        y = np.clip(y, 0, self._frame_reader.frame_shape[0] - 1)
        self._position = (x, y)

    def move_position(self, dx: int, dy: int):
        """
        Move the position of the view controller by dx and dy.

        Args:
            dx (int): The amount to move in the x-direction.
            dy (int): The amount to move in the y-direction.
        """
        self.set_position(self._position[0] + dx, self._position[1] + dy)

    def _calc_view_bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        """
        Calculate the bbox of the view, while taking padding into account.

        Args:
            w (int): The width of the view.
            h (int): The height of the view.

        Returns:
            tuple[int, int, int, int]: The bounding box of the view (x, y, w, h).
        """
        x = self._position[0] + self._padding_size[0] - w // 2
        y = self._position[1] + self._padding_size[1] - h // 2
        return x, y, w, h

    def _custom_view(self, w: int, h: int) -> np.ndarray:
        """
        Get a custom view of the frame.

        Args:
            w (int): The width of the view.
            h (int): The height of the view.

        Returns:
            np.ndarray: The custom view of the frame.
        """
        x, y, w, h = self._calc_view_bbox(w, h)
        frame = self.read()
        slice = frame[y : y + w, x : x + h]
        return slice

    def camera_view(self) -> np.ndarray:
        """
        Get the camera view.

        Returns:
            np.ndarray: The camera view.
        """
        return self._custom_view(*self.camera_size)

    def micro_view(self) -> np.ndarray:
        """
        Get the micro view.

        Returns:
            np.ndarray: The micro view.
        """
        return self._custom_view(*self.micro_size)

    def visualize_world(self, line_width: int = 4, timeout: int = 1):
        """
        Visualize the world view with bounding boxes.
        Both the camera and micro views are visualized, along with the center point.

        Args:
            line_width (int): The width of the bounding box lines.
        """
        x_mid, y_mid, _, _ = self._calc_view_bbox(0, 0)
        x_cam, y_cam, w_cam, h_cam = self._calc_view_bbox(*self.camera_size)
        x_mic, y_mic, w_mic, h_mic = self._calc_view_bbox(*self.micro_size)

        world = self.read()
        if len(self._frame_reader.frame_shape) == 2 or self._frame_reader.frame_shape[2] == 1:
            world = cv.cvtColor(world, cv.COLOR_GRAY2BGR)

        cv.rectangle(world, (x_cam, y_cam), (x_cam + w_cam, y_cam + h_cam), (0, 0, 255), line_width)
        cv.rectangle(world, (x_mic, y_mic), (x_mic + w_mic, y_mic + h_mic), (0, 255, 0), line_width)
        cv.circle(world, (x_mid, y_mid), 1, (255, 0, 0), line_width)

        cv.imshow("World View", world)
        cv.waitKey(timeout)
