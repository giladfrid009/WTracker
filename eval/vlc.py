import pandas as pd
import numpy as np
from math import ceil, floor
import os
import cv2 as cv
from typing import Callable
from dataclasses import dataclass, field
import matplotlib

matplotlib.use("QTAgg")

from utils.path_utils import Files, create_directory, join_paths
from utils.io_utils import ImageSaver
from utils.frame_reader import FrameReader, DummyReader
from sim.config import TimingConfig


@dataclass
class HotKey:
    """
    Represents a hotkey that can be used to trigger a specific function.

    Attributes:
        key (str): The key for the hotkey.
        func (Callable[[str], None]): The function to be called when the hotkey is triggered.
        description (str): The description of the hotkey (optional).
    """

    key: str
    func: Callable[[str], None]
    description: str = field(default="")

    def __post_init__(self):
        self.key = self.key.lower()


class StreamViewer:
    """
    A class for viewing and interacting with photos and video streams.

    Methods:
        register_hotkey(self, hotkey: HotKey): Registers a hotkey.
        create_trackbar(self, name: str, val: int, maxval: int, onChange=lambda x: x): Creates a trackbar.
        update_trackbar(self, name: str, val: int): Updates the value of a trackbar.
        set_title(self, title: str): Sets the title of the window.
        update(self, image: np.ndarray, wait: int = 1): Updates the window with a new image.
        waitKey(self, timeout: int = 0): Waits for a key press.
        open(self): Opens the window.
        close(self, key: str = "q"): Closes the window.
        imshow(self, image: np.ndarray, title: str = "image"): Displays an image in the window.

    Example:
        with StreamViewer() as streamer:
            streamer.imshow(image)
            streamer.waitKey()

    """

    def __init__(self, window_name: str = "streamer") -> None:
        """
        Initializes the StreamViewer object.

        Args:
            window_name (str, optional): The name of the window.
        """
        self.window_name = window_name
        self.window = None
        self.hotkeys: list[HotKey] = []

        self.register_hotkey(HotKey("q", self.close, "close the window"))

    def register_hotkey(self, hotkey: HotKey):
        """
        Registers a hotkey.

        Args:
            hotkey (HotKey): The hotkey to register.
        """
        self.hotkeys.append(hotkey)

    def create_trackbar(self, name: str, val: int, maxval: int, onChange=lambda x: x):
        """
        Creates a trackbar.

        Args:
            name (str): The name of the trackbar.
            val (int): The initial value of the trackbar.
            maxval (int): The maximum value of the trackbar.
            onChange (function): The function to call when the trackbar value changes.
        """
        cv.createTrackbar(name, self.window_name, val, maxval, onChange)

    def update_trackbar(self, name: str, val: int):
        """
        Updates the value of a trackbar.

        Args:
            name (str): The name of the trackbar.
            val (int): The new value of the trackbar.
        """
        cv.setTrackbarPos(name, self.window_name, val)

    def set_title(self, title: str):
        """
        Sets the title of the window.

        Args:
            title (str): The new title of the window.
        """
        cv.setWindowTitle(self.window_name, title)

    def __enter__(self):
        """
        Enters the context manager.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the context manager.
        """
        self.close()

    def __del__(self):
        """
        Destructor method.
        """
        self.close()

    def update(self, image: np.ndarray, wait: int = 1):
        """
        Updates the window with a new image.

        Args:
            image (np.ndarray): The image to display.
            wait (int): The delay in milliseconds before updating the window.
        """
        cv.imshow(self.window_name, image)
        self.waitKey(wait)

    def waitKey(self, timeout: int = 0):
        """
        Waits for a key press. This Function also triggers the hotkeys.

        Args:
            timeout (int): The timeout in milliseconds.

        Returns:
            str: The key that was pressed.
        """
        key = cv.waitKey(timeout)
        if key <= 0:
            return key
        key = chr(key).lower()
        for hotkey in self.hotkeys:
            if key in hotkey.key:
                hotkey.func(key)
        return key

    def open(self):
        """
        Opens the window.
        """
        self.close()
        self.window = cv.namedWindow(self.window_name, flags=cv.WINDOW_GUI_EXPANDED)
        # cv.setWindowProperty(self.window_name, cv.WND_PROP_TOPMOST, 1)
        self.set_title(self.window_name)

    def close(self, key: str = "q"):
        """
        Closes the window.

        Args:
            key (str): The key to close the window.
        """
        if self.window is not None:
            cv.destroyWindow(self.window_name)
            self.window = None

    def imshow(self, image: np.ndarray, title: str = "image"):
        """
        Displays an image in the window.

        Args:
            image (np.ndarray): The image to display.
            title (str): The title of the image.
        """
        self.update(image, wait=0)
        self.set_title(title)
        cv.setWindowProperty(self.windD_PROP_TOPMOST, 1)


class VLC:
    """
    The VLC class represents a video player for visualizing Simulations. This class supports saving Simulation frames (with or without boxes overlay) as well.

    Args:
        files (Files): The files to read frames from. If None, the video player will present the log data (simulation) on a white background.
        config (TimingConfig): The timing configuration of the system.
        log_path (str): The path to the log file.
        cam_type (str): The type of camera. This should match the prefix of the corresponding columns in the log file.
        show_pred (bool, optional): Whether to show the prediction box. Defaults to True.
        show_micro (bool, optional): Whether to show the microscope box. Defaults to False.
        show_cam (bool, optional): Whether to show the camera box. Defaults to False.

    Methods:
        initialize(): Initializes the video player.
        print_hotkeys(): Prints the available hotkeys along with their descriptions.
        get_attribute(col_name: str): Gets the value of an attribute from the current row in the log.
        get_photo() -> np.ndarray: Gets the current frame with overlays (i.e the image that will be displayed).
        seek(pos: int): Seeks to a specific frame.
        next(key=None): Moves to the next frame.
        prev(key=None): Moves to the previous frame.
        close(key=None): Closes the video player.
        set_delay(delay: int): Sets the delay between frames.
        mainloop(): Runs the main loop of the video player.
        save_stream(folder_path: str) -> None: Saves the frames as images in a folder and creates a video file of it. This function doesn't show the video player (thus mainloop should not be called).
        make_vid(folder_path: str, img_name_format: str, output_dir: str) -> None: Creates a video from the saved frames.
    """

    def __init__(
        self,
        files: Files | None,
        config: TimingConfig,
        log_path: str,
        cam_type: str,
        show_pred: bool = True,
        show_micro: bool = False,
        show_cam: bool = False,
    ) -> None:
        self.streamer = StreamViewer(window_name="VLC")
        self.index = 0
        self._curr_row = None
        self.exit = False
        self.delay = 0
        self.play = False
        self.show_pred = show_pred
        self.show_micro = show_micro
        self.show_cam = show_cam

        self.cam_type: str = cam_type
        self.config: TimingConfig = config
        self.log: pd.DataFrame = self._load_log(log_path)
        self.reader: FrameReader = self._create_reader(files)

    def initialize(self) -> None:
            """
            Initializes the VLC player by setting up hotkeys, opening the streamer,
            creating a window, and updating the trackbar.

            Parameters:
                None

            Returns:
                None
            """
            self._init_hotkeys()
            self._create_window()
            self.streamer.update_trackbar("delay", round(self.config.ms_per_frame))
            self.print_hotkeys()

    def _load_log(self, log_path: str) -> pd.DataFrame:
        if log_path is None:
            return None
        log = pd.read_csv(log_path, index_col="frame")
        if self.cam_type == "plt":
            log["plt_x"] = 0
            log["plt_y"] = 0
            log["plt_h"] = max(log["cam_y"]) + max(log["cam_h"])
            log["plt_w"] = max(log["cam_x"]) + max(log["cam_w"])
        # assert len(log.index) == len(self.reader)

        self._curr_row = log.iloc[self.index]

        return log

    def _init_hotkeys(self) -> None:
        self.streamer.register_hotkey(HotKey("q", self.close, "close VLC"))
        self.streamer.register_hotkey(HotKey("d", self.next, "next frame"))
        self.streamer.register_hotkey(HotKey("a", self.prev, "previous frame"))
        self.streamer.register_hotkey(HotKey("p", self.toggle_play, "play/pause"))
        self.streamer.register_hotkey(HotKey("h", self.toggle_pred, "toggle prediction box"))
        self.streamer.register_hotkey(HotKey("m", self.toggle_micro, "toggle microscope box"))
        self.streamer.register_hotkey(HotKey("c", self.toggle_cam, "toggle camera box"))

    def print_hotkeys(self):
        print("Hotkeys:")
        for hotkey in self.streamer.hotkeys:
            print(f" - {hotkey.key} : {hotkey.description}")

    def _create_window(self):
        self.streamer.open()
        self.streamer.create_trackbar("delay", 0, 250, self.set_delay)
        self.streamer.create_trackbar("#frame", 0, len(self.reader), self.seek)

    def _create_reader(self, files: Files) -> FrameReader:
        if files is None:
            frame_num = len(self.log.index)
            frame_size = (
                self.get_attribute(self.cam_type + "_h"),
                self.get_attribute(self.cam_type + "_w"),
            )
            return DummyReader(frame_num, frame_size)

        filenames = [f for f in files]
        reader = FrameReader(files.root, filenames)
        return reader

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.streamer.close()

    def _get_title(self):
        curr_phase = self.get_attribute("phase")
        phase_title = f"Action: {curr_phase}"

        cycle_len = self.config.imaging_frame_num + self.config.moving_frame_num
        cycle_progress = 1 + self.index % cycle_len
        cycle_title = (
            f"cycle progress [{cycle_progress}/{cycle_len}]: "
            + cycle_progress * "#"
            + (cycle_len - cycle_progress) * "_"
        )

        title = f"{phase_title} :: {cycle_title}"
        return title

    def get_attribute(self, col_name: str):
        return self._curr_row[col_name]

    def update_curr_row(self):
        self._curr_row = self.log.iloc[self.index]

    def get_photo(self) -> np.ndarray:
        photo = self.reader[self.index]
        if self.show_pred:
            self.add_pred(photo)
        if self.show_micro:
            self.add_micro_box(photo)
        if self.show_cam:
            self.add_cam_box(photo)

        self.draw_center(photo)
        return photo

    def seek(self, pos: int):
        self.index = (pos) % len(self.reader)
        self.update_curr_row()
        self.streamer.update(self.get_photo())
        self.streamer.set_title(self._get_title())

    def next(self, key=None):
        self.index = (self.index + 1) % len(self.reader)
        self.streamer.update_trackbar("#frame", self.index)

    def prev(self, key=None):
        self.index = (self.index - 1) % len(self.reader)
        self.streamer.update_trackbar("#frame", self.index)

    def close(self, key=None):
        self.exit = True

    def set_delay(self, delay: int):
        self.delay = delay

    def toggle_play(self, key: str = None):
        self.play = not self.play

    def toggle_pred(self, key: str = None):
        self.show_pred = not self.show_pred

    def toggle_micro(self, key: str = None):
        self.show_micro = not self.show_micro

    def toggle_cam(self, key: str = None):
        self.show_cam = not self.show_cam

    def mainloop(self):
        """
        Main loop for the VLC player. This method makes the VLC player interactive by allowing the user to control the player using hotkeys.

        This method continuously runs the VLC player until the `exit` flag is set to True (by self.close() (called by an hotkey)).
        It checks the `play` flag to determine if the player should continue playing or pause.
        The `delay` variable is used to control the delay between each iteration of the loop and is set to 0 to pause.

        Args:
            None

        Returns:
            None
        """
        with self as vlc:
            while not self.exit:
                delay = 0 if not self.play else self.delay
                if self.play:
                    self.next()
                vlc.streamer.waitKey(delay)

    def get_bbox(self, prefix: str) -> tuple[float, float, float, float]:
        x = self.get_attribute(prefix + "_x")
        y = self.get_attribute(prefix + "_y")
        w = self.get_attribute(prefix + "_w")
        h = self.get_attribute(prefix + "_h")
        return (x, y, w, h)

    def draw_box(
        self,
        photo: np.ndarray,
        bbox: tuple[float, float, float, float],
        color: tuple[int, int, int] = (0, 0, 255),
        width: int = 1,
    ) -> None:
        if any(bbox) is np.nan:
            return
        x, y, w, h = self.get_bbox(self.cam_type)
        pred_x, pred_y, pred_w, pred_h = bbox

        pred_x = floor(pred_x - x)
        pred_y = floor(pred_y - y)
        pred_w = ceil(pred_w)
        pred_h = ceil(pred_h)

        cv.rectangle(photo, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), color, width)

    def draw_marker(
        self,
        photo: np.ndarray,
        x: float,
        y: float,
        marker_size: int = 5,
        marker_type: int = cv.MARKER_CROSS,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
    ) -> None:
        frame_x, frame_y, frame_w, frame_h = self.get_bbox(self.cam_type)
        x, y = floor(x - frame_x), floor(y - frame_y)
        cv.drawMarker(photo, (x, y), color, marker_type, marker_size, thickness)

    def draw_center(self, photo: np.ndarray):
        x, y, w, h = self.get_bbox("mic")
        center = (x + w // 2, y + h // 2)
        cv.drawMarker(photo, center, (0, 0, 255), cv.MARKER_CROSS, 7, 1)

    def add_pred(self, photo: np.ndarray) -> None:
        worm_bbox = self.get_bbox("wrm")
        self.draw_box(photo, worm_bbox, (0, 0, 0), 1)

    def add_micro_box(self, photo: np.ndarray) -> None:
        mic_bbox = self.get_bbox("mic")
        self.draw_box(photo, mic_bbox, (0, 0, 255), 1)

    def add_cam_box(self, photo: np.ndarray) -> None:
        cam_bbox = self.get_bbox("cam")
        self.draw_box(photo, cam_bbox, (128, 0, 0), 2)

    def save_stream(
        self,
        folder_path: str,
    ) -> None:
        create_directory(folder_path)
        filename = f"{self.cam_type}_" + "{:07d}.png"

        with ImageSaver(folder_path, tqdm_kwargs={"total": len(self.log.index)}) as worker:

            for index in range(len(self.log.index)):
                self.index = index
                self.update_curr_row()

                path = join_paths(folder_path, filename.format(index))
                img = self.get_photo()
                worker.schedule_save(img, path)

        image_format = filename.replace("{:", "%").replace("}", "")
        self.make_vid(folder_path, image_format, folder_path)

    def make_vid(self, folder_path: str, img_name_format: str, output_dir: str) -> None:
        fps = self.config.frames_per_sec
        command = f"ffmpeg -framerate {fps} -start_number 0 -i {join_paths(folder_path, img_name_format)} -c:v copy {join_paths(output_dir, 'video.mp4')}"
        print(command)
        os.system(command)
