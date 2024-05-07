from utils.path_utils import Files
import cv2 as cv

from typing import Callable
from dataclasses import dataclass
import matplotlib

matplotlib.use("QTAgg")

from frame_reader import FrameReader
from evaluation.view_controller import ViewController
from evaluation.simulator import TimingConfig
import pandas as pd
import numpy as np
import cv2 as cv
from math import ceil, floor


@dataclass
class HotKey:
    key: str
    func: Callable[[str], None]

    def __post_init__(self):
        self.key = self.key.lower()


class StreamViewer:
    def __init__(self, window_name: str = "streamer") -> None:
        self.window_name = window_name
        self.window = None
        self.hotkeys: list[HotKey] = []

        self.register_hotkey(HotKey("q", self.close))
        self.set_title(self.window_name)

    def register_hotkey(self, hotkey: HotKey):
        self.hotkeys.append(hotkey)

    def create_trakbar(self, name: str, val: int, maxval: int, onChange=lambda x: x):
        cv.createTrackbar(name, self.window_name, val, maxval, onChange)

    def update_trakbar(self, name: str, val: int):
        cv.setTrackbarPos(name, self.window_name, val)

    def set_title(self, title: str):
        cv.setWindowTitle(self.window_name, title)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def update(self, image: np.ndarray, wait: int = 1):
        cv.imshow(self.window_name, image)
        self.waitKey(wait)

    def waitKey(self, timeout: int = 0):
        key = cv.waitKey(timeout)
        if key <= 0:
            return key
        key = chr(key).lower()
        for hotkey in self.hotkeys:
            if key in hotkey.key:
                hotkey.func(key)
        return key

    def open(self):
        self.close()
        self.window = cv.namedWindow(self.window_name, flags=cv.WINDOW_GUI_EXPANDED)
        # cv.setWindowProperty(self.window_name, cv.WINDOW_GUI_EXPANDED, 1)

    def close(self, key: str = "q"):
        if self.window is not None:
            cv.destroyWindow(self.window_name)
            self.window = None


class VLC:
    def __init__(self, files: Files, config: TimingConfig, log_path: str, cam_type: str = None) -> None:
        self.streamer = StreamViewer()
        self.index = 0
        self.exit = False
        self.delay = 0
        self.play = False
        self.show_pred = False

        self.config: TimingConfig = config
        self.reader = self._create_reader(files)
        self.log = self._load_log(log_path)
        self.cam_type = cam_type

        self.initialize()

    def initialize(self):
        self._init_hotkeys()
        self._create_window()
        self.streamer.update_trakbar("delay", round(self.config.ms_per_frame))

    def _load_log(self, log_path: str) -> pd.DataFrame:
        if log_path is None:
            return None
        log = pd.read_csv(log_path, index_col="frame")
        assert len(log.index) == len(self.reader)
        return log

    def _init_hotkeys(self) -> None:
        self.streamer.register_hotkey(HotKey("q", self.close))
        self.streamer.register_hotkey(HotKey("d", self.next))
        self.streamer.register_hotkey(HotKey("a", self.prev))
        self.streamer.register_hotkey(HotKey("p", self.toggle_play))
        self.streamer.register_hotkey(HotKey("h", self.toggle_pred))

    def _create_window(self):
        self.streamer.create_trakbar("delay", 0, 250, self.set_delay)
        self.streamer.create_trakbar("#frame", 0, len(self.reader), self.seek)

    def _create_reader(self, files: Files) -> FrameReader:
        filenames = [f for f in files]
        reader = FrameReader(files.root, filenames)
        return reader

    def _get_title(self):
        frame_num = self.index
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
        log_row = self.log.iloc[self.index]
        return log_row[col_name]

    def get_photo(self) -> np.ndarray:
        photo = self.reader[self.index]
        if self.show_pred:
            self.add_pred(photo)
        return photo

    def seek(self, pos: int):
        self.index = (pos) % len(self.reader)
        self.streamer.update(self.get_photo())
        self.streamer.set_title(self._get_title())

    def next(self, key=None):
        self.index = (self.index + 1) % len(self.reader)
        self.streamer.update_trakbar("#frame", self.index)

    def prev(self, key=None):
        self.index = (self.index - 1) % len(self.reader)
        self.streamer.update_trakbar("#frame", self.index)

    def close(self, key=None):
        self.exit = True

    def set_delay(self, delay: int):
        self.delay = delay

    def toggle_play(self, key: str = None):
        self.play = not self.play

    def toggle_pred(self, key: str = None):
        self.show_pred = not self.show_pred

    def mainloop(self):
        with self.streamer as streamer:
            while not self.exit:
                delay = 0 if not self.play else self.delay
                if self.play:
                    self.next()
                streamer.waitKey(delay)

    def get_bbox(self, prefix: str) -> tuple[float, float, float, float]:
        x = self.get_attribute(prefix + "_x")
        y = self.get_attribute(prefix + "_y")
        w = self.get_attribute(prefix + "_w")
        h = self.get_attribute(prefix + "_h")
        return (x, y, w, h)

    # TODO: FIX
    def add_pred(self, photo: np.ndarray) -> np.ndarray:
        x, y, w, h = self.get_bbox(self.cam_type)
        pred_x, pred_y, pred_w, pred_h = self.get_bbox("wrm")

        pred_x = floor(pred_x - x)
        pred_y = floor(pred_y - y)
        pred_w = ceil(pred_w)
        pred_h = ceil(pred_h)

        if photo.shape[0] <= pred_x or photo.shape[1] <= pred_y:
            print(f"Warning::pred = ({pred_x}, {pred_y}, {pred_w}, {pred_h}) # cam = ({x}, {y}, {w}, {h})")
            return

        pred_x = max(pred_x, 0)
        pred_y = max(pred_y, 0)
        pred_w = min(pred_w, w)
        pred_h = min(pred_h, h)
        cv.rectangle(photo, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 0, 255), 1)
