from typing import Callable
from dataclasses import dataclass
import cv2 as cv
import numpy as np


@dataclass
class HotKey:
    key: str
    func: Callable[[str], None]

    def _post_init_(self):
        self.key = self.key.lower()


class ImageDisplay:
    def __init__(self, window_name: str = "streamer") -> None:
        self.window_name = window_name
        self.window = None
        self.hotkeys: list[HotKey] = []

        self._open()
        self.register_hotkey(HotKey("q", lambda k: self.close()))

    def __enter__(self):
        self.close()
        self.window = cv.namedWindow(self.window_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def _open(self):
        self.close()
        self.window = cv.namedWindow(self.window_name, flags=cv.WINDOW_NORMAL)
        cv.waitKey(1)

    def wait_key(self, timeout: int = 0):
        key = cv.waitKey(timeout)
        if key <= 0:
            return key
        key = chr(key).lower()
        for hotkey in self.hotkeys:
            if key in hotkey.key:
                hotkey.func(key)
        return key

    def register_hotkey(self, hotkey: HotKey):
        self.hotkeys.append(hotkey)

    def create_trakbar(self, name: str, val: int, maxval: int, onChange=lambda x: x):
        cv.createTrackbar(name, self.window_name, val, maxval, onChange)

    def set_title(self, title: str):
        cv.setWindowTitle(self.window_name, title)

    def update_trakbar(self, name: str, val: int):
        cv.setTrackbarPos(name, self.window_name, val)

    def update(self, image: np.ndarray, timeout: int = 1):
        cv.imshow(self.window_name, image)
        self.wait_key(timeout)

    def close(self):
        if self.window is not None:
            cv.destroyWindow(self.window_name)
            self.window = None
