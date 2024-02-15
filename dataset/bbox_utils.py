import numpy as np
from enum import Enum


class BoxFormat(Enum):
    XYWH = 0
    XYXY = 1
    YOLO = 2


class BoxUtils:
    """
    Helper methods for bbox manipulation
    """

    @staticmethod
    def unpack(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.split(bbox.astype(int), bbox.shape[-1], axis=-1)

    @staticmethod
    def pack(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, c4: np.ndarray) -> np.ndarray:
        return np.concatenate((c1, c2, c3, c4), axis=-1).astype(int, copy=False)

    @staticmethod
    def change_format(bbox: np.ndarray, src_format: BoxFormat, dst_format: BoxFormat) -> np.ndarray:
        if src_format == dst_format:
            return bbox

        if (src_format, dst_format) == (BoxFormat.XYXY, BoxFormat.XYWH):
            x1, y1, x2, y2 = BoxUtils.unpack(bbox)
            w = x2 - x1
            h = y2 - y1
            return BoxUtils.pack(x1, y1, w, h)

        elif (src_format, dst_format) == (BoxFormat.XYXY, BoxFormat.YOLO):
            x1, y1, x2, y2 = BoxUtils.unpack(bbox)
            w = x2 - x1
            h = y2 - y1
            xm = x1 + w // 2
            ym = y1 + h // 2
            return BoxUtils.pack(xm, ym, w, h)

        elif (src_format, dst_format) == (BoxFormat.XYWH, BoxFormat.XYXY):
            x1, y1, w, h = BoxUtils.unpack(bbox)
            x2 = x1 + w
            y2 = y2 + h
            return BoxUtils.pack(x1, y1, x2, y2)

        elif (src_format, dst_format) == (BoxFormat.XYWH, BoxFormat.YOLO):
            x1, y1, w, h = BoxUtils.unpack(bbox)
            xm = x1 + w // 2
            ym = y1 + h // 2
            return BoxUtils.pack(xm, ym, w, h)

        elif (src_format, dst_format) == (BoxFormat.YOLO, BoxFormat.XYXY):
            xm, ym, w, h = BoxUtils.unpack(bbox)
            x1 = xm - w // 2
            y1 = ym - h // 2
            x2 = x1 + w
            y2 = y1 + h
            return BoxUtils.pack(x1, y1, x2, y2)

        elif (src_format, dst_format) == (BoxFormat.YOLO, BoxFormat.XYWH):
            xm, ym, w, h = BoxUtils.unpack(bbox)
            x1 = xm - w // 2
            y1 = ym - h // 2
            return BoxUtils.pack(x1, y1, w, h)

        else:
            raise Exception("unsupported bbox format conversion.")
