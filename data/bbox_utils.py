import numpy as np
from enum import Enum


class BoxFormat(Enum):
    XY_WH = 0
    XY_XY = 1
    XYmid_WH = 2


class BoxUtils:
    """
    Helper methods for bbox manipulation
    """

    class Format(Enum):
        XY_WH = 0
        XY_XY = 1
        XYmid_WH = 2

    @staticmethod
    def unpack(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.split(bbox.astype(int), bbox.shape[-1], axis=-1)

    @staticmethod
    def pack(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, c4: np.ndarray) -> np.ndarray:
        return np.concatenate((c1, c2, c3, c4), axis=-1).astype(int, copy=False)

    @staticmethod
    def change_format(bbox: np.ndarray, src_format: Format, dst_format: Format) -> np.ndarray:
        if src_format == dst_format:
            return bbox

        if (src_format, dst_format) == (BoxFormat.XY_XY, BoxFormat.XY_WH):
            x1, y1, x2, y2 = BoxUtils.unpack(bbox)
            w = x2 - x1
            h = y2 - y1
            return BoxUtils.pack(x1, y1, w, h)

        elif (src_format, dst_format) == (BoxFormat.XY_XY, BoxFormat.XYmid_WH):
            x1, y1, x2, y2 = BoxUtils.unpack(bbox)
            w = x2 - x1
            h = y2 - y1
            xm = x1 + w // 2
            ym = y1 + h // 2
            return BoxUtils.pack(xm, ym, w, h)

        elif (src_format, dst_format) == (BoxFormat.XY_WH, BoxFormat.XY_XY):
            x1, y1, w, h = BoxUtils.unpack(bbox)
            x2 = x1 + w
            y2 = y2 + h
            return BoxUtils.pack(x1, y1, x2, y2)

        elif (src_format, dst_format) == (BoxFormat.XY_WH, BoxFormat.XYmid_WH):
            x1, y1, w, h = BoxUtils.unpack(bbox)
            xm = x1 + w // 2
            ym = y1 + h // 2
            return BoxUtils.pack(xm, ym, w, h)

        elif (src_format, dst_format) == (BoxFormat.XYmid_WH, BoxFormat.XY_XY):
            xm, ym, w, h = BoxUtils.unpack(bbox)
            x1 = xm - w // 2
            y1 = ym - h // 2
            x2 = x1 + w
            y2 = y1 + h
            return BoxUtils.pack(x1, y1, x2, y2)

        elif (src_format, dst_format) == (BoxFormat.XYmid_WH, BoxFormat.XY_WH):
            xm, ym, w, h = BoxUtils.unpack(bbox)
            x1 = xm - w // 2
            y1 = ym - h // 2
            return BoxUtils.pack(x1, y1, w, h)

        else:
            raise Exception("unsupported bbox format conversion.")
