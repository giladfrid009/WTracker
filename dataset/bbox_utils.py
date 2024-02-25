import numpy as np
from enum import Enum


from enum import Enum

class BoxFormat(Enum):
    """
    Enumeration representing different box formats.

    Attributes:
        XYWH (int): Represents the box format as (x, y, width, height).
        XYXY (int): Represents the box format as (x1, y1, x2, y2).
        YOLO (int): Represents the box format as (center_x, center_y, width, height).
    """
    XYWH = 0
    XYXY = 1
    YOLO = 2


class BoxUtils:
    """
    A utility class for working with bounding boxes.
    """

    @staticmethod
    def is_bbox(array: np.ndarray) -> bool:
        """
        Check if the given array is a valid bounding box.

        Args:
            array (np.ndarray): The array to check.

        Returns:
            bool: True if the array is a valid bounding box, False otherwise.
        """
        return array is not None and array.shape[-1] == 4 and array.dtype == int

    @staticmethod
    def unpack(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unpack the given bounding box into its individual components.

        Args:
            bbox (np.ndarray): The bounding box to unpack.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The unpacked components of the bounding box.
        """
        return np.split(bbox.astype(int, copy=False), bbox.shape[-1], axis=-1)

    @staticmethod
    def pack(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, c4: np.ndarray) -> np.ndarray:
        """
        Pack the given components into a single bounding box.

        Args:
            c1 (np.ndarray): The first component of the bounding box.
            c2 (np.ndarray): The second component of the bounding box.
            c3 (np.ndarray): The third component of the bounding box.
            c4 (np.ndarray): The fourth component of the bounding box.

        Returns:
            np.ndarray: The packed bounding box.
        """
        return np.concatenate((c1, c2, c3, c4), axis=-1).astype(int, copy=False)

    @staticmethod
    def sanitize(bboxes: np.ndarray) -> np.ndarray:
        """
        Remove invalid bounding boxes from the given array.

        Args:
            bboxes (np.ndarray): The array of bounding boxes.

        Returns:
            np.ndarray: The array of sanitized bounding boxes.
        """
        mask = np.all(bboxes >= 0, axis=-1)
        return bboxes[mask]


class BoxConverter:
    """
    Utility class for converting bounding box coordinates between different formats.
    """

    @staticmethod
    def change_format(bbox: np.ndarray, src_format: BoxFormat, dst_format: BoxFormat) -> np.ndarray:
        """
        Converts the bounding box coordinates from one format to another.

        Args:
            bbox (np.ndarray): The bounding box coordinates to be converted.
            src_format (BoxFormat): The source format of the bounding box coordinates.
            dst_format (BoxFormat): The destination format of the bounding box coordinates.

        Returns:
            np.ndarray: The converted bounding box coordinates.

        Raises:
            Exception: If the conversion between the specified formats is not supported.
        """
        if dst_format == BoxFormat.XYXY:
            return BoxConverter.to_xyxy(bbox, src_format)
        elif dst_format == BoxFormat.XYWH:
            return BoxConverter.to_xywh(bbox, src_format)
        elif dst_format == BoxFormat.YOLO:
            return BoxConverter.to_xywh(bbox, src_format)
        else:
            raise Exception("unsupported bbox format conversion.")

    @staticmethod
    def to_xyxy(bbox: np.ndarray, src_format: BoxFormat):
        """
        Converts the bounding box coordinates to the XYXY format.

        Args:
            bbox (np.ndarray): The bounding box coordinates to be converted.
            src_format (BoxFormat): The source format of the bounding box coordinates.

        Returns:
            np.ndarray: The bounding box coordinates in the XYXY format.

        Raises:
            Exception: If the conversion from the specified source format is not supported.
        """
        if src_format == BoxFormat.XYXY:
            return bbox
        elif src_format == BoxFormat.XYWH:
            x1, y1, w, h = BoxUtils.unpack(bbox)
            x2 = x1 + w
            y2 = y1 + h
            return BoxUtils.pack(x1, y1, x2, y2)
        elif src_format == BoxFormat.YOLO:
            xm, ym, w, h = BoxUtils.unpack(bbox)
            x1 = xm - w // 2
            y1 = ym - h // 2
            x2 = x1 + w
            y2 = y1 + h
            return BoxUtils.pack(x1, y1, x2, y2)
        else:
            raise Exception("unsupported bbox format conversion.")

    @staticmethod
    def to_xywh(bbox: np.ndarray, src_format: BoxFormat):
        """
        Converts the bounding box coordinates to the XYWH format.

        Args:
            bbox (np.ndarray): The bounding box coordinates to be converted.
            src_format (BoxFormat): The source format of the bounding box coordinates.

        Returns:
            np.ndarray: The bounding box coordinates in the XYWH format.

        Raises:
            Exception: If the conversion from the specified source format is not supported.
        """
        if src_format == BoxFormat.XYWH:
            return bbox
        elif src_format == BoxFormat.XYXY:
            x1, y1, x2, y2 = BoxUtils.unpack(bbox)
            w = x2 - x1
            h = y2 - y1
            return BoxUtils.pack(x1, y1, w, h)
        elif src_format == BoxFormat.YOLO:
            xm, ym, w, h = BoxUtils.unpack(bbox)
            x1 = xm - w // 2
            y1 = ym - h // 2
            return BoxUtils.pack(x1, y1, w, h)
        else:
            raise Exception("unsupported bbox format conversion.")

    @staticmethod
    def to_yolo(bbox: np.ndarray, src_format: BoxFormat):
        """
        Converts the bounding box coordinates to the YOLO format.

        Args:
            bbox (np.ndarray): The bounding box coordinates to be converted.
            src_format (BoxFormat): The source format of the bounding box coordinates.

        Returns:
            np.ndarray: The bounding box coordinates in the YOLO format.

        Raises:
            Exception: If the conversion from the specified source format is not supported.
        """
        if src_format == BoxFormat.YOLO:
            return bbox
        elif src_format == BoxFormat.XYXY:
            x1, y1, x2, y2 = BoxUtils.unpack(bbox)
            w = x2 - x1
            h = y2 - y1
            xm = x1 + w // 2
            ym = y1 + h // 2
            return BoxUtils.pack(xm, ym, w, h)
        elif src_format == BoxFormat.XYWH:
            x1, y1, w, h = BoxUtils.unpack(bbox)
            xm = x1 + w // 2
            ym = y1 + h // 2
            return BoxUtils.pack(xm, ym, w, h)
        else:
            raise Exception("unsupported bbox format conversion.")
