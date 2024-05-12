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
        return array.shape[-1] == 4

    @staticmethod
    def unpack(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unpack the given bounding box into its individual components.

        Args:
            bbox (np.ndarray): The bounding box to unpack.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The unpacked components of the bounding box.
        """
        c1, c2, c3, c4 = np.split(bbox, bbox.shape[-1], axis=-1)
        c1 = np.squeeze(c1, axis=-1)
        c2 = np.squeeze(c2, axis=-1)
        c3 = np.squeeze(c3, axis=-1)
        c4 = np.squeeze(c4, axis=-1)
        return c1, c2, c3, c4

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
        c1 = np.expand_dims(c1, axis=-1)
        c2 = np.expand_dims(c2, axis=-1)
        c3 = np.expand_dims(c3, axis=-1)
        c4 = np.expand_dims(c4, axis=-1)
        return np.concatenate((c1, c2, c3, c4), axis=-1)

    @staticmethod
    def sanitize(bboxes: np.ndarray, box_format: BoxFormat, image_shape: tuple[int, int]) -> np.ndarray:
        """
        Sanitizes the bounding boxes by removing invalid boxes that are outside the image boundaries.

        Args:
            bboxes (np.ndarray): The input bounding boxes.
            box_format (BoxFormat): The format of the input bounding boxes.
            image_shape (tuple[int, int]): The shape of the image (w, h).

        Returns:
            np.ndarray: The sanitized bounding boxes.
        """
        max_height = image_shape[0] - 1
        max_width = image_shape[1] - 1

        bboxes = BoxConverter.change_format(bboxes, box_format, BoxFormat.XYWH)
        x, y, w, h = BoxUtils.unpack(bboxes)

        good_mask = (x >= 0) & (y >= 0) & (w > 0) & (h > 0)
        good_mask = good_mask & (x + w <= max_width) & (y + h <= max_height)
        bboxes = bboxes[good_mask]

        return BoxConverter.change_format(bboxes, BoxFormat.XYWH, box_format)
    
    @staticmethod
    def center(bboxes: np.ndarray, box_format: BoxFormat=BoxFormat.XYWH) -> np.ndarray:
        """
        Calculate the center of the bounding boxes.

        Args:
            bboxes (np.ndarray): The input bounding boxes.
            box_format (BoxFormat): The format of the input bounding boxes.

        Returns:
            np.ndarray: The center of the bounding boxes, in the format (center_x, center_y).
        """
        bboxes = BoxConverter.change_format(bboxes, box_format, BoxFormat.XYWH)
        center_x = bboxes[:,0] + bboxes[:,2] / 2
        center_y = bboxes[:,1] + bboxes[:,3] / 2
        return np.array([center_x, center_y]).T


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
