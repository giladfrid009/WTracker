import cv2 as cv
import numpy as np
from view_controller import VideoReader
from tqdm import tqdm
from typing import Callable


def calc_background(
    video: VideoReader,
    num_samples: int,
    transform: Callable[[np.ndarray], np.ndarray] = None,
) -> np.ndarray:
    video.restart()
    length = video.video_length()
    width, height = video.frame_size()

    # Randomly select frames
    frame_ids = np.random.choice(length, size=num_samples, replace=False)
    frame_ids = sorted(frame_ids)

    # Store selected frames in an array
    frames = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

    # Extract frames
    for i, id in tqdm(enumerate(frame_ids), desc="reading frames", total=num_samples):
        video.seek(id)
        frame = video.get_frame()

        # Apply transform if needed
        if transform is not None:
            frame = transform(frame).astype(np.uint8)

        frames[i] = frame

    # Calculate the median along the time axis
    median = np.median(frames, axis=0).astype(np.uint8)
    return median


def find_largest_box(
    image: np.ndarray,
    background: np.ndarray,
    diff_thresh: int,
    transform: Callable[[np.ndarray], np.ndarray] = None,
) -> np.ndarray:
    if transform is not None:
        image = transform(image).astype(np.uint8)

    # Calculate difference between background and image
    diff = np.abs(image.astype(np.int16) - background.astype(np.int16))
    diff = diff.astype(np.uint8)

    # Turn differences mask to black & white according to a threshold value
    _, mask = cv.threshold(diff, diff_thresh, 255, cv.THRESH_BINARY)

    # do some morphological magic to clean up noise from the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # dilate to increase all object sizes in the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=5)

    # find contours in the binary mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # find largest contour
    largest_contour = max(contours, key=lambda c: cv.contourArea(c))

    # Get matching bbox
    largest_bbox = cv.boundingRect(largest_contour)
    largest_bbox = np.asanyarray(largest_bbox, dtype=int)

    return largest_bbox


def extract_boxes(
    video: VideoReader,
    background: np.ndarray,
    diff_thresh=10,
    transform: Callable[[np.ndarray], np.ndarray] = None,
):
    video.restart()
    length = video.video_length()
    bboxes = np.zeros(shape=(length, 4), dtype=int)

    i = 0
    while video.next_frame():
        frame = video.get_frame()
        bbox = find_largest_box(frame, background, diff_thresh, transform)
        bboxes[i] = bbox
        i += 1

    return bboxes


def find_slice_indices(bboxes: np.ndarray, start_idx: int, slice_width: int, slice_height: int) -> tuple[int, int]:
    left = bboxes[start_idx:, 0]
    right = bboxes[start_idx:, 0] + bboxes[start_idx:, 2]
    bottom = bboxes[start_idx:, 1]
    top = bboxes[start_idx:, 1] + bboxes[start_idx:, 3]

    # The function calculates cumulative min up to the current index
    # We want to find the first index where the difference between the cumulative mins
    # and maxes is larger than the given thresholds
    left_min = np.minimum.accumulate(left)
    right_max = np.maximum.accumulate(right)
    bottom_min = np.minimum.accumulate(bottom)
    top_max = np.maximum.accumulate(top)

    # We want that both the width and height conditions to hold
    is_illegal = (right_max - left_min > slice_width) & (top_max - bottom_min > slice_height)

    print(is_illegal)

    # Returns the last index where `False` can be inserted while maintaining order of the array
    # Note that `is_illegal` array is sorted, since it's first always `False`, and then always `True`
    idx = np.searchsorted(is_illegal, v=False, side="right")

    # Return the exclusive last index
    end_idx = start_idx + idx + 1
    return start_idx, end_idx
