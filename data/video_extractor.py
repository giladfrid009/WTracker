import cv2 as cv
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import concurrent
import threading
import functools
import queue

from data.tqdm_utils import TqdmQueue
from data.file_utils import create_directory, join_paths, Files
from data.frame_reader import FrameReader
from dataset.bbox_utils import BoxFormat, BoxConverter


class VideoExtractor:
    """
    The VideoExtractor class provides functionality for extracting videos from a FrameReader object.

    Methods:
        background: Returns the background image of the video.
        all_bboxes: Returns all bounding boxes.
        initialize: Initializes the video extractor.
        generate_videos: Generates videos by randomly selecting starting frames and saving them.
        generate_all_videos: Generates all videos by iterating over all the frames, calculating video bounds, and saving the videos.

    Example:
        # create a frame reader
        frame_reader = FrameReader.from_folder("path/to/frames_folder")

        # create a video extractor
        extractor = VideoExtractor(frame_reader)

        # generate videos
        extractor.generate_videos(10, (256, 256), "path/to/save_folder_{:03d}")
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        bg_probes: int = 100,
        diff_thresh: int = 10,
        num_workers: int = 2,
        chunk_size: int = 50,
    ):
        """
        Initializes a VideoExtractor object.

        Args:
            frame_reader (FrameReader): An object that provides access to video frames.
            bg_probes (int, optional): The number of frames to use for background calculation. Defaults to 100.
            diff_thresh (int, optional): The threshold value for creating the mask which separates background and foreground. Defaults to 10.
            num_workers (int, optional): The number of worker processes to use for parallel processing. Defaults to 2.
            chunk_size (int, optional): The size of chunks for parallel processing. Defaults to 50.
        """
        self._frame_reader = frame_reader
        self._bg_probes = bg_probes
        self._diff_thresh = diff_thresh
        self._num_workers = num_workers
        self._chunk_size = chunk_size

        # for caching
        self._cached_all_bboxes: np.ndarray = None
        self._cached_background: np.ndarray = None

    def background(self) -> np.ndarray:
        """
        Returns the background image of the video.
        If the background image has not been calculated yet, it will be calculated and cached for future use.

        Returns:
            np.ndarray: The background image of the video.
        """
        if self._cached_background is None:
            self._cached_background = self._calc_background()
        return self._cached_background

    def all_bboxes(self) -> np.ndarray:
        """
        Returns all bounding boxes.
        If the bounding boxes have not been calculated yet, it calculates them and caches the result.

        Returns:
            np.ndarray: An array of bounding boxes.
        """
        if self._cached_all_bboxes is None:
            self._cached_all_bboxes = self._calc_all_bboxes()
        return self._cached_all_bboxes

    def initialize(self, cache_bboxes: bool = False):
        """
        Initializes the video extractor.

        Args:
            cache_bboxes (bool, optional): Whether to cache bounding boxes. Defaults to False.
        """
        self.background()
        if cache_bboxes:
            self.all_bboxes()

    def _calc_background(self) -> np.ndarray:
        """
        Calculate the background image by extracting a subset of frames and taking the median along the time axis.

        Returns:
            np.ndarray: The calculated background image as a numpy array.
        """
        length = len(self._frame_reader)
        size = min(self._bg_probes, length)

        # get frames
        frame_ids = np.random.choice(length, size=size, replace=False)
        extracted_list = []
        for frame_id in tqdm(frame_ids, desc="Extracting background", unit="fr"):
            frame = self._frame_reader[frame_id]
            extracted_list.append(frame)

        # calculate the median along the time axis
        extracted = np.stack(extracted_list, axis=0)
        median = np.median(extracted, axis=0).astype(np.uint8, copy=False)
        return median

    @staticmethod
    def _calc_bbox(frame: np.ndarray, background: np.ndarray, thresh: int) -> np.ndarray:
        """
        Calculate the bounding box of the largest contour in the frame.

        Parameters:
            frame (np.ndarray): The input frame.
            background (np.ndarray): The background frame.
            thresh (int): The threshold value for creating the mask which separates background and foreground.
                If the difference between the pixel value and the background is less than `thresh` then the pixel is considered as background.

        Returns:
            np.ndarray: The bounding box coordinates (x, y, width, height).
        """
        # get mask according to the threshold value
        diff = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        diff = diff.astype(np.uint8)
        _, mask = cv.threshold(diff, thresh, 255, cv.THRESH_BINARY)

        # apply morphological ops to the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv.dilate(mask, np.ones((1, 1), np.uint8))

        # extract contours and bbox
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=lambda c: cv.contourArea(c))
        largest_bbox = cv.boundingRect(largest_contour)
        largest_bbox = np.asanyarray(largest_bbox, dtype=int)

        return largest_bbox
    
    def create_samples(self, count:int, target_size: tuple[int, int], save_folder:str, format: str = "png"):
        rnd_fids = np.random.choice(len(self._frame_reader), size=count, replace=False)
        for i in rnd_fids:
            frame = self._frame_reader[i]
            sample = self.sample_camera_view(frame, target_size)
            
            file_name = self._frame_reader.files[i] if format is None else self._frame_reader.files[i][:-3] + format
            full_path = join_paths(save_folder, file_name)
            cv.imwrite(full_path, sample)

    
    def sample_camera_view(self, frame:np.ndarray, target_size: tuple[int, int], method:str='uniform'):
        cam_height, cam_width = target_size
        x, y, width, height = self._calc_bbox(frame, self.background(), self._diff_thresh)
        x_range = [x+width-cam_width, x]
        y_range = [y+height-cam_height, y]

        cam_x, cam_y = None, None
        if method == 'normal':
            x_mean = (x_range[0] + x_range[1])/2
            y_mean = (y_range[0] + y_range[1])/2
            std_x = (x_range[1] - x_range[1])
            std_y = (y_range[1] - y_range[1])

            cam_x = int(np.random.normal(x_mean, std_x))
            cam_y = int(np.random.normal(y_mean, std_y))
        if method == 'uniform':
            cam_x = int(np.random.uniform(x_range[0], x_range[1]))
            cam_y = int(np.random.uniform(y_range[0], y_range[1]))

        cam_x, cam_y, _, _ = self.expand_bbox((cam_x, cam_y, cam_width, cam_height), target_size)
        frame = frame[cam_y:cam_y+cam_height, cam_x:cam_x+cam_width]
        return frame

    def _calc_all_bboxes(self) -> np.ndarray:
        """
        Calculate bounding boxes for all frames in the `frame_reader` provided during the creation of this class.

        Returns:
            np.ndarray: Array of bounding boxes for each frame.
        """
        if self._num_workers > 0:
            bboxes = concurrent.process_map(
                functools.partial(VideoExtractor._calc_bbox, background=self.background(), thresh=self._diff_thresh),
                self._frame_reader,
                max_workers=self._num_workers,
                chunksize=self._chunk_size,
                desc="Extracting bboxes",
                unit="fr",
            )
            return np.stack(bboxes, axis=0)

        bboxes = []
        for frame in tqdm(self._frame_reader, desc="Extracting bboxes", unit="fr"):
            bbox = VideoExtractor._calc_bbox(frame, background=self.background(), thresh=self._diff_thresh)
            bboxes.append(bbox)
        return np.stack(bboxes, axis=0)

    def _calc_video_bounds_cached(
        self,
        start_index: int,
        target_size: tuple[int, int],
        max_length: int = None,
    ) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """
        Calculates video bounds utilizing cached bboxes for each frame, based on the given parameters.

        Args:
            start_index (int): The starting index of the video frames.
            target_size (tuple[int, int]): The target size of the video crop.
            max_length (int, optional): The maximum length of the video frames to consider. Defaults to None.

        Returns:
            tuple[tuple[int, int], tuple[int, int, int, int]]: A tuple where first and last frame index of the video,
            and another tuple of bounding box coords of video frames.
        """
        bboxes = self.all_bboxes()
        bboxes = bboxes[start_index:] if max_length is None else bboxes[start_index : start_index + max_length]

        # get bbox coordinates
        left, bottom, width, height = bboxes.T
        right = left + width
        top = bottom + height

        min_left = np.minimum.accumulate(left)
        min_bottom = np.minimum.accumulate(bottom)
        max_right = np.maximum.accumulate(right)
        max_top = np.maximum.accumulate(top)

        # Find where bboxes are out of legal bounds
        illegal_width = max_right - min_left > target_size[0]
        illegal_height = max_top - min_bottom > target_size[1]
        is_illegal = illegal_width | illegal_height

        last_legal_idx = np.searchsorted(is_illegal, v=False, side="right")
        if last_legal_idx >= len(is_illegal):
            last_legal_idx = len(is_illegal) - 1

        slice_indices = (start_index, start_index + last_legal_idx + 1)
        slice_width = max_right[last_legal_idx] - min_left[last_legal_idx]
        slice_height = max_top[last_legal_idx] - min_bottom[last_legal_idx]
        slice_bbox = (min_left[last_legal_idx], min_bottom[last_legal_idx], slice_width, slice_height)
        slice_bbox = self.expand_bbox(slice_bbox, target_size)
        return slice_indices, slice_bbox

    def _calc_video_bounds_dynamic(
        self,
        start_index: int,
        target_size: tuple[int, int],
        max_length: int = None,
        granularity: int = 2,
    ) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """
        Calculates video bounds frame by frame dynamically based on the given parameters.

        Args:
            start_index (int): The starting index of the video frames.
            target_size (tuple[int, int]): The target size of the video crop, in format (w, h).
            max_length (int, optional): The maximum length of the video frames to consider. Defaults to None.
            granularity (int, optional): The step size for iterating through the video frames. Defaults to 2.

        Returns:
            tuple[tuple[int, int], tuple[int, int, int, int]]: A tuple where first and last frame index of the video,
            and another tuple of bounding box coords of video frames.
        """
        min_x, min_y = np.nan, np.nan
        max_x, max_y = np.nan, np.nan
        end_index = start_index

        max_length = len(self._frame_reader) if max_length is None else max_length
        last_index = min(len(self._frame_reader), start_index + max_length)

        for i in range(start_index, last_index, granularity):
            frame = self._frame_reader[i]
            bbox = self._calc_bbox(frame, background=self.background(), thresh=self._diff_thresh)

            x1, y1, x2, y2 = BoxConverter.change_format(bbox, BoxFormat.XYWH, BoxFormat.XYXY)

            new_min_x, new_min_y = np.nanmin([min_x, x1]), np.nanmin([min_y, y1])
            new_max_x, new_max_y = np.nanmax([max_x, x2]), np.nanmax([max_y, y2])

            # if the new max width or height is larger than the target then stop
            if new_max_x - new_min_x >= target_size[0] or new_max_y - new_min_y >= target_size[1]:
                break

            min_x, min_y = new_min_x, new_min_y
            max_x, max_y = new_max_x, new_max_y
            end_index = i

        # slice indices - video frame range; slice_bbox - video crop coords
        slice_indices = (start_index, end_index + 1)
        slice_bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        slice_bbox = self.expand_bbox(slice_bbox, target_size)
        return slice_indices, slice_bbox

    def expand_bbox(self, bbox: tuple[int, int, int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """
        Expands the bounding box coordinates to to be of `target_size` dimensions.

        Args:
            bbox (tuple[int, int, int, int]): The original bounding box coordinates (x, y, w, h).
            target_size (tuple[int, int]): The target size of the bounding box (w, h).

        Returns:
            tuple[int, int, int, int]: The expanded bounding box coordinates (x, y, w, h).
        """

        frame_height, frame_width = self._frame_reader.frame_size

        assert target_size[0] <= frame_width and target_size[1] <= frame_height

        x, y = bbox[:2]
        w, h = target_size

        if x + w - frame_width > 0:
            x = frame_width - w

        if y + h - frame_height > 0:
            y = frame_height - h

        return x, y, w, h

    def _calc_video_bounds(
        self,
        start_index: int,
        target_size: tuple[int, int],
        max_length: int = None,
        granularity: int = 1,
    ) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """
        Calculate the video bounds based on the given parameters.

        Args:
            start_index (int): The start index of the video.
            target_size (tuple[int, int]): The target size of the video.
            max_length (int, optional): The maximum length of the video. Defaults to None.
            granularity (int, optional): The granularity of the video bounds calculation. Defaults to 1.

        Returns:
            tuple[tuple[int, int], tuple[int, int, int, int]]: A tuple where first and last frame index of the video,
            and another tuple of bounding box coords of video frames.
        """
        if self._cached_all_bboxes is not None:
            return self._calc_video_bounds_cached(start_index, target_size, max_length)
        else:
            return self._calc_video_bounds_dynamic(start_index, target_size, max_length, granularity)

    def generate_videos(
        self,
        count: int,
        frame_size: tuple[int, int],
        save_folder_format: str,
        max_length: int = None,
        name_format: str = "frame_{:06d}.png",
        granularity: int = 2,
    ):
        """
        Generates videos by randomly selecting starting frames and saving them.
        Note, that resulting videos might overlap.

        Args:
            count (int): The number of videos to generate.
            frame_size (tuple[int, int]): The desired size of each frame in the videos.
            save_folder_format (str): The format string for the save folder path of each video.
            max_length (int, optional): The maximum length of each video in frames. Defaults to None.
            name_format (str, optional): The format string for the name of each video. If None then original name is preserved.
            granularity (int, optional): The granularity of out of bounds check. Defaults to 2.

        Returns:
            None
        """

        self.initialize(cache_bboxes=False)

        # create a different thread which will save the videos
        progress_queue = TqdmQueue(desc="Saving videos", unit="vid")
        worker_thread = threading.Thread(target=self._video_saver_worker, args=(progress_queue,))
        worker_thread.start()

        # Randomly select frames
        rnd_fids = np.random.choice(len(self._frame_reader), size=count, replace=False)

        for i, fid in tqdm(enumerate(rnd_fids), desc="Calculating video samples", unit="vid", total=count):
            trim_range, crop_dims = self._calc_video_bounds(fid, frame_size, max_length, granularity)
            progress_queue.put((save_folder_format.format(i), trim_range, crop_dims, name_format))

        progress_queue.join()  # wait for queue to empty
        progress_queue.put(None)  # put stop signal into queue
        worker_thread.join()  # wait for worker thread to finish

    def generate_all_videos(
        self,
        frame_size: tuple[int, int],
        save_folder_format: str,
        max_length: int = None,
        name_format: str = "frame_{:06d}.png",
    ):
        """
        Generates all videos by iterating over all the frames, calculating video bounds, and saving the videos.
        Note, that the resulting videos do not overlap, and each video starts after the last frame of the previous video.

        Args:
            frame_size (tuple[int, int]): The size of each frame in the generated videos.
            save_folder_format (str): The format string for the save folder path of each video.
            max_length (int, optional): The maximum length of each video in frames. Defaults to None.
            name_format (str, optional): The format string for the name of each video frame. If None then original name is preserved.
        """
        self.initialize(cache_bboxes=True)

        # create a different thread which will save the videos
        progress_queue = TqdmQueue(desc="Saving videos", unit="vid")
        worker_thread = threading.Thread(target=self._video_saver_worker, args=(progress_queue,))
        worker_thread.start()

        calc_bar = tqdm(desc="Calculating video samples", total=len(self._frame_reader), unit="fr")
        start_frame = 0
        i = 0

        while start_frame < len(self._frame_reader):
            (trim_start, trim_end), crop_dims = self._calc_video_bounds(start_frame, frame_size, max_length)
            progress_queue.put((save_folder_format.format(i), (trim_start, trim_end), crop_dims, name_format))

            # update loop params
            i += 1
            start_frame = trim_end
            calc_bar.update(trim_end - trim_start)

        calc_bar.close()
        progress_queue.join()  # wait for queue to empty
        progress_queue.put(None)  # put stop signal into queue
        worker_thread.join()  # wait for worker thread to finish

    def _video_saver_worker(self, video_params: queue.Queue):
        """
        Worker method that processes video saving tasks from the video_params queue.

        Args:
            video_params (queue.Queue): The queue containing video saving tasks.

        Returns:
            None
        """
        while True:
            task = video_params.get()

            # exit if signaled
            if task is None:
                break

            save_folder, trim_range, crop_dims, name_format = task
            self._crop_and_save_video(save_folder, trim_range, crop_dims, name_format)
            video_params.task_done()

    def _crop_and_save_video(
        self,
        save_folder: str,
        trim_range: tuple[int, int],
        crop_dims: tuple[int, int, int, int],
        name_format: str = None,
    ):
        """
        Crop and save video frames within the specified trim range and crop dimensions.

        Args:
            save_folder (str): The path to the folder where the cropped frames will be saved.
            trim_range (tuple[int, int]): The range of frames to be cropped and saved, specified as a tuple of start and end indices.
            crop_dims (tuple[int, int, int, int]): The dimensions of the crop area, specified as a tuple of x, y, width, and height.
            name_format (str, optional): The format string for the name of each video frame. If None then original name is preserved.

        Returns:
            None
        """
        # create dir if doesn't exist
        create_directory(save_folder)

        x, y, w, h = crop_dims
        start, end = trim_range

        for i in range(start, end):
            # get frame and crop it
            frame = self._frame_reader[i]
            frame = frame[y : y + h, x : x + w]

            file_name = self._frame_reader.files[i] if name_format is None else name_format.format(i)
            full_path = join_paths(save_folder, file_name)
            cv.imwrite(full_path, frame)
