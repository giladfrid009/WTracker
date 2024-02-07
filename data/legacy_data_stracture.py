import pickle
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VideoProperties:
    resolution: tuple[int, int]
    pixel_size: float
    fps: float
    total_frames: int


@dataclass
class BBox:
    x: int
    y: int
    h: int
    w: int


@dataclass
class BBoxes:
    num_of_samples: int
    bboxes: np.ndarray = field(init=False)  # ndarray of shape Nx4; each column is (x,y,w,h)^T

    def __post_init__(self):
        self.bboxes = np.full((self.num_of_samples, 4), fill_value=np.nan)

    def validate_frame(self, frame):
        if frame < 0 or frame >= self.num_of_samples:
            raise ValueError("ERROR:: invalid frame num")

    def get_bbox(self, frame: int) -> tuple[int, int, int, int]:
        self.validate_frame(frame)
        sample = self.bboxes[frame]
        return (int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3]))

    def add_bbox(self, frame: int, y: int, x: int, height: int, width: int) -> None:
        self.validate_frame(frame)
        self.bboxes[frame] = np.array([x, y, height, width]).T


@dataclass
class HeadCoord:
    x: int
    y: int


@dataclass
class HeadCoordinates:
    num_of_samples: int
    head_coords: np.ndarray = field(init=False)  # ndarray of shape Nx2; each column is (x,y)^T

    def __post_init__(self):
        self.head_coords = np.full((self.num_of_samples, 2), fill_value=np.nan)

    def validate_frame(self, frame: int):
        if frame < 0 or frame >= self.num_of_samples:
            raise ValueError("ERROR:: invalid frame num")

    def get_coord(self, frame: int) -> tuple[int, int]:
        self.validate_frame(frame)
        sample = self.head_coords[frame]
        return (int(sample[0]), int(sample[1]))

    def add_coord(self, frame: int, y: int, x: int) -> None:
        self.validate_frame(frame)
        self.head_coords[frame] = np.array([int(x), int(y)]).T


@dataclass
class Labels:
    bboxes: BBoxes
    head_coords: HeadCoordinates


@dataclass
class VideoSample:
    video_path: str
    resolution: tuple[int, int]
    num_of_frames: int
    starting_frame: int
    bboxes: BBoxes = field(init=False)
    head_coords: HeadCoordinates = field(init=False)

    def __post_init__(self):
        self.bboxes = BBoxes(self.num_of_frames)
        self.head_coords = HeadCoordinates(self.num_of_frames)


@dataclass
class ExperimentData:
    video_properties: VideoProperties
    video_samples: [VideoSample] = field(default_factory=list)
    description: str = ""

    def add_video_sample(self, sample: VideoSample) -> None:
        self.video_samples.append(sample)


class Pickler:
    def __init__(self, filename):
        self.file_path = filename

    def save_object(self, obj):
        """Saves an object to a pickle file."""
        try:
            with open(self.file_path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise ValueError(f"Error saving object to pickle file: {e}")

    def load_object(self):
        """Loads an object from a pickle file."""
        try:
            with open(self.file_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error file does not exist: {self.file_path}")
            return None  # Return None if file doesn't exist
        except Exception as e:
            raise ValueError(f"Error loading object from pickle file: {e}")


properties = VideoProperties((4000, 5600), 6.6667, 40, 1200)
sample1 = VideoSample("vid1.avi", 200, 0)
sample1.head_coords.add_coord(0, 2, 3)
print(sample1.head_coords.get_coord(0))
experiment = ExperimentData(properties)
experiment.add_video_sample(sample1)


print(experiment)

# pickler = Pickler('data_test1.pickle')
# data = pickler.load_object()
# print(data)
