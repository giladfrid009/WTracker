import os
import cv2 as cv
import threading
from data.tqdm_utils import TqdmQueue
import numpy as np
from view_controller import ViewController
from dataclasses import dataclass, field
import math
from ultralytics import YOLO
from data.frame_reader import FrameReader
from data.file_utils import pickle_load_object, pickle_save_object
import json
from logging_utils import CSVLogger, TrackingLog

@dataclass
class SimConfig:
    fps:int
    micro_view_time_ms:float
    tracking_time_ms:float
    moving_time_ms:float

    frame_time_ms:float = field(init=False)
    micro_view_frames:int = field(init=False)
    tracking_frames:int = field(init=False)
    moving_frames:int = field(init=False)

    database_path:str
    output_path:str
    YOLO_weights:str
    
    px_per_mm:int
    camera_size_mm:tuple[float,float]
    micro_size_mm:tuple[float,float]

    camera_size_px:tuple[float, float] = field(init=False)
    micro_size_px:tuple[float, float] = field(init=False)
    
    init_position:tuple[int, int]
    reader_padding_value:list[int,int,int]=field(default_factory=lambda:[255, 255, 255])

    tracking_color:list[int,int,int] = field(default_factory=lambda:[0, 0, 255])
    moving_color:list[int,int,int] = field(default_factory=lambda:[255, 255, 255])
    micro_view_color:list[int,int,int] = field(default_factory=lambda:[0, 0, 0])

    def __post_init__(self):
        self.frame_time_ms = 1000 / self.fps
        self.micro_view_frames = math.ceil(self.micro_view_time_ms / self.frame_time_ms)
        self.tracking_frames = math.ceil(self.tracking_time_ms / self.frame_time_ms)
        self.moving_frames = math.ceil(self.moving_time_ms / self.frame_time_ms)

        self.camera_size_px = (round(self.px_per_mm * self.camera_size_mm[0]), round(self.px_per_mm * self.camera_size_mm[1]))
        self.micro_size_px = (round(self.px_per_mm * self.micro_size_mm[0]), round(self.px_per_mm * self.micro_size_mm[1]))
    
    @staticmethod
    def load_from_json(filepath):
        """
        Loads a JSON file and returns the data as a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: The dictionary containing the data from the JSON file.
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return SimConfig(**data)
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in file: {filepath}")
            print(f"Error details: {e}")
            return None

    def save_json(self, filepath):
        """
        Saves a dictionary as JSON data to a file.

        Args:
            data (dict): The dictionary to save as JSON.
            filepath (str): The path to the output JSON file.
        """
        try:
            with open(filepath, "w") as f:
                json.dump(self.__dict__, f, indent=4)  # Optional: Add indentation for readability
        except IOError as e:
            print(f"Error: Failed to save JSON data to file: {filepath}")
            print(f"Error details: {e}")


class Evaluator:
    def __init__(self, config:SimConfig) -> None:
        # self.worker = ImageSaver()
        self.config = config
        self.output_paths = {"root": config.output_path,
                             "micro": config.output_path+"micro/",
                             "cam": config.output_path+"cam/",
                             "errors": config.output_path+"errors/",
                            }
        
        self.model = self.load_model()
        self.view_controller = self.create_view_controller()
        self.create_folders()
        
        self.box_logger:CSVLogger = CSVLogger(self.output_paths["root"]+"tracking_log.csv", TrackingLog().__dict__.keys())

        print(self.__dict__)
        pass

    @staticmethod
    def create_boundry(image:np.ndarray, boundry_size:int, color:int) -> np.ndarray:
        frame = cv.copyMakeBorder(
            src=image,
            left=boundry_size,
            right=boundry_size,
            top=boundry_size,
            bottom=boundry_size,
            borderType=cv.BORDER_CONSTANT,
            value=color,
        )
        return frame

    def create_folders(self):
        for folder in self.output_paths.values():
            os.makedirs(folder)

    def load_model(self)->YOLO:
        return YOLO(self.config.YOLO_weights, task='detect')
    
    def create_view_controller(self, file_list:list[str]=None):
        frame_reader:FrameReader = None
        if file_list is not None:
            frame_reader = FrameReader(self.config.database_path, file_list, cv.IMREAD_COLOR)
        frame_reader = FrameReader.create_from_directory(self.config.database_path, cv.IMREAD_COLOR)
    
        return ViewController(frame_reader, self.config.camera_size_px, self.config.micro_size_px, self.config.init_position, self.config.reader_padding_value)

    def check_init_coords(self, x:int, y:int):
        self.view_controller.seek(0)
        self.view_controller.set_position(x, y)
        self.view_controller.visualize_world()
    
    def dump_config(self):
        self.config.save_json(self.output_paths["root"]+"config.json")

    def make_vid(self):
        command = f"ffmpeg -framerate 60 -start_number 0 -i {self.output_paths["micro"]}frame_%09d.png -c:v ffv1 {self.output_paths["root"]}output.avi"
        os.system(command)

    def simulate(self, save_cam_view:bool=False, device='cpu'):
        self.dump_config()
        self.view_controller.reset()
        mic_w, mic_h = self.view_controller.micro_size
        cam_w, cam_h = self.view_controller.camera_size
        dx, dy = (0,0)
        for i, _ in enumerate(self.view_controller):
            phase_color = [0,0,0]
            filename =  f"frame_{i:09d}.png"
            file_path = self.output_paths["micro"] + filename
            phase_pos = i%(self.config.moving_frames+self.config.micro_view_frames)
            micro = self.view_controller.micro_view()

            view_pos = self.view_controller.position
            head_box = (-1, -1, -1, -1) # x,y,w,h
            mic_box = (view_pos[0] - mic_w//2, view_pos[1]-mic_h//2, mic_w, mic_h)
            cam_x, cam_y = (view_pos[0] - cam_w//2, view_pos[1]-cam_h//2)
            
            cam_view = self.view_controller.camera_view()
            pred = self.model.predict(cam_view, conf=0.1, device=device, imgsz=416)
            
            if len(pred) > 0 and pred[0].boxes.xyxy.shape[0] > 0:
                head_box = pred[0].boxes.xywh[0]
                head_box = (head_box[0].item()+cam_x, head_box[1].item()+cam_y, head_box[2].item(), head_box[3].item())
                head_box = (math.floor(head_box[0]), math.floor(head_box[1]), math.ceil(head_box[2]), math.ceil(head_box[3]))
                pred_box = pred[0].boxes.xyxy[0]
                pred_center = (((pred_box[0]+pred_box[2])/2).to(int).item(), ((pred_box[1]+pred_box[3])/2).to(int).item())
                if phase_pos == 0:
                    dx, dy = (pred_center[0]-self.config.camera_size_px[0]//2, pred_center[1]-self.config.camera_size_px[1]//2)  # Takes 'track_frames' frames, can't do anything meanwhile
            else:
                if phase_pos == 0:
                    dx, dy = (0,0)
                print(f"ERROR:: No Head detected in frame {i}")
                cv.imwrite(self.output_paths["errors"]+f"NoPred_{i}_cam.png", self.view_controller.camera_view())
                cv.imwrite(self.output_paths["errors"]+f"NoPred_{i}_micro.png", self.view_controller.micro_view())

            track_log = TrackingLog.from_boxes(i, mic_box, head_box)
            self.box_logger.write(track_log.__dict__)
            self.box_logger.flush()

            #Change pos - end of movement
            if phase_pos == self.config.tracking_frames + self.config.moving_frames - 1:
                self.view_controller.move_position(dx, dy)
            
            # All micro view phase
            if (0 <= phase_pos < self.config.tracking_frames) or (self.config.tracking_frames+self.config.moving_frames <= phase_pos):
                phase_color = self.config.micro_view_color
                # micro = self.create_boundry(micro,2,self.config.moving_color)
            # All tracking phase
            if 0 <= phase_pos < self.config.tracking_frames:
                phase_color = self.config.tracking_color
                # micro = self.create_boundry(micro,2,self.config.moving_color)
            # All movement phase
            if self.config.tracking_frames <= phase_pos < self.config.tracking_frames + self.config.moving_frames:
                phase_color = self.config.moving_color
                # micro = self.create_boundry(micro,2,self.config.moving_color)
            
            micro = self.create_boundry(micro,2,phase_color)
            cv.imwrite(file_path, micro)
            if save_cam_view:
                cam_view = self.create_boundry(self.view_controller.camera_view(),2,phase_color)
                cv.imwrite(self.output_paths["cam"]+filename, cam_view)

        
        self.make_vid()
    