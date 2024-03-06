import typing
from dataclasses import dataclass, field
import csv


@dataclass
class Logger:
    path:str
    mode:str = field(default='w')
    file:typing.TextIO = field(init=False)

    def __post_init__(self):
        self.file = open(self.path, self.mode)
    
    def close(self):
        if not self.file.closed():
            self.file.flush()
            self.file.close()
        pass

    def write(self, string:str):
        if self.file.writable():
            self.file.write(string)
    
    def writelines(self, strings:list[str]):
        if self.file.writable():
            self.file.writelines(strings)
    
    def flush(self):
        self.file.flush()

@dataclass
class CSVLogger:
    path:str
    col_names:list[str]
    mode:str = field(default="w+")
    file:typing.TextIO = field(init=False)
    writer:csv.DictWriter = field(init=False)
    

    def __post_init__(self):
        self.file = open(self.path, self.mode, newline='')
        self.writer = csv.DictWriter(self.file, self.col_names,escapechar=',')
        self.writer.writeheader()
        self.flush()
    
    def close(self):
        if not self.file.closed():
            self.file.flush()
            self.file.close()
        pass

    def write(self, row:dict):
        if self.file.writable():
            self.writer.writerow(row)
    
    def writerows(self, rows:list[dict]):
        if self.file.writable():
            self.writer.writerows(rows)
    
    def flush(self):
        self.file.flush()

from ultralytics import YOLO
from data.frame_reader import FrameReader
from view_controller import ViewController
from multiprocessing import Process, Queue
import numpy as np
import cv2 as cv

@dataclass(frozen=True)
class TrackingLog:
    frame:int = -1
    micro_x:int = -1
    micro_y:int = -1
    micro_w:int = -1
    micro_h:int = -1
    head_x:int = -1
    head_y:int = -1
    head_w:int = -1
    head_h:int = -1

    @staticmethod
    def from_boxes(frame:int, micro_box:tuple[int,int,int,int], head_box:tuple[int,int,int,int]):
        micro_x, micro_y, micro_w, micro_h = micro_box
        head_x, head_y, head_w, head_h = head_box
        return TrackingLog(frame, micro_x, micro_y, micro_w, micro_h, head_x, head_y, head_w, head_h)

class TrackingProcess:
    def __init__(self, yolo_weights:str, pred_queue:Queue, update_queue:Queue) -> None:
        self.queue = pred_queue
        self.model:YOLO = YOLO(yolo_weights, task='detect')
        pass

    def worker(self):
        pass
    def _saver_worker(self, video_params: Queue):
        """
        Worker method that processes saving tasks from the queue.
        Returns:
            None
        """
        while True:
            task = self.queue.get()

            # exit if signaled
            if task is None:
                break

            save_folder, trim_range, crop_dims, name_format = task
            self._crop_and_save_video(save_folder, trim_range, crop_dims, name_format)
            video_params.task_done()

    def predict(self, cam_view:np.ndarray, frame_num:int, device:str='cpu'):
        pred = self.model.predict(cam_view, conf=0.1, device=device, imgsz=416)
                
        if len(pred) > 0 and pred[0].boxes.xywh.shape[0] > 0:
            head_x,head_y,head_w,head_h = pred[0].boxes.xywh[0]
            pred_box = pred[0].boxes.xyxy[0]
            pred_center = (((pred_box[0]+pred_box[2])/2).to(int).item(), ((pred_box[1]+pred_box[3])/2).to(int).item())
            dx, dy = (pred_center[0]-self.config.camera_size_px[0]//2, pred_center[1]-self.config.camera_size_px[1]//2)  # Takes 'track_frames' frames, can't do anything meanwhile
            
        else:
            print(f"ERROR:: No Head detected in frame {frame_num}")
            # cv.imwrite(self.output_paths["errors"]+f"NoPred_{frame_num}_cam.png", self.view_controller.camera_view())
            # cv.imwrite(self.output_paths["errors"]+f"NoPred_{frame_num}_micro.png", self.view_controller.micro_view())
