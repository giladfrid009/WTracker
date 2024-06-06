import matplotlib.pyplot as plt
from wtracker.eval import *
from wtracker.sim.config import TimingConfig
from wtracker.utils.gui_utils import UserPrompt
from pprint import pprint
from wtracker.eval.plotter import Plotter
from wtracker.eval.data_analyzer import DataAnalyzer

from wtracker.eval.vlc import StreamViewer
from wtracker.eval.error_calculator import ErrorCalculator
from wtracker.utils.frame_reader import FrameReader
import pandas as pd
import numpy as np
from wtracker.utils.path_utils import Files


def run(base_path:str, background_path:str, exp_bounds=None, worm_folder_path:str="", diff_thresh:float=1.0):
    log_file = base_path + "\\bboxes.csv"
    # data_save_path = base_path + "\\data.pkl"
    # path to the timing config file. 
    # If None, a file dialog will open to select a file
    time_config_path = base_path + "\\time_config.json"
    timing_config = TimingConfig.load_json(time_config_path)

    
    pprint(timing_config)
    pprint(log_file)

    data = DataAnalyzer(
        time_config=timing_config,
        log_path=log_file,
        unit="sec",
    )

    data.run_analysis(
        period=10,
        imaging_only=False,
        legal_bounds=exp_bounds,
    )

    # if background_path is None:
    #     background_path = UserPrompt.open_file(title="Select background images", file_types=[("Numpy files", "*.npy")])

    # if worm_folder_path is None:
    #     worm_folder_path = UserPrompt.open_directory(title="Select worm image folders")

    # print("Background Files: ", background_path)
    # print("Worm Image Folders: ", worm_folder_path)

    # background = np.load(background_path, allow_pickle=True)

    # worm_reader = FrameReader.create_from_directory(worm_folder_path)

    # data.calc_precise_error(
    #     worm_image_paths=worm_folder_path,
    #     background=background,
    #     diff_thresh=diff_thresh,
    # )

    # if data_save_path is None:
    #     data_save_path = UserPrompt.save_file(title="Save data", filetypes=[("Pickle files", "*.pkl")])

    # data.save(base_path + "\\dataAnalyzer.pkl")
    data.table.to_csv(base_path + "\\analysis.csv")





if __name__ == "__main__":
    
    
    # background_path = "data\\Exp1_GuyGilad_logs_yolo\\background.npy"

    # worm_folder_path = "D:\\Guy_Gilad\\Exp1_GuyGilad\\logs_yolo\\worms"

    # diff_thresh = 20

    # exp_threshes = {
    #     "Exp0": 0,
    #     "Exp1": 0,
    #     "Exp2": 0,
    #     "Exp3": 0,
    # }

    evals_path = "D:\\Guy_Gilad\\FinalEvaluations"

    folders = Files(evals_path, scan_dirs=True)

    exp_bounds = {
        "Exp0": (73,38,1551,1359),
        "Exp1": (39,32,1467,1301),
        "Exp2": (6,57,1495,1336),
        "Exp3": (10,60,1498,1322),
        "Exp4": (24,71,1496,1335),
    }

    total = len(folders)
    for i, eval_path in enumerate(folders):

        print(f"[{i}/{total}] :: Analizing: {eval_path}")
        bounds = None
        for exp_name, bound in exp_bounds.items():
            if exp_name in eval_path:
                bounds = bound
                break

        run(eval_path, "", bounds)
 









