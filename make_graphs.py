import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from wtracker.eval import *
from wtracker.sim.config import TimingConfig
from wtracker.utils.path_utils import Files
from wtracker.utils.gui_utils import UserPrompt
from wtracker.utils.path_utils import join_paths, create_directory
import seaborn as sns

exp_bounds = {
    "Exp0": (73, 38, 1551, 1359),
    "Exp1": (39, 32, 1467, 1301),
    "Exp2": (6, 57, 1495, 1336),
    "Exp3": (10, 60, 1498, 1322),
    "Exp4": (24, 71, 1496, 1335),
}


BAD_frames = {
    "Exp0": [(41109, 41113)],
    "Exp1": [(21930, 21942), (18741, 18763)],
    "Exp2": [(22474, 22745), (32859, 32860)],
    "Exp3": [(6843, 6853), (6872, 6915), (37750, 37751), (37807, 37808), (47485, 47506), (53117, 53122)],
    "Exp4": [
        (5700, 5798),
        (14963, 14986),
        (27816, 27817),
        (35785, 35804),
        (38839, 38864),
        (39250, 39300),
        (43543, 43544),
        (43661, 43674),
        (46848, 46859),
        (64874, 64905),
    ],
}


def remove_cycles(analyzer: DataAnalyzer, frames: tuple[int, int]):
    frame_range = np.asanyarray(list(range(frames[0], frames[1])), dtype=int)
    mask = analyzer.data["frame"].isin(frame_range)
    cycles = analyzer.data[mask]["cycle"].unique()
    analyzer.remove_cycle(cycles)
    return analyzer

def save_fig(fig:plt.Figure, path:str, dpi:int=100):
    fig.set_dpi(dpi)
    fig.tight_layout()
    fig.savefig(path + ".png", format="png")


config_names = ["config1", "config2", "config3", "config4"]
contr_names = ["CSV", "Optimal", "Poly", "ResMLP"]

all_dirs = Files("D:\\Guy_Gilad\\FinalEvaluations", scan_dirs=True)

save_dir_base = UserPrompt.open_directory("choose save dir base")

for conf_name in tqdm(config_names):
    for contr_name in tqdm(contr_names):

        save_dir = join_paths(save_dir_base, f"{conf_name}_{contr_name}")
        create_directory(save_dir)

        exp_dirs = [dir for dir in all_dirs if conf_name in dir and contr_name in dir]
        
        exp_dirs2 = []
        for exp_name in ["Exp0", "Exp1", "Exp2"]:
            for exp_dir in exp_dirs:
                if exp_name in exp_dir:
                    exp_dirs2.append(exp_dir)

        exp_dirs = exp_dirs2

        time_configs = [TimingConfig.load_json(join_paths(exp_dir, "time_config.json")) for exp_dir in exp_dirs]

        all_analyzers = [
            DataAnalyzer.load(time_conf, join_paths(exp_dir,"bboxes.csv")) for time_conf, exp_dir in zip(time_configs, exp_dirs)
        ]

        exp_names = []
        for exp_dir in exp_dirs:
            for name in BAD_frames.keys():
                if name in exp_dir:
                    exp_names.append(name)
                    break

        for i, an in enumerate(all_analyzers):
            exp_name = exp_names[i]
            an.initialize(period=10)
            an.clean(trim_cycles=True, bounds=exp_bounds[exp_name], imaging_only=False)
            an.calc_anomalies(no_preds=True, remove_anomalies=True)

            bad_preds = BAD_frames[exp_name]
            for frames in bad_preds:
                an = remove_cycles(an, frames)

            an.change_unit("sec")

        pltr = Plotter([an.data for an in all_analyzers], plot_height=7, palette=None)
        
        plot = pltr.plot_cycle_error(
            log_wise=False,
            plot_kind="boxen",
            k_depth="proportion",
            outlier_prop=0.02,
            saturation=0.5,
        )
        save_fig(plot.figure, path=join_paths(save_dir, "deviation"))

        q = pltr.data["worm_deviation"].quantile(0.995)
        cond = lambda d: d["worm_deviation"] < q
        plot = pltr.create_distplot(
            x_col="cycle_step",
            y_col="worm_deviation",
            x_label="cycle step",
            y_label="distance",
            title="Distance between worm and microscope centers as function of cycle step",
            plot_kind="hist",
            #condition=cond,
        )
        save_fig(plot.figure, path=join_paths(save_dir, "deviation_hist"))

        for an in all_analyzers:
            an.clean(imaging_only=True)

        pltr = Plotter([an.data for an in all_analyzers], plot_height=7, palette="bright")

        plot = pltr.plot_error(
            error_kind="bbox",
            log_wise=True,
            cycle_wise=True,
            hue_col="log_num",
            condition=lambda df: df["bbox_error"] > 1e-5,
        )
        save_fig(plot.figure, path=join_paths(save_dir, "bbox_error_hist"))

        plot = pltr.plot_error(
            error_kind="dist",
            log_wise=True,
            cycle_wise=True,
            hue_col="log_num",
            condition=lambda df: (df["worm_deviation"] > 1e-5) & (df["worm_deviation"] < 300),
        )
        save_fig(plot.figure, path=join_paths(save_dir, "dist_error_hist"))

        plot = pltr.plot_speed_vs_error(
            error_kind="bbox",
            cycle_wise=True,
            condition=lambda df: (df["wrm_speed"] < 1000) & (df["bbox_error"] > 1e-5),
        )
        save_fig(plot.figure, path=join_paths(save_dir, "speed_vs_bbox_error"))

        plot = pltr.plot_speed_vs_error(
            error_kind="dist",
            cycle_wise=True,
            condition=lambda df: (df["wrm_speed"] < 1000) & (df["worm_deviation"] < 300),
        )
        save_fig(plot.figure, path=join_paths(save_dir, "speed_vs_dist_error"))

        table = pltr.data[["wrm_speed", "bbox_error", "worm_deviation"]].describe(np.linspace(0.05, 1, 19, endpoint=False))

        table.to_csv(join_paths(save_dir, "error_table.csv"))


