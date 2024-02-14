import glob
from file_helpers import create_directory
from bbox_utils import BoxFormat, BoxUtils
from pathlib import Path
from data.frame_dataset import *


class DatasetConverter:
    @staticmethod
    def to_yolo(frame_dataset: FrameDataset, dst_dir: str):
        create_directory(dst_dir)

        for sample in frame_dataset:
            src_img_path = sample.metadata.path
            dst_ann_path = Path(dst_dir) / f"{Path(src_img_path).stem}.txt"
            dst_img_path = Path(dst_dir) / Path(src_img_path).name

            keypoints = sample.keypoints
            bboxes = sample.bboxes

            if keypoints is None or bboxes is None:
                raise Exception("both bboxes and keypoints must be present for YOLO format conversion")

            # transfer bbox to yolo format
            bboxes = BoxUtils.change_format(bboxes, sample.bbox_format, BoxFormat.XYmid_WH)

            # normalize all sizes between 0 and 1
            width, height = sample.metadata.size
            bboxes[:, ::2] /= width
            bboxes[:, 1::2] /= height
            keypoints[:, :, 0] /= width
            keypoints[:, :, 1] /= height

            # round values to not save too many digits
            bboxes = bboxes.round(decimals=4)
            keypoints = keypoints.round(decimals=4)

            num_objects = bboxes.shape[0]
            bbox_list = np.split(bboxes, num_objects, axis=0)
            kps_list = np.split(keypoints, num_objects, axis=0)

            class_index = 0
            with dst_ann_path.open(mode="w") as ann_file:
                for bbox, kps in zip(bbox_list, kps_list):
                    object_data = [class_index] + bbox.tolist() + kps.flatten().tolist()
                    object_str = " ".join([str(x) for x in object_data])
                    ann_file.write(object_str + "\n")

            dst_img_path.symlink_to(src_img_path)

    def from_yolo(experiment_metadata: ExperimentMeta, yolo_dir: str) -> FrameDataset:
        dataset = FrameDataset(experiment_metadata)

        ann_paths = glob.glob(yolo_dir, "*.txt")
        ann_paths = [Path(p) for p in glob.glob(yolo_dir, "*.txt")]
        img_paths = [Path(yolo_dir) / Path(ann_file).stem / ".bmp" for ann_file in ann_paths]

        for ann_file_path, img_file_path in zip(ann_paths, img_paths):
            frame_meta = FrameMeta.from_file(img_file_path.as_posix(), pixel_size=experiment_metadata.pixel_size)

            with ann_file_path.open("r") as ann_file:
                all_lines = ann_file.readlines()

            all_lines = [line.strip().split(" ") for line in all_lines]
            all_lines = [[float(x) for x in row] for row in all_lines]

            data = np.asanyarray(all_lines)

            bboxes = data[:, 1:5]
            keypoints = data[:, 5:].reshape((data.shape[0], -1, 2))

            # resize bboxes and kps from [0-1 x 0-1] to [width x height]
            width, height = frame_meta.size
            bboxes[:, ::2] *= width
            bboxes[:, 1::2] *= height
            keypoints[:, :, 0] *= width
            keypoints[:, :, 1] *= height
            
            bboxes = bboxes.astype(int)
            keypoints = keypoints.astype(int)

            # change bbox format to something normal
            bboxes = BoxUtils.change_format(bboxes, BoxFormat.XYmid_WH, BoxFormat.XY_XY)

            sample = Sample(frame_meta, bboxes, BoxFormat.XY_XY, keypoints=keypoints)

            dataset.add_sample(sample)
