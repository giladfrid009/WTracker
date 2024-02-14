import glob
from pathlib import Path

from data.file_utils import create_directory
from data.bbox_utils import BoxFormat, BoxUtils
from data.frame_dataset import *


class DatasetConverter:
    @staticmethod
    def to_yolo(frame_dataset: FrameDataset, labels_path: str, images_path: str):
        create_directory(labels_path)
        create_directory(images_path)

        for sample in frame_dataset:
            src_img_path = sample.metadata.path
            dst_ann_path = Path(labels_path) / f"{Path(src_img_path).stem}.txt"
            dst_img_path = Path(images_path) / Path(src_img_path).name

            keypoints = sample.keypoints
            bboxes = sample.bboxes

            if keypoints is None or bboxes is None:
                raise Exception("both bboxes and keypoints must be present for YOLO format conversion")

            # transfer bbox to yolo format
            bboxes = BoxUtils.change_format(bboxes, sample.bbox_format, BoxFormat.XYmid_WH)

            # normalize all sizes between 0 and 1
            width, height = sample.metadata.size
            norm_bboxes = np.zeros_like(bboxes, dtype=float)
            norm_keypoints = np.zeros_like(keypoints, dtype=float)
            
            norm_bboxes[:, ::2] = bboxes[:, ::2] / width 
            norm_bboxes[:, 1::2] = bboxes[:, 1::2] / height
            
            norm_keypoints[:, :, 0] = keypoints[:, :, 0] / width
            norm_keypoints[:, :, 1] = keypoints[:, :, 1] / height

            # round values to not save too many digits
            norm_bboxes = norm_bboxes.round(decimals=4)
            norm_keypoints = norm_keypoints.round(decimals=4)

            # unpack along first axis into lists
            bbox_list = [box for box in norm_bboxes]
            kps_list = [kp for kp in norm_keypoints]

            class_index = 0
            with dst_ann_path.open(mode="w") as ann_file:
                for bbox, kps in zip(bbox_list, kps_list):
                    object_data = [class_index] + bbox.tolist() + kps.flatten().tolist()
                    object_str = " ".join([str(x) for x in object_data])
                    ann_file.write(object_str + "\n")

            dst_img_path.hardlink_to(src_img_path)

    def from_yolo(experiment_metadata: ExperimentMeta, labels_path: str, images_path: str) -> FrameDataset:
        dataset = FrameDataset(experiment_metadata, [])

        glob_ann_format = (Path(labels_path) / "*.txt").as_posix()
        ann_paths = glob.glob(glob_ann_format)
        ann_paths = sorted([Path(p) for p in ann_paths], key=lambda p: p.stem)

        glob_img_format = (Path(images_path) / "*.*").as_posix()
        img_paths = glob.glob(glob_img_format)
        img_paths = sorted([Path(p) for p in img_paths], key=lambda p: p.stem)

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

        return dataset
