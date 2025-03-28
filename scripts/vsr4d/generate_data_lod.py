import os
import sys
import argparse
import tempfile
import subprocess
from typing import List
import cv2
import yaml
import numpy as np
import torch
import imageio
from PIL import Image
from easyvolcap.utils.console_utils import *

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(base_path)

from lib.utils import *


class EVCDataManager:
    def __init__(self, data_root, gen_lr, gen_sr):
        self.data_root = data_root
        assert os.path.exists(data_root), f"{data_root} does not exist"

        self.gen_lr = gen_lr  # whether to generate low resolution images
        self.gen_sr = gen_sr  # whether to generate super resolution images

        self.create_data_folder()
        self.metadata = self.read_base_metadata()

    def create_data_folder(self):
        self.image_orig_dir = join(self.data_root, "images")
        self.video_dir = join(self.data_root, "videos")
        os.makedirs(self.image_orig_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        if self.gen_lr:
            self.image_lr_dir = join(self.data_root, f"images_lr")
            self.video_lr_dir = join(self.data_root, f"videos_lr")

        if self.gen_sr:
            self.image_sr_dir = join(self.data_root, "images_sr")
            self.video_sr_dir = join(self.data_root, "videos_sr")

    def read_base_metadata(self):
        metadata = dotdict()

        # get the number of views
        sub_image_dirs = sorted(os.listdir(self.image_orig_dir))
        metadata.n_views = len(sub_image_dirs)

        # get the metadata of each view
        metadata.view_meta = dotdict()
        for i, sub_image_dir in enumerate(sub_image_dirs):
            view_meta = dotdict()
            view_meta.viewId = int(sub_image_dir)
            view_meta.image_dir = join(self.image_orig_dir, sub_image_dir)
            view_meta.n_frames = len(os.listdir(view_meta.image_dir))
            tmp_img_path = join(view_meta.image_dir, os.listdir(view_meta.image_dir)[0])
            view_meta.img_format = tmp_img_path.split(".")[-1]
            tmp_img = cv2.imread(tmp_img_path)
            view_meta.height, view_meta.width = tmp_img.shape[:2]
            metadata.view_meta[i] = view_meta

        return metadata

    def generate_lr_images(self, frame_range: List[int], lr_res="1080p"):
        if lr_res == "1080p":
            data_factor = 2
        elif lr_res == "512p":
            data_factor = 4
        elif lr_res == "256p":
            data_factor = 8
        elif lr_res == "128p":
            data_factor = 16
        else:
            raise NotImplementedError
            assert (
                False
            ), f"Unsupported low resolution resolution: {lr_res}, choose 1080p and 512p"

        self.metadata.lr_res = lr_res
        self.metadata.lr_data_factor = data_factor
        self.metadata.lr_frame_range = frame_range
        self.image_lr_dir = self.image_lr_dir + f"_{self.metadata.lr_res}"
        self.metadata.lr_n_frames = len(range(*frame_range))
        os.makedirs(self.image_lr_dir, exist_ok=True)

        for idx, view_meta in tqdm(
            self.metadata.view_meta.items(),
            desc=f"Generating low resolution images with {lr_res}",
        ):
            viewId = view_meta.viewId
            img_dir = view_meta.image_dir
            img_save_dir = join(self.image_lr_dir, f"{viewId:02d}")
            os.makedirs(img_save_dir, exist_ok=True)
            resize_image(
                img_dir, img_save_dir, frame_range, data_factor, view_meta.img_format
            )

    def generate_lr_videos(self, fps=30):
        self.video_lr_dir = self.video_lr_dir + f"_{self.metadata.lr_res}"
        self.metadata.lr_fps = fps
        os.makedirs(self.video_lr_dir, exist_ok=True)
        b, e, s = self.metadata.lr_frame_range
        for idx, view_meta in tqdm(
            self.metadata.view_meta.items(),
            desc=f"Generating low resolution videos with {self.metadata.lr_res}",
        ):
            image_dir = join(self.image_lr_dir, f"{view_meta.viewId:02d}")
            os.makedirs(image_dir, exist_ok=True)
            images = read_images(image_dir, self.metadata.lr_frame_range)
            video_base_name = f"{view_meta.viewId:02d}_{b:06d}_{e:06d}.mp4"
            video_path = join(self.video_lr_dir, video_base_name)
            assert (
                self.metadata.lr_n_frames % fps == 0
            ), f"Number of frames should be divisible by fps"
            write_video(images, video_path, fps=fps)

    def generate_sr_images(
        self, frame_range, lr_res, video_dir=None, image_dir=None, use_resize=False
    ):
        self.video_sr_dir = self.video_sr_dir + f"_from_{lr_res}"
        self.image_sr_dir = self.image_sr_dir + f"_from_{lr_res}"
        os.makedirs(self.image_sr_dir, exist_ok=True)
        os.makedirs(self.video_sr_dir, exist_ok=True)
        read_video_dir = self.video_lr_dir if video_dir is None else video_dir
        read_image_dir = self.image_lr_dir if image_dir is None else image_dir

        pointer = 24
        for video in tqdm(
            sorted(os.listdir(video_dir)), desc="Extracting super resolution images"
        ):
            video_path = join(read_video_dir, video)
            viewId = int(video.split("_")[0])
            # image_dir = join(read_image_dir, f'{viewId:02d}')
            image_dir = join(read_image_dir, f"{pointer:02d}")
            pointer += 1
            target_size = (
                self.metadata.view_meta[viewId].width,
                self.metadata.view_meta[viewId].height,
            )
            if use_resize:
                extract_video_with_resize(video_path, image_dir, target_size)
            else:
                extract_video(video_path, image_dir)

    def read_roi_meta(self, roi_path):
        roi_metas = []
        with open(roi_path, "r") as f:
            roi = f.readlines()
            # split the roi into 5 parts: viewid, x, y, w, h
            roi_meta = roi[1:]
            for r in roi_meta:
                roi_metas.append([int(x) for x in r.split()])
        roi_metas = sorted(roi_metas, key=lambda x: x[0])
        return roi_metas

    def crop_images_with_roi(self, roi_path, frame_range):
        roi_metas = self.read_roi_meta(roi_path)
        image_crop_dir = self.image_orig_dir + "_cropped"
        os.makedirs(image_crop_dir, exist_ok=True)

        for idx, roi_meta in enumerate(
            tqdm(self.roi_metas, desc="Cropping images with roi")
        ):
            viewId = roi_meta[0]
            img_dir = join(self.image_orig_dir, f"{viewId:02d}")
            img_save_dir = join(image_crop_dir, f"{viewId:02d}")
            os.makedirs(img_save_dir, exist_ok=True)
            images = read_images(img_dir, frame_range)
            videos = []
            for i, img in enumerate(tqdm(images, desc=f"Processing view {viewId:02d}")):
                x, y, w, h = roi_meta[1:]
                img = img[y : y + h, x : x + w]
                videos.append(img)
                img_path = join(img_save_dir, f"{i+200:06d}.png")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, img)
            os.makedirs(join(image_crop_dir, "video"), exist_ok=True)
            video_path = join(image_crop_dir, "video", f"{viewId:02d}.mp4")
            write_video(videos, video_path, fps=30)

    def update_evc_cameras(self, roi_path):
        roi_metas = self.read_roi_meta(roi_path)
        self.cam_intri_path = join(self.data_root, "intri.yml")
        self.cam_extri_path = join(self.data_root, "extri.yml")
        with open(self.cam_intri_path, "r") as f:
            intri = yaml.load(f)
        with open(self.cam_extri_path, "r") as f:
            extri = yaml.load(f)

        all_names_intri = sorted(intri["names"])
        all_names_extri = sorted(extri["names"])
        assert all_names_intri == all_names_extri, "Intri and Extri names do not match"
        max_view_id = max([int(x) for x in all_names_intri])
        for idx, roi_meta in enumerate(
            tqdm(roi_metas, desc="Updating camera intrinsics")
        ):
            coord_viewID = roi_meta[0]
            new_view_id = max_view_id + 1 + idx
            intri["names"].append(str(new_view_id))
            extri["names"].append(str(new_view_id))

            roi_x, roi_y, roi_w, roi_h, sr_w, sr_h = roi_meta[1:]
            fx, fy, cx, cy = (
                intri["K_00"]["data"][0],
                intri["K_00"]["data"][4],
                intri["K_00"]["data"][2],
                intri["K_00"]["data"][5],
            )
            cx = cx - roi_x
            cy = cy - roi_y

            ratio_x = sr_w // roi_w
            ratio_y = sr_h // roi_h
            fx = fx * ratio_x
            fy = fy * ratio_y
            cx = cx * ratio_x
            cy = cy * ratio_y

            # update the intrinsics
            intri[f"K_{new_view_id:2d}"] = intri["K_00"].copy()
            intri[f"K_{new_view_id:2d}"]["data"] = intri["K_00"]["data"].copy()
            intri[f"K_{new_view_id:2d}"]["data"][0] = fx
            intri[f"K_{new_view_id:2d}"]["data"][4] = fy
            intri[f"K_{new_view_id:2d}"]["data"][2] = cx
            intri[f"K_{new_view_id:2d}"]["data"][5] = cy
            intri[f"H_{new_view_id:2d}"] = intri["H_00"]
            intri[f"W_{new_view_id:2d}"] = intri["W_00"]
            intri[f"D_{new_view_id:2d}"] = intri["D_00"].copy()
            intri[f"D_{new_view_id:2d}"]["data"] = intri["D_00"]["data"].copy()
            intri[f"ccm_{new_view_id:2d}"] = intri["ccm_00"].copy()
            intri[f"ccm_{new_view_id:2d}"]["data"] = intri["ccm_00"]["data"].copy()

            # update the extrinsics
            print(f"now: {idx} orig: {coord_viewID} new: {new_view_id}")
            extri[f"Rot_{new_view_id:2d}"] = extri[f"Rot_{coord_viewID:02d}"].copy()
            extri[f"Rot_{new_view_id:2d}"]["data"] = extri[f"Rot_{coord_viewID:02d}"][
                "data"
            ].copy()
            extri[f"R_{new_view_id:2d}"] = extri[f"R_{coord_viewID:02d}"].copy()
            extri[f"R_{new_view_id:2d}"]["data"] = extri[f"R_{coord_viewID:02d}"][
                "data"
            ].copy()
            extri[f"T_{new_view_id:2d}"] = extri[f"T_{coord_viewID:02d}"].copy()
            extri[f"T_{new_view_id:2d}"]["data"] = extri[f"T_{coord_viewID:02d}"][
                "data"
            ].copy()

        # dump yaml
        intri_save_path = self.cam_intri_path.replace(".yml", "_new.yml")
        with open(intri_save_path, "w") as f:
            yaml.dump(intri, f)
        extri_save_path = self.cam_extri_path.replace(".yml", "_new.yml")
        with open(extri_save_path, "w") as f:
            yaml.dump(extri, f)

    def update_view_images(self, image_update_dir):
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automation script for generating data for LOD training"
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        type=str,
        default="/home/zhouchenxu/datasets/vsr_4dv/shijie_far",
        help="Path to root data",
    )
    parser.add_argument(
        "--gen_lr", action="store_true", help="Generate low resolution images"
    )
    parser.add_argument(
        "--gen_sr", action="store_true", help="Generate super resolution images"
    )
    parser.add_argument(
        "--lr_res",
        type=str,
        default="1080p",
        help="Low resolution resolution(1080p/512p/256p/128p)",
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--frame_range", nargs=3, type=int, default=[0, 30, 1], help="start, end, step"
    )
    parser.add_argument("--use_crop", action="store_true", help="Crop images with roi")
    parser.add_argument(
        "--roi_path", type=str, default=None, help="Path to roi info path"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Path to videos for extracting sr images",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to images for extracting sr images",
    )
    args = parser.parse_args()

    return args


def main():
    # set args
    args = parse_args()

    # init data manager
    data_root = args.root_dir
    gen_lr = args.gen_lr
    gen_sr = args.gen_sr
    lr_res = args.lr_res
    fps = args.fps
    frame_range = args.frame_range
    use_crop = args.use_crop
    roi_path = args.roi_path
    video_dir = args.video_dir
    image_dir = args.image_dir

    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
    if image_dir is not None:
        os.makedirs(image_dir, exist_ok=True)

    evc_dm = EVCDataManager(data_root, gen_lr=gen_lr, gen_sr=gen_sr)

    # generate low resolution images and convert to videos
    if gen_lr:
        evc_dm.generate_lr_images(frame_range=frame_range, lr_res=lr_res)
        evc_dm.generate_lr_videos(fps=fps)

    # extract super resolution videos to images
    if gen_sr:
        evc_dm.generate_sr_images(
            frame_range=frame_range,
            lr_res=lr_res,
            video_dir=video_dir,
            image_dir=image_dir,
            use_resize=False,
        )

    if use_crop and roi_path is not None:
        evc_dm.crop_images_with_roi(roi_path, frame_range)

    if roi_path is not None:
        evc_dm.update_evc_cameras(roi_path)


if __name__ == "__main__":
    main()
