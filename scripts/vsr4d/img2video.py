import os
import sys
from glob import glob

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)

from lib.utils.data_utils import load_image, generate_video
from lib.utils.console_utils import *
from lib.utils.parallel_utils import parallel_execution


def main():
    img_dir = (
        "/home/zhouchenxu/datasets/vsr_4dv/shijie_far/exp_data/images_cropped_back"
    )
    output_dir = "/home/zhouchenxu/datasets/vsr_4dv/shijie_far/exp_data/images_cropped_back/videos_crop"
    os.makedirs(output_dir, exist_ok=True)
    viewdir = sorted(os.listdir(img_dir))
    for view in viewdir:
        result_str = f"{img_dir}/{view}/*.png"
        generate_video(result_str, join(output_dir, f"{view}.mp4"), fps=30)
        print(f"Video saved to {output_dir}/{view}.mp4")


if __name__ == "__main__":
    main()
    main()
