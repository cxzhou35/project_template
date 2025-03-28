from typing import List
import cv2
import numpy as np
import torch
import imageio
from torch.nn import functional as F
from lib.utils.console_utils import *

def read_image(
    img_path: str,
):
    pass


def interpolate_image(
    img: torch.Tensor, mode="bilinear", align_corners=False, *args, **kwargs
):
    # Performs F.interpolate as images (always augment to B, C, H, W)
    sh = img.shape
    img = img.view(-1, *sh[-3:])
    img = F.interpolate(
        img,
        *args,
        mode=mode,
        align_corners=align_corners if mode != "nearest" else None,
        **kwargs,
    )
    img = img.view(sh[:-3] + img.shape[-3:])
    return img


def resize_image(
    img: torch.Tensor, mode="bilinear", align_corners=False, *args, **kwargs
):
    # example: resize_image(img, uH, uW, 'bilinear')
    sh = img.shape
    if len(sh) == 4:
        img = img.permute(0, 3, 1, 2)
    elif len(sh) == 3:
        img = img.permute(2, 0, 1)[None]
    img = interpolate_image(
        img, mode=mode, align_corners=align_corners, *args, **kwargs
    )  # uH, uW, 3
    if len(sh) == 4:
        img = img.permute(0, 2, 3, 1)
    elif len(sh) == 3:
        img = img[0].permute(1, 2, 0)
    return img
