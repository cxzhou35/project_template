import os
from dataclasses import dataclass
from os.path import dirname, isabs, join, relpath
from typing import List, Tuple

import cv2
import numpy as np
from easyvolcap.utils.console_utils import *


# Camera and Image classes from colmap_utils
@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: List[Tuple[float, float]]
    point3D_ids: List[int]


# rotmat2qvec from colmap_utils
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# write_cameras_text from colmap_utils
def write_cameras_text(cameras, path):
    HEADER = "# Camera list with one line of data per camera:\n"
    "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


# write_images_text from colmap_utils
def write_images_text(images, path):
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items())) / len(images)
    HEADER = "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


# load_ply and export_pts from data_utils
def load_ply(path):
    from plyfile import PlyData

    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    positions = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
    return positions, dotdict(rgb=colors)


def export_pts(path, pts, color=None):
    from plyfile import PlyData, PlyElement

    pts = pts.astype(np.float32)

    if color is None:
        vertex = np.array(
            [(pts[i, 0], pts[i, 1], pts[i, 2]) for i in range(pts.shape[0])],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )
    else:
        color = color.astype(np.uint8)
        vertex = np.array(
            [(pts[i, 0], pts[i, 1], pts[i, 2], color[i, 0], color[i, 1], color[i, 2]) for i in range(pts.shape[0])],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(path)


# read_cama_new from easy_utils
def read_camera_new(data_root):
    # intri_path = join(data_root, "intri.yml")
    # extri_path = join(data_root, "extri.yml")
    intri_path = join(data_root, "intri_new.yml")
    extri_path = join(data_root, "extri_new.yml")

    fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_ext = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

    cams = dotdict()

    # Get camera names from the names node
    names_node = fs.getNode("names")
    if names_node is None or names_node.empty():
        raise ValueError(f"Missing 'names' node in {intri_path}")

    cam_names = []
    for i in range(names_node.size()):
        cam_names.append(names_node.at(i).string())

    for cam in cam_names:
        cams[cam] = dotdict()
        K_node = fs.getNode(f"K_{cam}")
        if K_node is None or K_node.empty():
            raise ValueError(f"Missing K matrix for camera {cam}")
        cams[cam].K = K_node.mat()

        # Get H and W, default to -1 if not found
        H_node = fs.getNode(f"H_{cam}")
        W_node = fs.getNode(f"W_{cam}")
        cams[cam].H = int(H_node.real()) if H_node is not None and H_node.isReal() else -1
        cams[cam].W = int(W_node.real()) if W_node is not None and W_node.isReal() else -1
        cams[cam].invK = np.linalg.inv(cams[cam].K)

        # Get extrinsics
        T_node = fs_ext.getNode(f"T_{cam}")
        R_node = fs_ext.getNode(f"R_{cam}")
        Rot_node = fs_ext.getNode(f"Rot_{cam}")

        if T_node is None or T_node.empty():
            raise ValueError(f"Missing translation vector for camera {cam}")
        Tvec = T_node.mat()

        if R_node is not None and not R_node.empty():
            R = cv2.Rodrigues(R_node.mat())[0]
        elif Rot_node is not None and not Rot_node.empty():
            R = Rot_node.mat()
        else:
            raise ValueError(f"Missing rotation matrix/vector for camera {cam}")

        cams[cam].R = R
        cams[cam].T = Tvec

        # Get distortion parameters if they exist
        dist_node = fs.getNode(f"D_{cam}")
        if dist_node is not None and not dist_node.empty():
            cams[cam].dist = dist_node.mat()

        rdist_node = fs.getNode(f"rdist_{cam}")
        if rdist_node is not None and not rdist_node.empty():
            cams[cam].rdist = rdist_node.mat()

    fs.release()
    fs_ext.release()

    if not cams:
        raise ValueError(f"No cameras found in {intri_path} and {extri_path}")

    return cams

def filter_pcd(xyz_fg, xyz_bg, rgb_fg, rgb_bg, bbox_fit, offset=1.5):
    if bbox_fit is not None:
        fg_mask = np.logical_and.reduce(
            (xyz_fg[:, 0] >= bbox_fit[0, 0], xyz_fg[:, 0] <= bbox_fit[1, 0],
            xyz_fg[:, 1] >= bbox_fit[0, 1], xyz_fg[:, 1] <= bbox_fit[1, 1],
            xyz_fg[:, 2] >= bbox_fit[0, 2], xyz_fg[:, 2] <= bbox_fit[1, 2])
        )
        xyz_fg = xyz_fg[fg_mask]
        rgb_fg.rgb = rgb_fg.rgb[fg_mask]

        fg_min = bbox_fit[0]
        fg_max = bbox_fit[1]
    else:
        fg_min = np.min(xyz_fg, axis=0)
        fg_max = np.max(xyz_fg, axis=0)

    fg_scale = np.max(fg_max - fg_min)
    offset = max(fg_scale * 0.5, offset)
    fg_bbox = np.array([fg_min-offset, fg_max+offset])

    # filter the bg pcds in the fg bbox
    bg_mask = np.logical_and.reduce(
        (xyz_bg[:, 0] >= fg_bbox[0, 0], xyz_bg[:, 0] <= fg_bbox[1, 0],
         xyz_bg[:, 1] >= fg_bbox[0, 1], xyz_bg[:, 1] <= fg_bbox[1, 1],
         xyz_bg[:, 2] >= fg_bbox[0, 2], xyz_bg[:, 2] <= fg_bbox[1, 2])
    )

    xyz_bg = xyz_bg[~bg_mask]
    rgb_bg.rgb = rgb_bg.rgb[~bg_mask]

    xyz = np.concatenate([xyz_fg, xyz_bg], axis=0)
    rgb = np.concatenate([rgb_fg.rgb, rgb_bg.rgb], axis=0)
    return xyz, rgb

def main(data_root, output, camera_model, type, frame_range, lr_res):
    cams = read_camera_new(data_root)
    cam_names = sorted(list(cams.keys()))
    cams = {k: cams[k] for k in cam_names}
    print(f"Camera names: {cam_names}")

    os.makedirs(output, exist_ok=True)

    if type == "orig":
        images_folder = "images"
    elif type == "vsr":
        # images_folder = f"images_sr_from_{lr_res}"
        images_folder = "images"
    else:
        raise ValueError(f"Unknown type: {type}")
    image_ext = ".jpg"

    cameras = {}
    images = {}
    sizes = {}
    frames = os.listdir(join(data_root, images_folder, cam_names[0]))
    frames = sorted([x.split(".")[0] for x in frames])

    b, e, s = frame_range
    frames = frames[b:e:s]

    for frame in tqdm(frames, desc="evc to colmap"):
        output_dir = join(output, frame)
        os.makedirs(join(output_dir, "images"), exist_ok=True)
        os.makedirs(join(output_dir, "sparse/0"), exist_ok=True)
        for cam_id, cam_name in enumerate(cam_names):
            try:
                os.remove(join(output_dir, "images", f"{cam_name}{image_ext}"))
                os.remove(join(output_dir, "masks", f"{cam_name}{args.mask_ext}"))
            except:
                pass

            src = join(data_root, images_folder, cam_name, f"{frame}{image_ext}")
            tar = join(output_dir, "images", f"{cam_name}{image_ext}")
            os.symlink(relpath(src, dirname(tar)), tar)

            # read image
            if cam_name not in sizes.keys():
                img = cv2.imread(join(output_dir, "images", f"{cam_name}{image_ext}"))
                sizes[cam_name] = img.shape[:2]

            cam_dict = cams[cam_name]
            K = cam_dict["K"]
            R = cam_dict["R"]
            T = cam_dict["T"]
            if "H" in cam_dict.keys() or "W" in cam_dict.keys():
                cam_dict["H"] = sizes[cam_name][0]
                cam_dict["W"] = sizes[cam_name][1]

            if camera_model == "PINHOLE":
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]  # fmt: off
                params = [fx, fy, cx, cy]
                camera = Camera(id=cam_id, model="PINHOLE", width=cam_dict["W"], height=cam_dict["H"], params=params)

            elif camera_model == "OPENCV":
                D = cam_dict["dist"]  # !: losing k3 parameter
                if D.shape[0] == 1:
                    fx, fy, cx, cy, k1, k2, p1, p2, _k3 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3], D[0, 4]  # fmt: off
                else:
                    fx, fy, cx, cy, k1, k2, p1, p2, _k3 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0], D[4, 0]  # fmt: off
                params = [fx, fy, cx, cy, k1, k2, p1, p2]
                camera = Camera(id=cam_id, model="OPENCV", width=cam_dict["W"], height=cam_dict["H"], params=params)

            elif camera_model == "RADIAL_FISHEYE":
                D = cam_dict["rdist"]
                assert K[0, 0] == K[1, 1]
                if D.shape[0] == 1:
                    f, cx, cy, k1, k2 = K[0, 0], K[0, 2], K[1, 2], D[0, 0], D[0, 1]
                else:
                    f, cx, cy, k1, k2 = K[0, 0], K[0, 2], K[1, 2], D[0, 0], D[1, 0]

                params = [f, cx, cy, k1, k2]
                camera = Camera(
                    id=cam_id, model="RADIAL_FISHEYE", width=cam_dict["W"], height=cam_dict["H"], params=params
                )

            else:
                raise ValueError(f"Unknown camera model: {camera_model}")

            qvec = rotmat2qvec(R)
            tvec = T.T[0]
            name = f"{cam_name}.jpg"

            image = Image(id=cam_id, qvec=qvec, tvec=tvec, camera_id=cam_id, name=name, xys=[], point3D_ids=[])

            cameras[cam_id] = camera
            images[cam_id] = image

        write_cameras_text(cameras, join(output_dir, "sparse/0", "cameras.txt"))
        write_images_text(images, join(output_dir, "sparse/0", "images.txt"))
        with open(join(output_dir, "sparse/0", "points3D.txt"), "w") as f:
            f.writelines(["# 3D point list with one line of data per point:\n"])

        ## points3D.ply
        pcd_fg = join(data_root, f"pcds_roma_33k/{frame}.ply")
        pcd_bg = join(data_root, f"pcds_roma_full/000200.ply")
        pcd_out = join(output_dir, "sparse/0", "points3D.ply")
        if os.path.exists(pcd_fg):
            xyz1, scalars1 = load_ply(pcd_fg)
            xyz2, scalars2 = load_ply(pcd_bg)
            xyz = np.concatenate([xyz1, xyz2], axis=0)
            rgb = np.concatenate([scalars1.rgb, scalars2.rgb], axis=0)
            export_pts(pcd_out, xyz, color=rgb)
        else:
            print(f"No foreground point cloud found for frame {frame}")

        # pcd_fg = join(data_root, f"pcds_roma_33k/{frame}.ply")
        # pcd_bg = join(data_root, f"pcds_roma_full/002000.ply")
        # pcd_fg_filtered = join(data_root, f"pcds_roma_full/000000_modified.ply")
        # pcd_out = join(output_dir, "sparse/0", "points3D.ply")
        # xyz_fit, scalars_fit = load_ply(pcd_fg_filtered)
        # bbox_fit = np.array([np.min(xyz_fit, axis=0), np.max(xyz_fit, axis=0)])
        # if os.path.exists(pcd_fg):
        #     xyz1, scalars1 = load_ply(pcd_fg)
        #     xyz2, scalars2 = load_ply(pcd_bg)
        #     xyz, rgb = filter_pcd(xyz1, xyz2, scalars1, scalars2, bbox_fit)
        #     export_pts(pcd_out, xyz, color=rgb)
        # else:
        #     print(f"No foreground point cloud found for frame {frame}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="colmap")
    parser.add_argument("--camera_model", type=str, choices=["OPENCV", "PINHOLE"], default="PINHOLE")
    parser.add_argument("--type", "-t", type=str, choices=["orig", "vsr"], default="orig")
    parser.add_argument("--frame_range", nargs=3, type=int, default=[0, 30, 1], help="start, end, step")
    parser.add_argument('--lr_res', type=str, default='1080p', help='Low resolution resolution(1080p/512p/256p/128p)')
    args = parser.parse_args()

    if not isabs(args.output):
        args.output = join(args.data_root, args.output)

    main(args.data_root, args.output, args.camera_model, args.type, args.frame_range, args.lr_res)
