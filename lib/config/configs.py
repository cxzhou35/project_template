import os
import argparse
from .yacs import CfgNode as CN
from lib.utils.cfg_utils import make_cfg

cfg = CN()

# base configs
cfg.workspace = os.environ["PWD"]
cfg.loaded_iter = -1
cfg.ip = "127.0.0.1"
cfg.port = 7081
cfg.data_device = "cuda"
cfg.mode = "train"
cfg.task = "exp_run"  # task folder name
cfg.exp_name = "test"  # experiment folder name
cfg.gpus = [0]  # list of gpus to use
cfg.debug = False
cfg.resume = True  # If set to True, resume training from the last checkpoint.
cfg.to_cuda = False  # higher GPU utilization with larger memory required

cfg.source_path = ""
cfg.model_path = ""
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

# train configs
cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False

# eval configs
cfg.eval = CN()
cfg.eval.skip_train = False
cfg.eval.skip_test = False
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

# optimizer configs
cfg.optim = CN()
# learning rate
cfg.optim.position_lr_init = 0.00016
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
# densification and pruning
cfg.optim.percent_dense = 0.01
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002  # densify_grad_threshold_{bkgd, obj ...}
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1
# loss weight
cfg.optim.lambda_l1 = 1.0
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.0
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.0
cfg.optim.lambda_reg = 0.0
cfg.optim.lambda_depth_lidar = 0.0
cfg.optim.lambda_depth_mono = 0.0
cfg.optim.lambda_normal_mono = 0.0
cfg.optim.lambda_color_correction = 0.0
cfg.optim.lambda_pose_correction = 0.0
cfg.optim.lambda_scale_flatten = 0.0
cfg.optim.lambda_opacity_sparse = 0.0

# model configs
cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1  # fourier spherical harmonics dimension
cfg.model.gaussian.fourier_scale = 1.0
cfg.model.gaussian.flip_prob = 0.0  # symmetry prior for rigid objects, flip gaussians with this probability during training
cfg.model.gaussian.semantic_mode = "logits"

# data configs
cfg.data = CN()
cfg.data.white_background = False  # If set to True, use white background. Should be False when using sky cubemap.
cfg.data.use_colmap_pose = False  # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
cfg.data.filter_colmap = False  # If set to True, filter out SfM points by camera poses.
cfg.data.box_scale = 1.0  # Scale the bounding box by this factor.
cfg.data.split_test = -1
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = "Colmap"
cfg.data.images = "images"
cfg.data.use_semantic = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = True

# render configs
cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = "world"  # ['world', 'vehicle']
cfg.render.concat_cameras = []
cfg.viewer = CN()
cfg.viewer.frame_id = 0  # Select the frame_id (start from 0) to save for viewer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument("--det", type=str, default="")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
cfg = make_cfg(cfg, args)
