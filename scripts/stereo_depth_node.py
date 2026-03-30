#!/usr/bin/env python3

import os
import sys
import yaml
import logging
import glob
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omegaconf import OmegaConf
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField


def _ensure_local_module_paths() -> None:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, '..')),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_local_module_paths()


class FoundationStereoNode(Node):
    def __init__(self) -> None:
        super().__init__('foundation_stereo_node')

        self.declare_parameter('left_topic', '/camera/fisheye1/image_raw/rectified')
        self.declare_parameter('right_topic', '/camera/fisheye2/image_raw/rectified')
        self.declare_parameter('pointcloud_topic', '/stereo/points')
        self.declare_parameter('depth_topic', '/stereo/depth')
        self.declare_parameter('backend', 'pytorch')
        self.declare_parameter('input_source', 'ros')
        self.declare_parameter('zmq_host', 'localhost')
        self.declare_parameter('zmq_port', 5555)
        self.declare_parameter('zmq_rcv_hwm', 2)
        self.declare_parameter('zmq_poll_period_sec', 0.001)
        self.declare_parameter('output_frame_id', 'camera_left')
        self.declare_parameter('model_name', '23-36-37')
        self.declare_parameter('models_root', '/home/kqu/capstone/foundation_stereo_ros/models')
        self.declare_parameter('trt_input_height', 0)
        self.declare_parameter('trt_input_width', 0)
        self.declare_parameter('intrinsic_file', '/home/kqu/capstone/foundation_stereo_ros/K.txt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('input_height', 400)
        self.declare_parameter('input_width', 488)
        self.declare_parameter('valid_iters', 8)
        self.declare_parameter('max_disp', 192)
        self.declare_parameter('hiera', 0)
        self.declare_parameter('get_pc', 1)
        self.declare_parameter('remove_invisible', 1)
        self.declare_parameter('denoise_cloud', 0)
        self.declare_parameter('denoise_nb_points', 30)
        self.declare_parameter('denoise_radius', 0.03)
        self.declare_parameter('denoise_voxel_size', 0.0)
        self.declare_parameter('zfar', 5.0)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('queue_size', 8)

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.backend = str(self.get_parameter('backend').value).strip().lower()
        self.input_source = str(self.get_parameter('input_source').value).strip().lower()
        self.zmq_host = self._resolve_zmq_host(str(self.get_parameter('zmq_host').value).strip())
        self.zmq_port = int(self.get_parameter('zmq_port').value)
        self.zmq_rcv_hwm = int(self.get_parameter('zmq_rcv_hwm').value)
        self.zmq_poll_period_sec = float(self.get_parameter('zmq_poll_period_sec').value)
        self.output_frame_id = str(self.get_parameter('output_frame_id').value).strip()
        self.model_name = str(self.get_parameter('model_name').value).strip()
        self.models_root = os.path.expanduser(self.get_parameter('models_root').value)
        self.trt_input_height = int(self.get_parameter('trt_input_height').value)
        self.trt_input_width = int(self.get_parameter('trt_input_width').value)
        self.intrinsic_file = os.path.expanduser(self.get_parameter('intrinsic_file').value)
        self.device = self.get_parameter('device').value
        self.scale = float(self.get_parameter('scale').value)
        self.input_height = int(self.get_parameter('input_height').value)
        self.input_width = int(self.get_parameter('input_width').value)
        self.requested_valid_iters = int(self.get_parameter('valid_iters').value)
        self.valid_iters = self.requested_valid_iters
        self.max_disp = int(self.get_parameter('max_disp').value)
        self.hiera = int(self.get_parameter('hiera').value)
        self.get_pc = int(self.get_parameter('get_pc').value)
        self.remove_invisible = int(self.get_parameter('remove_invisible').value)
        self.denoise_cloud = int(self.get_parameter('denoise_cloud').value)
        self.denoise_nb_points = int(self.get_parameter('denoise_nb_points').value)
        self.denoise_radius = float(self.get_parameter('denoise_radius').value)
        self.denoise_voxel_size = float(self.get_parameter('denoise_voxel_size').value)
        self.zfar = float(self.get_parameter('zfar').value)
        sync_slop = float(self.get_parameter('sync_slop').value)
        queue_size = int(self.get_parameter('queue_size').value)

        from Utils import set_seed
        self.set_seed = set_seed

        if self.backend == 'pytorch':
            from core.utils.utils import InputPadder
            from Utils import AMP_DTYPE
            self.InputPadder = InputPadder
            self.AMP_DTYPE = AMP_DTYPE
        elif self.backend == 'tensorrt':
            from core.foundation_stereo import TrtRunner
            self.TrtRunner = TrtRunner
        else:
            raise RuntimeError(f"Unsupported backend '{self.backend}'. Use 'pytorch' or 'tensorrt'.")

        self._configure_logging()
        self.set_seed(0)
        self._resolve_model_paths()

        if self.backend == 'pytorch':
            self.model_args = self._load_model_args_from_cfg()
            self.model = self._load_model(self.model_dir)
        else:
            self.trt_args = self._load_trt_args_from_cfg()
            self.trt_input_size = self._resolve_trt_input_size()
            self.model = self._load_trt_runner()

        self.K_base, self.baseline = self._load_intrinsics(self.intrinsic_file)
        self._init_projection_params()
        self._frame_count = 0

        self.o3d = None
        if self.denoise_cloud or self.denoise_voxel_size > 0.0:
            try:
                import open3d as o3d  # type: ignore

                self.o3d = o3d
            except Exception as exc:
                raise RuntimeError(
                    'Point cloud filtering is enabled, but open3d import failed. '
                    'Install open3d or set denoise_cloud:=0 and denoise_voxel_size:=0.0.'
                ) from exc

        self.bridge = CvBridge()
        self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, 10)
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)

        self._zmq_ctx = None
        self._zmq_sock = None
        self._zmq_timer = None
        self._zmq_mod: Optional[Any] = None
        self._msgpack_mod: Optional[Any] = None

        if self.input_source == 'ros':
            self.left_sub = Subscriber(self, Image, left_topic)
            self.right_sub = Subscriber(self, Image, right_topic)
            self.sync = ApproximateTimeSynchronizer(
                [self.left_sub, self.right_sub],
                queue_size=queue_size,
                slop=sync_slop,
                allow_headerless=False,
            )
            self.sync.registerCallback(self.stereo_callback)
        elif self.input_source == 'zmq':
            if self.zmq_poll_period_sec <= 0.0:
                raise RuntimeError('zmq_poll_period_sec must be > 0')

            try:
                import zmq as zmq_mod  # type: ignore
                import msgpack as msgpack_mod  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    'input_source:=zmq requires python packages pyzmq and msgpack. '
                    'Install them in the active environment.'
                ) from exc

            self._zmq_mod = zmq_mod
            self._msgpack_mod = msgpack_mod

            self._zmq_ctx = self._zmq_mod.Context()
            self._zmq_sock = self._zmq_ctx.socket(self._zmq_mod.PULL)
            self._zmq_sock.setsockopt(self._zmq_mod.RCVHWM, self.zmq_rcv_hwm)
            self._zmq_sock.setsockopt(self._zmq_mod.LINGER, 0)
            self._zmq_sock.connect(f'tcp://{self.zmq_host}:{self.zmq_port}')
            self._zmq_timer = self.create_timer(self.zmq_poll_period_sec, self.zmq_callback)
        else:
            raise RuntimeError(f"Unsupported input_source '{self.input_source}'. Use 'ros' or 'zmq'.")

        self.get_logger().info(
            f'Started foundation_stereo_node | backend: {self.backend}, source: {self.input_source}, left: {left_topic}, right: {right_topic}, '
            f'pointcloud: {pointcloud_topic}, depth: {depth_topic}, model_name: {self.model_name}, valid_iters: {self.valid_iters}'
        )
        if self.input_source == 'zmq':
            self.get_logger().info(
                f'ZMQ input connected to tcp://{self.zmq_host}:{self.zmq_port} '
                f'| rcv_hwm: {self.zmq_rcv_hwm}, poll_period: {self.zmq_poll_period_sec}s'
            )
        if self.backend == 'tensorrt':
            self.get_logger().info(
                f'TensorRT model dir: {self.trt_engine_dir}, input trim size (HxW): {self.trt_input_size[0]}x{self.trt_input_size[1]}'
            )
        else:
            self.get_logger().info(f'PyTorch model path: {self.model_dir}')
        self.get_logger().info(
            f'Projection setup | input: {self.input_height}x{self.input_width}, model input: {self.model_input_size[0]}x{self.model_input_size[1]}, '
            f'scale: ({self.scale_x:.4f}, {self.scale_y:.4f}), crop: ({self.crop_x}, {self.crop_y})'
        )
        self.get_logger().info(
            f'Point cloud options | get_pc: {self.get_pc}, remove_invisible: {self.remove_invisible}, '
            f'denoise_cloud: {self.denoise_cloud}, '
            f'denoise_nb_points: {self.denoise_nb_points}, '
            f'denoise_radius: {self.denoise_radius}, denoise_voxel_size: {self.denoise_voxel_size}, zfar: {self.zfar}'
        )

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
        )

    def _resolve_zmq_host(self, host_value: str) -> str:
        expanded_host = os.path.expandvars(host_value).strip()
        if not expanded_host:
            raise RuntimeError('zmq_host resolved to an empty value')

        unresolved_patterns = ('$' in expanded_host) or ('${' in expanded_host)
        if unresolved_patterns:
            raise RuntimeError(
                f"zmq_host contains unresolved environment variable syntax: '{host_value}'. "
                'Set the variable before launch or provide a concrete host/IP.'
            )

        return expanded_host

    def _resolve_model_paths(self) -> None:
        if not self.model_name:
            raise RuntimeError('model_name must be non-empty')

        if self.backend == 'pytorch':
            model_dir = os.path.join(self.models_root, 'torch', self.model_name)
            model_path = os.path.join(model_dir, 'model_best_bp2_serialize.pth')
            if not os.path.isfile(model_path):
                raise RuntimeError(
                    f'PyTorch model not found for model_name={self.model_name}. Expected: {model_path}'
                )
            self.model_dir = model_path
            return

        trt_dir = os.path.join(
            self.models_root,
            'trt',
            f'{self.model_name}_iter{self.requested_valid_iters}',
        )
        if not os.path.isdir(trt_dir):
            trt_glob = os.path.join(self.models_root, 'trt', f'{self.model_name}_iter*')
            available_trt_dirs = sorted([path for path in glob.glob(trt_glob) if os.path.isdir(path)])
            available_text = ', '.join(available_trt_dirs) if available_trt_dirs else 'none'
            raise RuntimeError(
                f'TensorRT model folder not found for model_name={self.model_name}, '
                f'valid_iters={self.requested_valid_iters}. Expected: {trt_dir}. '
                f'Available variants: {available_text}'
            )

        feature_engine = os.path.join(trt_dir, 'feature_runner.engine')
        post_engine = os.path.join(trt_dir, 'post_runner.engine')
        cfg_file = os.path.join(trt_dir, 'onnx.yaml')
        if not (os.path.isfile(feature_engine) and os.path.isfile(post_engine) and os.path.isfile(cfg_file)):
            raise RuntimeError(
                f'TensorRT folder is incomplete: {trt_dir}. Required files: '
                'feature_runner.engine, post_runner.engine, onnx.yaml.'
            )

        self.trt_engine_dir = trt_dir
        self.trt_cfg_file = cfg_file

    def _load_model_args_from_cfg(self):
        cfg_file = os.path.join(os.path.dirname(self.model_dir), 'cfg.yaml')
        if not os.path.isfile(cfg_file):
            raise RuntimeError(f'cfg.yaml not found next to model: {cfg_file}')

        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        cfg_dict['valid_iters'] = self.valid_iters
        cfg_dict['max_disp'] = self.max_disp
        cfg_dict['hiera'] = self.hiera
        cfg_dict['scale'] = self.scale
        return OmegaConf.create(cfg_dict)

    def _load_model(self, model_path: str):
        if not os.path.isfile(model_path):
            raise RuntimeError(f'model file not found: {model_path}')

        if self.device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but unavailable; falling back to CPU.')
            self.device = 'cpu'

        map_location = 'cpu' if self.device == 'cpu' else None
        model = torch.load(model_path, map_location=map_location, weights_only=False)
        model.args.valid_iters = self.valid_iters
        model.args.max_disp = self.max_disp

        if self.device == 'cuda':
            model = model.cuda()
        else:
            model = model.cpu()
        model.eval()

        return model

    def _load_trt_args_from_cfg(self):
        cfg_file = self.trt_cfg_file
        if not os.path.isfile(cfg_file):
            raise RuntimeError(f'onnx config not found: {cfg_file}')

        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        if 'valid_iters' not in cfg_dict:
            raise RuntimeError(f"onnx config missing required key 'valid_iters': {cfg_file}")

        trt_valid_iters = int(cfg_dict['valid_iters'])
        if trt_valid_iters <= 0:
            raise RuntimeError(
                f"onnx config has invalid 'valid_iters'={trt_valid_iters}; expected positive integer: {cfg_file}"
            )

        if trt_valid_iters != self.requested_valid_iters:
            raise RuntimeError(
                f"TensorRT model config mismatch for {self.trt_engine_dir}: requested valid_iters="
                f"{self.requested_valid_iters}, but onnx.yaml has valid_iters={trt_valid_iters}."
            )

        self.valid_iters = trt_valid_iters

        cfg_dict['onnx_dir'] = self.trt_engine_dir
        return OmegaConf.create(cfg_dict)

    def _read_trt_input_size_from_onnx(self, onnx_file: str) -> Optional[Tuple[int, int]]:
        if not os.path.isfile(onnx_file):
            return None

        try:
            import onnx  # type: ignore
        except Exception:
            return None

        try:
            model = onnx.load(onnx_file)
            graph_inputs = list(model.graph.input)
            if not graph_inputs:
                return None

            target = None
            for inp in graph_inputs:
                if inp.name == 'left':
                    target = inp
                    break
            if target is None:
                target = graph_inputs[0]

            dims = target.type.tensor_type.shape.dim
            if len(dims) != 4:
                return None

            h_dim, w_dim = dims[2], dims[3]
            if not h_dim.HasField('dim_value') or not w_dim.HasField('dim_value'):
                return None

            out_h = int(h_dim.dim_value)
            out_w = int(w_dim.dim_value)
            if out_h <= 0 or out_w <= 0:
                return None
            return out_h, out_w
        except Exception:
            return None

    def _resolve_trt_input_size(self) -> Tuple[int, int]:
        if self.trt_input_height > 0 and self.trt_input_width > 0:
            return self.trt_input_height, self.trt_input_width

        if 'image_size' in self.trt_args and len(self.trt_args.image_size) == 2:
            out_h = int(self.trt_args.image_size[0])
            out_w = int(self.trt_args.image_size[1])
            if out_h > 0 and out_w > 0:
                return out_h, out_w

        if 'height' in self.trt_args and 'width' in self.trt_args:
            out_h = int(self.trt_args.height)
            out_w = int(self.trt_args.width)
            if out_h > 0 and out_w > 0:
                return out_h, out_w

        onnx_size = self._read_trt_input_size_from_onnx(
            os.path.join(self.trt_engine_dir, 'feature_runner.onnx')
        )
        if onnx_size is not None:
            return onnx_size

        raise RuntimeError(
            'Unable to determine TensorRT static input size. Set trt_input_height and trt_input_width '
            'or provide image_size/height/width in onnx.yaml.'
        )

    def _load_trt_runner(self):
        if self.device != 'cuda':
            raise RuntimeError('TensorRT backend requires device:=cuda')
        if not torch.cuda.is_available():
            raise RuntimeError('TensorRT backend requested but CUDA is unavailable')

        feature_engine = os.path.join(self.trt_engine_dir, 'feature_runner.engine')
        post_engine = os.path.join(self.trt_engine_dir, 'post_runner.engine')
        if not os.path.isfile(feature_engine):
            raise RuntimeError(f'Missing TensorRT engine: {feature_engine}')
        if not os.path.isfile(post_engine):
            raise RuntimeError(f'Missing TensorRT engine: {post_engine}')

        return self.TrtRunner(self.trt_args, feature_engine, post_engine)

    def _load_intrinsics(self, intrinsic_file: str) -> Tuple[np.ndarray, float]:
        if not os.path.isfile(intrinsic_file):
            raise RuntimeError(f'intrinsic file not found: {intrinsic_file}')

        with open(intrinsic_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
        baseline = float(lines[1])
        return K, baseline

    def _init_projection_params(self) -> None:
        if self.input_height <= 0 or self.input_width <= 0:
            raise RuntimeError('input_height and input_width must be positive')

        in_h = self.input_height
        in_w = self.input_width

        if self.backend == 'pytorch':
            out_h = int(round(in_h * self.scale))
            out_w = int(round(in_w * self.scale))
            if out_h <= 0 or out_w <= 0:
                raise RuntimeError(
                    f'Invalid scale {self.scale} for input size {in_h}x{in_w}; got output {out_h}x{out_w}'
                )
            self.scale_x = out_w / float(in_w)
            self.scale_y = out_h / float(in_h)
            self.crop_x = 0
            self.crop_y = 0
            self.model_input_size = (out_h, out_w)
            self._pytorch_resize_dsize = (out_w, out_h)
        else:
            out_h, out_w = self.trt_input_size
            if in_h < out_h or in_w < out_w:
                raise RuntimeError(
                    f'Input image {in_h}x{in_w} is smaller than TensorRT trim size {out_h}x{out_w}. '
                    'Use smaller trt_input_height/trt_input_width or rebuild engines for this input size.'
                )
            self.scale_x = 1.0
            self.scale_y = 1.0
            self.crop_y = (in_h - out_h) // 2
            self.crop_x = (in_w - out_w) // 2
            self.model_input_size = (out_h, out_w)

        self.K = self.K_base.copy()
        self.K[0] *= self.scale_x
        self.K[1] *= self.scale_y
        self.K[0, 2] -= float(self.crop_x)
        self.K[1, 2] -= float(self.crop_y)

        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])

        out_h, out_w = self.model_input_size
        yy, xx = np.indices((out_h, out_w), dtype=np.float32)
        self._ray_x_flat = ((xx - self.cx) / self.fx).reshape(-1)
        self._ray_y_flat = ((yy - self.cy) / self.fy).reshape(-1)

    def _to_rgb_np(self, msg: Image) -> np.ndarray:
        if msg.encoding == 'mono8':
            mono = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _prepare_tensors(self, left: np.ndarray, right: np.ndarray):
        left_t = torch.as_tensor(left).float()[None].permute(0, 3, 1, 2)
        right_t = torch.as_tensor(right).float()[None].permute(0, 3, 1, 2)

        if self.device == 'cuda':
            left_t = left_t.cuda(non_blocking=True)
            right_t = right_t.cuda(non_blocking=True)

        padder = self.InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)
        return left_t, right_t, padder

    def _infer_disparity(self, left_t: torch.Tensor, right_t: torch.Tensor, H: int, W: int, padder) -> np.ndarray:
        use_amp = self.device == 'cuda'
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=self.AMP_DTYPE):
            if not self.hiera:
                disp = self.model.forward(
                    left_t,
                    right_t,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume='pytorch1',
                )
            else:
                disp = self.model.run_hierachical(
                    left_t,
                    right_t,
                    iters=self.valid_iters,
                    test_mode=True,
                    small_ratio=0.5,
                )

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
        return disp

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        if self.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        with np.errstate(divide='ignore', invalid='ignore'):
            depth = self.fx * self.baseline / disp

        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    def _depth_to_points(self, depth: np.ndarray) -> np.ndarray:
        z = depth.astype(np.float32, copy=False)
        valid = (z > 0.0) & np.isfinite(z)
        if self.zfar > 0.0:
            valid &= z <= self.zfar

        valid_flat = valid.reshape(-1)
        if not np.any(valid_flat):
            return np.empty((0, 3), dtype=np.float32)

        z_valid = z.reshape(-1)[valid_flat]
        points = np.empty((z_valid.shape[0], 3), dtype=np.float32)
        points[:, 0] = self._ray_x_flat[valid_flat] * z_valid
        points[:, 1] = self._ray_y_flat[valid_flat] * z_valid
        points[:, 2] = z_valid
        return points

    def _denoise_points(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points

        if not self.denoise_cloud and self.denoise_voxel_size <= 0.0:
            return points

        assert self.o3d is not None
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))

        if self.denoise_voxel_size > 0.0:
            pcd = pcd.voxel_down_sample(voxel_size=self.denoise_voxel_size)
            if len(pcd.points) == 0:
                return np.empty((0, 3), dtype=np.float32)

        if not self.denoise_cloud:
            return np.asarray(pcd.points, dtype=np.float32)

        _, inlier_ids = pcd.remove_radius_outlier(
            nb_points=self.denoise_nb_points,
            radius=self.denoise_radius,
        )
        if len(inlier_ids) == 0:
            return np.empty((0, 3), dtype=np.float32)

        pcd = pcd.select_by_index(inlier_ids)
        return np.asarray(pcd.points, dtype=np.float32)

    def _points_to_pointcloud_msg(self, points: np.ndarray, header) -> PointCloud2:

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = points.tobytes()
        return msg

    def _infer_disparity_pytorch(self, left: np.ndarray, right: np.ndarray):
        if (left.shape[0], left.shape[1]) != self.model_input_size:
            left = cv2.resize(left, dsize=self._pytorch_resize_dsize)
            right = cv2.resize(right, dsize=self._pytorch_resize_dsize)

        H, W = left.shape[:2]
        left_t, right_t, padder = self._prepare_tensors(left, right)
        disp = self._infer_disparity(left_t, right_t, H, W, padder)
        return disp

    def _infer_disparity_tensorrt(self, left: np.ndarray, right: np.ndarray):
        out_h, out_w = self.model_input_size
        y0 = self.crop_y
        x0 = self.crop_x
        y1 = y0 + out_h
        x1 = x0 + out_w
        left = left[y0:y1, x0:x1]
        right = right[y0:y1, x0:x1]

        left_t = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_t = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)

        disp = self.model.forward(left_t, right_t)
        disp = disp.data.cpu().numpy().reshape(out_h, out_w).clip(0, None)
        return disp

    def stereo_callback(self, left_msg: Image, right_msg: Image) -> None:
        try:
            t0 = time.perf_counter()
            left = self._to_rgb_np(left_msg)
            right = self._to_rgb_np(right_msg)
            if left.shape[:2] != (self.input_height, self.input_width):
                raise RuntimeError(
                    f'Unexpected left image size {left.shape[0]}x{left.shape[1]}; expected {self.input_height}x{self.input_width}'
                )
            if right.shape[:2] != (self.input_height, self.input_width):
                raise RuntimeError(
                    f'Unexpected right image size {right.shape[0]}x{right.shape[1]}; expected {self.input_height}x{self.input_width}'
                )
            t_to_rgb_ms = (time.perf_counter() - t0) * 1000.0

            self._process_stereo_pair(
                left=left,
                right=right,
                header=left_msg.header,
                source='ros',
                t_source_decode_ms=t_to_rgb_ms,
            )

        except Exception as exc:
            self.get_logger().error(f'stereo callback failed: {exc}')

    def _decode_jpeg_to_rgb(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError('JPEG decode failed')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _recv_latest_zmq_frame(self):
        if self._zmq_sock is None:
            return None
        if self._zmq_mod is None:
            return None

        latest_raw = None
        while True:
            try:
                latest_raw = self._zmq_sock.recv(flags=self._zmq_mod.NOBLOCK)
            except self._zmq_mod.Again:
                break

        return latest_raw

    def zmq_callback(self) -> None:
        try:
            cb_t0 = time.perf_counter()
            raw = self._recv_latest_zmq_frame()
            t_recv_ms = (time.perf_counter() - cb_t0) * 1000.0
            if raw is None:
                return
            if self._msgpack_mod is None:
                return

            t0 = time.perf_counter()
            frame = self._msgpack_mod.unpackb(raw, raw=False)
            t_unpack_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            left = self._decode_jpeg_to_rgb(frame['left'])
            right = self._decode_jpeg_to_rgb(frame['right'])
            t_decode_ms = (time.perf_counter() - t0) * 1000.0

            if left.shape[:2] != (self.input_height, self.input_width):
                raise RuntimeError(
                    f'Unexpected left image size {left.shape[0]}x{left.shape[1]}; expected {self.input_height}x{self.input_width}'
                )
            if right.shape[:2] != (self.input_height, self.input_width):
                raise RuntimeError(
                    f'Unexpected right image size {right.shape[0]}x{right.shape[1]}; expected {self.input_height}x{self.input_width}'
                )

            if 'ts' in frame and isinstance(frame['ts'], (list, tuple)) and len(frame['ts']) == 2:
                sec = int(frame['ts'][0])
                nanosec = int(frame['ts'][1])
                stamp = Time(seconds=sec, nanoseconds=nanosec).to_msg()
            else:
                stamp = self.get_clock().now().to_msg()

            header = Image().header
            header.stamp = stamp
            header.frame_id = self.output_frame_id

            self._process_stereo_pair(
                left=left,
                right=right,
                header=header,
                source='zmq',
                t_source_decode_ms=t_recv_ms + t_unpack_ms + t_decode_ms,
            )

        except Exception as exc:
            self.get_logger().error(f'zmq callback failed: {exc}')

    def _process_stereo_pair(self, left: np.ndarray, right: np.ndarray, header, source: str, t_source_decode_ms: float) -> None:
        cb_t0 = time.perf_counter()

        t0 = time.perf_counter()
        if self.backend == 'pytorch':
            disp = self._infer_disparity_pytorch(left, right)
        else:
            disp = self._infer_disparity_tensorrt(left, right)
        t_infer_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        depth = self._disparity_to_depth(disp)
        t_disp_to_depth_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)
        t_depth_pub_ms = (time.perf_counter() - t0) * 1000.0

        t_points_ms = 0.0
        t_denoise_ms = 0.0
        t_pc_pub_ms = 0.0
        num_points = 0

        if self.get_pc:
            t0 = time.perf_counter()
            points = self._depth_to_points(depth)
            t_points_ms = (time.perf_counter() - t0) * 1000.0

            if self.denoise_cloud or self.denoise_voxel_size > 0.0:
                t0 = time.perf_counter()
                points = self._denoise_points(points)
                t_denoise_ms = (time.perf_counter() - t0) * 1000.0

            num_points = int(points.shape[0])
            t0 = time.perf_counter()
            pointcloud_msg = self._points_to_pointcloud_msg(points, header)
            self.pointcloud_pub.publish(pointcloud_msg)
            t_pc_pub_ms = (time.perf_counter() - t0) * 1000.0

        cb_total_ms = (time.perf_counter() - cb_t0) * 1000.0
        self._frame_count += 1
        self.get_logger().info(
            f'Timing frame={self._frame_count} source={source} | src_decode={t_source_decode_ms:.2f} ms, '
            f'infer={t_infer_ms:.2f} ms, disp_to_depth={t_disp_to_depth_ms:.2f} ms, depth_pub={t_depth_pub_ms:.2f} ms, '
            f'pc_points={t_points_ms:.2f} ms, pc_denoise={t_denoise_ms:.2f} ms, pc_pub={t_pc_pub_ms:.2f} ms, '
            f'pc_count={num_points}, total={cb_total_ms:.2f} ms'
        )

    def destroy_node(self):
        if self._zmq_sock is not None:
            self._zmq_sock.close()
            self._zmq_sock = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
        return super().destroy_node()


def main(args=None):
    torch.autograd.set_grad_enabled(False)
    rclpy.init(args=args)

    node = FoundationStereoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
