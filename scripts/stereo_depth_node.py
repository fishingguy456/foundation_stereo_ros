#!/usr/bin/env python3

import os
import sys
import yaml
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omegaconf import OmegaConf
from rclpy.node import Node
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
        self.declare_parameter('backend', 'pytorch')
        self.declare_parameter('model_dir', 'model_best_bp2_serialize.pth')
        self.declare_parameter('trt_engine_dir', '/home/kqu/Fast-FoundationStereo/output')
        self.declare_parameter('trt_cfg_file', '')
        self.declare_parameter('trt_input_height', 0)
        self.declare_parameter('trt_input_width', 0)
        self.declare_parameter('intrinsic_file', '/home/kqu/capstone/foundation_stereo_ros/K.txt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('valid_iters', 8)
        self.declare_parameter('max_disp', 192)
        self.declare_parameter('hiera', 0)
        self.declare_parameter('remove_invisible', 1)
        self.declare_parameter('zfar', 5.0)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('queue_size', 8)

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.backend = str(self.get_parameter('backend').value).strip().lower()
        self.model_dir = os.path.expanduser(self.get_parameter('model_dir').value)
        self.trt_engine_dir = os.path.expanduser(self.get_parameter('trt_engine_dir').value)
        self.trt_cfg_file = os.path.expanduser(self.get_parameter('trt_cfg_file').value)
        self.trt_input_height = int(self.get_parameter('trt_input_height').value)
        self.trt_input_width = int(self.get_parameter('trt_input_width').value)
        self.intrinsic_file = os.path.expanduser(self.get_parameter('intrinsic_file').value)
        self.device = self.get_parameter('device').value
        self.scale = float(self.get_parameter('scale').value)
        self.valid_iters = int(self.get_parameter('valid_iters').value)
        self.max_disp = int(self.get_parameter('max_disp').value)
        self.hiera = int(self.get_parameter('hiera').value)
        self.remove_invisible = int(self.get_parameter('remove_invisible').value)
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

        if self.backend == 'pytorch':
            self.model_args = self._load_model_args_from_cfg()
            self.model = self._load_model(self.model_dir)
        else:
            self.trt_args = self._load_trt_args_from_cfg()
            self.trt_input_size = self._resolve_trt_input_size()
            self.model = self._load_trt_runner()

        self.K_base, self.baseline = self._load_intrinsics(self.intrinsic_file)

        self.bridge = CvBridge()
        self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, 10)

        self.left_sub = Subscriber(self, Image, left_topic)
        self.right_sub = Subscriber(self, Image, right_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=queue_size,
            slop=sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.stereo_callback)

        self.get_logger().info(
            f'Started foundation_stereo_node | backend: {self.backend}, left: {left_topic}, right: {right_topic}, pointcloud: {pointcloud_topic}'
        )
        if self.backend == 'tensorrt':
            self.get_logger().info(
                f'TensorRT input trim size (HxW): {self.trt_input_size[0]}x{self.trt_input_size[1]}'
            )

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
        )

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
        if not cfg_file:
            cfg_file = os.path.join(os.path.dirname(self.trt_engine_dir), 'onnx.yaml')
        if not os.path.isfile(cfg_file):
            raise RuntimeError(f'onnx config not found: {cfg_file}')

        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

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

    def _disparity_to_depth(self, disp: np.ndarray, sx: float, sy: float, crop_x: int, crop_y: int) -> np.ndarray:
        if self.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        K = self.K_base.copy()
        K[0] *= sx
        K[1] *= sy
        K[0, 2] -= float(crop_x)
        K[1, 2] -= float(crop_y)

        with np.errstate(divide='ignore', invalid='ignore'):
            depth = K[0, 0] * self.baseline / disp

        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    def _depth_to_pointcloud_msg(self, depth: np.ndarray, header, sx: float, sy: float, crop_x: int, crop_y: int) -> PointCloud2:
        H, W = depth.shape
        K = self.K_base.copy()
        K[0] *= sx
        K[1] *= sy
        K[0, 2] -= float(crop_x)
        K[1, 2] -= float(crop_y)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        yy, xx = np.indices((H, W), dtype=np.float32)
        z = depth.astype(np.float32)

        valid = (z > 0.0) & np.isfinite(z)
        if self.zfar > 0.0:
            valid &= z <= self.zfar

        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        points = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)

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
        if self.scale != 1.0:
            left = cv2.resize(left, fx=self.scale, fy=self.scale, dsize=None)
            right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]))

        H, W = left.shape[:2]
        left_t, right_t, padder = self._prepare_tensors(left, right)
        disp = self._infer_disparity(left_t, right_t, H, W, padder)
        return disp, self.scale, self.scale, 0, 0

    def _center_trim_stereo(self, left: np.ndarray, right: np.ndarray, out_h: int, out_w: int):
        in_h, in_w = left.shape[:2]
        if right.shape[:2] != (in_h, in_w):
            raise RuntimeError('Left/right image size mismatch before TensorRT trim')
        if in_h < out_h or in_w < out_w:
            raise RuntimeError(
                f'Input image {in_h}x{in_w} is smaller than TensorRT trim size {out_h}x{out_w}. '
                'Use smaller trt_input_height/trt_input_width or rebuild engines for this input size.'
            )

        y0 = (in_h - out_h) // 2
        x0 = (in_w - out_w) // 2
        y1 = y0 + out_h
        x1 = x0 + out_w
        return left[y0:y1, x0:x1], right[y0:y1, x0:x1], x0, y0

    def _infer_disparity_tensorrt(self, left: np.ndarray, right: np.ndarray):
        out_h, out_w = self.trt_input_size
        left, right, crop_x, crop_y = self._center_trim_stereo(left, right, out_h, out_w)

        left_t = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_t = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)

        disp = self.model.forward(left_t, right_t)
        disp = disp.data.cpu().numpy().reshape(out_h, out_w).clip(0, None)
        return disp, 1.0, 1.0, crop_x, crop_y

    def stereo_callback(self, left_msg: Image, right_msg: Image) -> None:
        try:
            left = self._to_rgb_np(left_msg)
            right = self._to_rgb_np(right_msg)

            if self.backend == 'pytorch':
                disp, sx, sy, crop_x, crop_y = self._infer_disparity_pytorch(left, right)
            else:
                disp, sx, sy, crop_x, crop_y = self._infer_disparity_tensorrt(left, right)

            depth = self._disparity_to_depth(disp, sx, sy, crop_x, crop_y)
            pointcloud_msg = self._depth_to_pointcloud_msg(depth, left_msg.header, sx, sy, crop_x, crop_y)
            self.pointcloud_pub.publish(pointcloud_msg)

        except Exception as exc:
            self.get_logger().error(f'stereo callback failed: {exc}')


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
