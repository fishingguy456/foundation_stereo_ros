#!/usr/bin/env python3

import os
import sys
import yaml
import logging
from typing import Tuple

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omegaconf import OmegaConf
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField


class FoundationStereoNode(Node):
    def __init__(self) -> None:
        super().__init__('foundation_stereo_node')

        self.declare_parameter('left_topic', '/camera/fisheye1/image_raw/rectified')
        self.declare_parameter('right_topic', '/camera/fisheye2/image_raw/rectified')
        self.declare_parameter('pointcloud_topic', '/stereo/points')
        self.declare_parameter('foundation_stereo_root', '/home/kqu/Fast-FoundationStereo')
        self.declare_parameter('model_dir', '/home/kqu/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth')
        self.declare_parameter('intrinsic_file', '/home/kqu/capstone/foundation_stereo_ros/K.txt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('valid_iters', 8)
        self.declare_parameter('max_disp', 192)
        self.declare_parameter('hiera', 0)
        self.declare_parameter('remove_invisible', 1)
        self.declare_parameter('zfar', 100.0)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('queue_size', 8)

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.foundation_stereo_root = self.get_parameter('foundation_stereo_root').value
        self.model_dir = self.get_parameter('model_dir').value
        self.intrinsic_file = self.get_parameter('intrinsic_file').value
        self.device = self.get_parameter('device').value
        self.scale = float(self.get_parameter('scale').value)
        self.valid_iters = int(self.get_parameter('valid_iters').value)
        self.max_disp = int(self.get_parameter('max_disp').value)
        self.hiera = int(self.get_parameter('hiera').value)
        self.remove_invisible = int(self.get_parameter('remove_invisible').value)
        self.zfar = float(self.get_parameter('zfar').value)
        sync_slop = float(self.get_parameter('sync_slop').value)
        queue_size = int(self.get_parameter('queue_size').value)

        self._add_foundation_stereo_paths()

        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE, set_seed

        self.InputPadder = InputPadder
        self.AMP_DTYPE = AMP_DTYPE
        self.set_seed = set_seed

        self._configure_logging()
        self.set_seed(0)

        self.model_args = self._load_model_args_from_cfg()
        self.model = self._load_model(self.model_dir)

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
            f'Started foundation_stereo_node | left: {left_topic}, right: {right_topic}, pointcloud: {pointcloud_topic}'
        )

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
        )

    def _add_foundation_stereo_paths(self) -> None:
        if not os.path.isdir(self.foundation_stereo_root):
            raise RuntimeError(f'foundation_stereo_root does not exist: {self.foundation_stereo_root}')
        if self.foundation_stereo_root not in sys.path:
            sys.path.append(self.foundation_stereo_root)

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

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        if self.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        K = self.K_base.copy()
        K[:2] *= self.scale

        with np.errstate(divide='ignore', invalid='ignore'):
            depth = K[0, 0] * self.baseline / disp

        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    def _depth_to_pointcloud_msg(self, depth: np.ndarray, header) -> PointCloud2:
        H, W = depth.shape
        K = self.K_base.copy()
        K[:2] *= self.scale

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

    def stereo_callback(self, left_msg: Image, right_msg: Image) -> None:
        try:
            left = self._to_rgb_np(left_msg)
            right = self._to_rgb_np(right_msg)

            if self.scale != 1.0:
                left = cv2.resize(left, fx=self.scale, fy=self.scale, dsize=None)
                right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]))

            H, W = left.shape[:2]
            left_t, right_t, padder = self._prepare_tensors(left, right)
            disp = self._infer_disparity(left_t, right_t, H, W, padder)
            depth = self._disparity_to_depth(disp)

            pointcloud_msg = self._depth_to_pointcloud_msg(depth, left_msg.header)
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
