# foundation_stereo_ros

ROS 2 package that subscribes to rectified stereo camera topics, runs Fast-FoundationStereo inference, and publishes a point cloud.

## Subscribed topics
- `/camera/fisheye1/image_raw/rectified`
- `/camera/fisheye2/image_raw/rectified`

## Published topic
- `/stereo/points` (`sensor_msgs/PointCloud2`, XYZ in meters)

## Build
```bash
cd /home/kqu/capstone
colcon build --symlink-install --packages-select foundation_stereo_ros
```

## Run
```bash
source install/setup.bash
ros2 launch foundation_stereo_ros stereo_depth.launch.py
```

Override model paths if needed:
```bash
ros2 launch foundation_stereo_ros stereo_depth.launch.py \
  foundation_stereo_root:=/path/to/Fast-FoundationStereo \
  model_dir:=/path/to/model_best_bp2_serialize.pth \
  intrinsic_file:=/path/to/K.txt
```

Use TensorRT backend (same TrtRunner path as `run_demo_tensorrt.py`):
```bash
ros2 launch foundation_stereo_ros stereo_depth.launch.py \
  backend:=tensorrt \
  device:=cuda \
  trt_engine_dir:=/path/to/output \
  trt_cfg_file:=/path/to/onnx.yaml
```

Notes:
- `backend:=pytorch` uses `model_dir` and supports `scale`, `valid_iters`, `max_disp`, `hiera`.
- `backend:=tensorrt` uses `trt_engine_dir` with `feature_runner.engine` and `post_runner.engine`.
