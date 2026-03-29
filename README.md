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
  params_file:=/path/to/stereo_depth.params.yaml
```

Default parameters are stored in `config/stereo_depth.params.yaml`.

Use TensorRT backend (same TrtRunner path as `run_demo_tensorrt.py`):
```bash
# edit config/stereo_depth.params.yaml, then launch
ros2 launch foundation_stereo_ros stereo_depth.launch.py
```

Notes:
- Set one `model_name` (for example `23-36-37`) and one `models_root` (default `foundation_stereo_ros/models`).
- `backend:=pytorch` resolves model path to `models_root/torch/<model_name>/model_best_bp2_serialize.pth` and uses `valid_iters`, `scale`, `max_disp`, `hiera`.
- `backend:=tensorrt` resolves engine dir from `models_root/trt/onnx_<model_name>_iter*`, picks the highest available iter variant, and reads `valid_iters` from that variant's `onnx.yaml`.
- Input image size defaults to 400x488 (`input_height`, `input_width`) and intrinsics are adjusted once at node startup.
