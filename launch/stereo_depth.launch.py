from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('left_topic', default_value='/camera/fisheye1/image_raw/rectified'),
        DeclareLaunchArgument('right_topic', default_value='/camera/fisheye2/image_raw/rectified'),
        DeclareLaunchArgument('pointcloud_topic', default_value='/stereo/points'),
        DeclareLaunchArgument('backend', default_value='pytorch'),
        DeclareLaunchArgument('model_dir', default_value='/home/kqu/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth'),
        # DeclareLaunchArgument('model_dir', default_value='/home/kqu/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth'),
        DeclareLaunchArgument('trt_engine_dir', default_value='/home/kqu/Fast-FoundationStereo/output'),
        DeclareLaunchArgument('trt_cfg_file', default_value=''),
        DeclareLaunchArgument('trt_input_height', default_value='0'),
        DeclareLaunchArgument('trt_input_width', default_value='0'),
        DeclareLaunchArgument('intrinsic_file', default_value='/home/kqu/capstone/foundation_stereo_ros/K.txt'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('scale', default_value='1.0'),
        DeclareLaunchArgument('valid_iters', default_value='8'),
        DeclareLaunchArgument('max_disp', default_value='192'),
        DeclareLaunchArgument('hiera', default_value='0'),
        DeclareLaunchArgument('remove_invisible', default_value='1'),
        DeclareLaunchArgument('zfar', default_value='5.0'),
        DeclareLaunchArgument('sync_slop', default_value='0.03'),
        DeclareLaunchArgument('queue_size', default_value='8'),
        Node(
            package='foundation_stereo_ros',
            executable='stereo_depth_node.py',
            name='foundation_stereo_node',
            output='screen',
            parameters=[{
                'left_topic': LaunchConfiguration('left_topic'),
                'right_topic': LaunchConfiguration('right_topic'),
                'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
                'backend': LaunchConfiguration('backend'),
                'model_dir': LaunchConfiguration('model_dir'),
                'trt_engine_dir': LaunchConfiguration('trt_engine_dir'),
                'trt_cfg_file': LaunchConfiguration('trt_cfg_file'),
                'trt_input_height': LaunchConfiguration('trt_input_height'),
                'trt_input_width': LaunchConfiguration('trt_input_width'),
                'intrinsic_file': LaunchConfiguration('intrinsic_file'),
                'device': LaunchConfiguration('device'),
                'scale': LaunchConfiguration('scale'),
                'valid_iters': LaunchConfiguration('valid_iters'),
                'max_disp': LaunchConfiguration('max_disp'),
                'hiera': LaunchConfiguration('hiera'),
                'remove_invisible': LaunchConfiguration('remove_invisible'),
                'zfar': LaunchConfiguration('zfar'),
                'sync_slop': LaunchConfiguration('sync_slop'),
                'queue_size': LaunchConfiguration('queue_size'),
            }],
        )
    ])
