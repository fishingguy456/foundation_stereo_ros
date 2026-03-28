import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('foundation_stereo_ros'),
        'config',
        'stereo_depth.params.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params_file),
        Node(
            package='foundation_stereo_ros',
            executable='stereo_depth_node.py',
            name='foundation_stereo_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        )
    ])
