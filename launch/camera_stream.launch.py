#!/usr/bin/env python3
# *****************************************************************************
# @file      camera_stream.launch.py
# @brief     Khởi chạy camera_stream_node cho 4 camera Arducam ISX031 trên Jetson.
# *****************************************************************************

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('camera_stream')
    config_file = os.path.join(pkg_dir, 'config', 'camera_stream.yaml')

    # Đảm bảo đường dẫn thư viện OpenCV CUDA /opt/opencv_cuda/lib được nạp
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    opencv_cuda_lib = '/opt/opencv_cuda/lib'
    if opencv_cuda_lib not in ld_path:
        ld_path = f"{opencv_cuda_lib}:{ld_path}" if ld_path else opencv_cuda_lib

    camera_stream_node = Node(
        package='camera_stream',
        executable='camera_stream_node',
        name='camera_stream_node',
        output='screen',
        parameters=[config_file],
        additional_env={'LD_LIBRARY_PATH': ld_path},
    )

    return LaunchDescription([
        camera_stream_node
    ])
