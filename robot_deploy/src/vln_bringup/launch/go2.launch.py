"""Launch the complete VLN stack for Unitree Go2."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    orbbec_launch = (
        Path(get_package_share_directory("orbbec_camera"))
        / "launch"
        / "gemini_330_series.launch.py"
    )
    network_interface = LaunchConfiguration("network_interface")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "network_interface",
                default_value="enP8p1s0",
                description="Network interface connected to the Go2",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(orbbec_launch)),
                launch_arguments={
                    "connection_delay": "1",
                    "enable_color": "true",
                    "color_width": "640",
                    "color_height": "360",
                    "color_fps": "30",
                    "color_format": "RGB",
                    "enable_depth": "false",
                    "enable_left_ir": "false",
                    "enable_right_ir": "false",
                    "enable_accel": "false",
                    "enable_gyro": "false",
                    "enable_point_cloud": "false",
                    "enable_colored_point_cloud": "false",
                    "enable_laser": "false",
                    "enable_ldp": "false",
                    "publish_tf": "false",
                    "enable_publish_extrinsic": "false",
                    "enumerate_net_device": "false",
                }.items(),
            ),
            Node(
                package="vln_client",
                executable="vln_client",
                name="vln_client",
                output="screen",
            ),
            Node(
                package="vln_web",
                executable="vln_web",
                name="vln_web",
                output="screen",
                parameters=[
                    {
                        "manual_linear_limit": 3.0,
                        "manual_angular_limit": 3.0,
                        "manual_linear_accel": 10.0,
                        "manual_angular_accel": 10.0,
                    }
                ],
            ),
            Node(
                package="vln_mpc",
                executable="vln_mpc",
                name="vln_mpc",
                output="screen",
                parameters=[
                    {
                        "track_v_max": 1.5,
                        "objnav_v_max": 0.8,
                        "w_max": 3.0,
                        "a_max_v": 10.0,
                        "a_max_w": 10.0,
                        "q_x": 10.0,
                        "q_y": 10.0,
                        "q_yaw": 1.0,
                        "r_v": 0.1,
                        "r_w": 0.1,
                        "v_output_scale": 1.0,
                        "w_output_scale": 1.0,
                    }
                ],
            ),
            Node(
                package="go2_adapter",
                executable="go2_adapter",
                name="go2_adapter",
                output="screen",
                parameters=[{"network_interface": network_interface}],
            ),
        ]
    )
