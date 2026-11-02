from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="relay_robot_bringup",
            executable="telemetry_publisher",
            name="telemetry_publisher",
            output="screen",
        ),
        Node(
            package="relay_robot_bringup",
            executable="telemetry_subscriber",
            name="telemetry_subscriber",
            output="screen",
        ),
        Node(
            package="relay_robot_bringup",
            executable="communication_risk_analyzer",
            name="communication_risk_analyzer",
            output="screen",
        ),
        Node(
            package="relay_robot_bringup",
            executable="relay_deployment_decision",
            name="relay_deployment_decision",
            output="screen",
        ),
        Node(
            package="relay_robot_bringup",
            executable="relay_dashboard",
            name="relay_dashboard",
            output="screen",
        ),
    ])
