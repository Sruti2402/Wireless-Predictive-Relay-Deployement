import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time


class TelemetrySubscriber(Node):

    def __init__(self):
        super().__init__('telemetry_subscriber')

        self.subscription = self.create_subscription(
            String,
            '/robot/telemetry',
            self.telemetry_callback,
            10
        )

        self.previous_rssi = None
        self.previous_time = None

        self.get_logger().info('Telemetry subscriber started')

    def telemetry_callback(self, msg):

        telemetry = json.loads(msg.data)

        robot_x = telemetry['robot_x_m']
        robot_y = telemetry['robot_y_m']
        rssi = telemetry['rssi_dbm']
        time_to_risk = telemetry['estimated_time_to_risk_s']

        current_time = time.time()

        if self.previous_rssi is not None:

            time_difference = current_time - self.previous_time

            rssi_gradient = (
                rssi - self.previous_rssi
            ) / time_difference

        else:
            rssi_gradient = 0.0

        self.get_logger().info(
            f'Position: ({robot_x}, {robot_y}) m | '
            f'RSSI: {rssi} dBm | '
            f'RSSI Gradient: {rssi_gradient:.2f} dB/s | '
            f'Time to Risk: {time_to_risk} s'
        )

        self.previous_rssi = rssi
        self.previous_time = current_time


def main(args=None):

    rclpy.init(args=args)

    node = TelemetrySubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()