import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json


class TelemetryPublisher(Node):

    def __init__(self):
        super().__init__('telemetry_publisher')

        # Create a publisher
        self.publisher_ = self.create_publisher(
            String,
            '/robot/telemetry',
            10
        )

        # Publish data every second
        self.timer = self.create_timer(
            1.0,
            self.publish_telemetry
        )

        self.time_step = 0

        self.get_logger().info(
            'Robot telemetry publisher started'
        )

    def publish_telemetry(self):

        self.time_step += 1

        # Simulated robot information
        robot_x = self.time_step * 0.1
        robot_y = 0.0

        # Simulated wireless signal strength
        # Simulated wireless signal deterioration
        if self.time_step <= 10:
            rssi = -40.0 - (self.time_step * 1.0)

        elif self.time_step <= 20:
            rssi = -50.0 - ((self.time_step - 10) * 1.5)

        else:
            rssi = -65.0 - ((self.time_step - 20) * 1.0)

        # Simulated time until communication risk
        # Simulated predicted time until communication risk
        if self.time_step <= 10:
            time_to_risk = 30.0 - self.time_step

        elif self.time_step <= 20:
            time_to_risk = 20.0 - (self.time_step - 10)

        else:
            time_to_risk = max(0.0, 10.0 - (self.time_step - 20))
            
        telemetry = {
            'robot_x_m': round(robot_x, 2),
            'robot_y_m': round(robot_y, 2),
            'rssi_dbm': round(rssi, 2),
            'estimated_time_to_risk_s': round(time_to_risk, 2)
        }

        message = String()
        message.data = json.dumps(telemetry)

        self.publisher_.publish(message)

        self.get_logger().info(
            f'Published telemetry: {message.data}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = TelemetryPublisher()

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