import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class CommunicationRiskAnalyzer(Node):

    def __init__(self):
        super().__init__('communication_risk_analyzer')

        self.subscription = self.create_subscription(
            String,
            '/robot/telemetry',
            self.analyze_telemetry,
            10
        )

        self.publisher_ = self.create_publisher(
            String,
            '/robot/communication_risk',
            10
        )

        self.get_logger().info(
            'Communication Risk Analyzer started'
        )

    def analyze_telemetry(self, msg):

        try:
            telemetry = json.loads(msg.data)

            rssi = float(telemetry['rssi_dbm'])
            time_to_risk = float(
                telemetry['estimated_time_to_risk_s']
            )

            # Temporary simulation thresholds
            if rssi > -60:
                risk_status = 'NORMAL'
            elif rssi > -70:
                risk_status = 'WARNING'
            else:
                risk_status = 'CRITICAL'

            risk_result = {
                'rssi_dbm': rssi,
                'estimated_time_to_risk_s': time_to_risk,
                'risk_status': risk_status
            }

            output = String()
            output.data = json.dumps(risk_result)

            self.publisher_.publish(output)

            self.get_logger().info(
                f'Communication Status: {risk_status} | '
                f'RSSI: {rssi} dBm'
            )

        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(
                f'Invalid telemetry data: {error}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = CommunicationRiskAnalyzer()

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