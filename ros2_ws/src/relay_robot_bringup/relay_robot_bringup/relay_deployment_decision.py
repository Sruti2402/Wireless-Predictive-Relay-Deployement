import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class RelayDeploymentDecision(Node):

    def __init__(self):
        super().__init__('relay_deployment_decision')

        # Subscribe to communication risk results
        self.subscription = self.create_subscription(
            String,
            '/robot/communication_risk',
            self.make_deployment_decision,
            10
        )

        # Publish relay deployment decisions
        self.publisher_ = self.create_publisher(
            String,
            '/robot/relay_decision',
            10
        )

        # Relay deployment parameters (demonstration values)
        self.relay_distance = 10.0
        self.relay_speed = 1.0
        self.setup_time = 5.0
        self.reconnection_time = 3.0
        self.safety_margin = 2.0

        self.get_logger().info(
            'Relay Deployment Decision Node started'
        )

    def make_deployment_decision(self, msg):

        try:
            risk_data = json.loads(msg.data)

            rssi = float(risk_data['rssi_dbm'])
            warning_lead_time = float(
                risk_data['estimated_time_to_risk_s']
            )

            # Calculate relay readiness time
            travel_time = self.relay_distance / self.relay_speed

            ready_time = (
                travel_time
                + self.setup_time
                + self.reconnection_time
                + self.safety_margin
            )

            # Make deployment decision
            if warning_lead_time <= 0:
                decision = 'TOO_LATE'
                recommendation = 'Communication risk window has arrived'

            elif warning_lead_time >= ready_time:
                decision = 'DEPLOYMENT_FEASIBLE'
                recommendation = 'Relay can be deployed within the warning window'

            else:
                decision = 'DEPLOYMENT_AT_RISK'
                recommendation = 'Available warning time is insufficient'

            # Prepare output
            result = {
                'rssi_dbm': rssi,
                'warning_lead_time_s': warning_lead_time,
                'relay_travel_time_s': travel_time,
                'relay_ready_time_s': ready_time,
                'decision': decision,
                'recommendation': recommendation
            }

            output = String()
            output.data = json.dumps(result)

            self.publisher_.publish(output)

            self.get_logger().info(
                f'Decision: {decision} | '
                f'Warning: {warning_lead_time}s | '
                f'Ready time: {ready_time}s'
            )

        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(
                f'Invalid risk data: {error}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = RelayDeploymentDecision()

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