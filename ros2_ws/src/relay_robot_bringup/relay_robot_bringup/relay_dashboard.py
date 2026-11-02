import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DashboardServer(Node):

    def __init__(self):
        super().__init__('relay_dashboard')

        self.latest_data = {
            'robot_x_m': '--',
            'robot_y_m': '--',
            'rssi_dbm': '--',
            'rssi_gradient_dbps': '--',
            'estimated_time_to_risk_s': '--',
            'risk_level': 'WAITING',
            'total_readiness_time_s': 20,
            'decision': 'Waiting for data',
            'recommendation': 'Waiting for robot telemetry'
        }
        self.previous_rssi_dbm = None
        self.previous_rssi_time = None

        self.telemetry_sub = self.create_subscription(
            String,
            '/robot/telemetry',
            self.telemetry_callback,
            10
        )

        self.risk_sub = self.create_subscription(
            String,
            '/robot/communication_risk',
            self.risk_callback,
            10
        )

        self.decision_sub = self.create_subscription(
            String,
            '/robot/relay_decision',
            self.decision_callback,
            10
        )

        self.start_server()

    def telemetry_callback(self, msg):
        try:
            data = json.loads(msg.data)

            current_time = time.monotonic()

            current_rssi = float(data['rssi_dbm'])

            gradient = '--'

            if (
                self.previous_rssi_dbm is not None
                and self.previous_rssi_time is not None
            ):
                elapsed_time = current_time - self.previous_rssi_time

                if elapsed_time > 0:
                    gradient = (
                        current_rssi - self.previous_rssi_dbm
                    ) / elapsed_time

                    gradient = round(gradient, 2)

            self.latest_data.update({
                'robot_x_m': data.get('robot_x_m', '--'),
                'robot_y_m': data.get('robot_y_m', '--'),
                'rssi_dbm': current_rssi,
                'rssi_gradient_dbps': gradient,
                'estimated_time_to_risk_s':
                    data.get('estimated_time_to_risk_s', '--')
            })

            self.previous_rssi_dbm = current_rssi
            self.previous_rssi_time = current_time

        except Exception as error:
            self.get_logger().error(
                f'Telemetry error: {error}'
            )
    def risk_callback(self, msg):
        try:
            data = json.loads(msg.data)

            risk_status = data.get(
                'risk_status',
                data.get(
                    'risk_level',
                    data.get('status', 'WAITING')
                )
            )

            self.latest_data['risk_level'] = risk_status

        except Exception as error:
            self.get_logger().error(
                f'Risk data error: {error}'
            )

            self.latest_data['risk_level'] = msg.data

    def decision_callback(self, msg):
        try:
            data = json.loads(msg.data)

            self.latest_data['decision'] = data.get(
                'decision',
                msg.data
            )

            self.latest_data['recommendation'] = data.get(
                'recommendation',
                data.get('message', 'Decision received')
            )

            self.latest_data['total_readiness_time_s'] = data.get(
                'total_readiness_time_s',
                20
            )

        except Exception:
            self.latest_data['decision'] = msg.data

    def start_server(self):

        dashboard = self

        class RequestHandler(BaseHTTPRequestHandler):

            def do_GET(self):

                if self.path == '/data':
                    response = json.dumps(
                        dashboard.latest_data
                    ).encode()

                    self.send_response(200)
                    self.send_header(
                        'Content-Type',
                        'application/json'
                    )
                    self.send_header(
                        'Access-Control-Allow-Origin',
                        '*'
                    )
                    self.end_headers()
                    self.wfile.write(response)

                elif self.path == '/' or self.path == '/index.html':
                    try:
                        with open(
                            '/home/sruti/relay_ws/src/relay_robot_bringup/dashboard/index.html',
                            'rb'
                        ) as file:
                            page = file.read()

                        self.send_response(200)
                        self.send_header(
                            'Content-Type',
                            'text/html'
                        )
                        self.end_headers()
                        self.wfile.write(page)

                    except FileNotFoundError:
                        self.send_error(
                            404,
                            'index.html not found'
                        )

                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass

        server = HTTPServer(
            ('0.0.0.0', 8080),
            RequestHandler
        )

        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True
        )

        thread.start()

        self.get_logger().info(
            'Dashboard available at http://localhost:8080'
        )


def main(args=None):
    rclpy.init(args=args)

    node = DashboardServer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
