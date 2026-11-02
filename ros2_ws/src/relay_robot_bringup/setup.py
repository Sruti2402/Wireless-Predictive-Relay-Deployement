from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'relay_robot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sruti',
    maintainer_email='sruti@todo.todo',
    description='Predictive Wireless Relay Deployment ROS 2 package',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'telemetry_publisher = relay_robot_bringup.telemetry_publisher:main',
            'telemetry_subscriber = relay_robot_bringup.telemetry_subscriber:main',
            'communication_risk_analyzer = relay_robot_bringup.communication_risk_analyzer:main',
            'relay_deployment_decision = relay_robot_bringup.relay_deployment_decision:main',
            'relay_dashboard = relay_robot_bringup.relay_dashboard:main',
        ],
    },
)