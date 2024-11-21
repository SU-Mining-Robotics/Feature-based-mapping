import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import numpy as np
import math

class NoisyOdometryNode(Node):
    def __init__(self):
        super().__init__('noisy_odometry_node')

        # Define motion dispersion
        self.MOTION_DISPERSION_X = 0.05
        self.MOTION_DISPERSION_Y = 0.025
        self.MOTION_DISPERSION_THETA = 0.1

        # Publisher
        self.noisy_odom_pub = self.create_publisher(Odometry, '/ego_racecar/noisy_odom', 10)

        # Subscriber
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

    def odom_callback(self, msg: Odometry):
        '''
        Callback to receive odometry data, add noise, and publish the noisy data.
        '''
        # Extract the current pose and orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation

        # Add noise to the position
        noisy_x = x + np.random.normal(0, self.MOTION_DISPERSION_X)
        noisy_y = y + np.random.normal(0, self.MOTION_DISPERSION_Y)

        # Add noise to the orientation (theta)
        # Convert quaternion to angle, then add noise
        theta = quaternion_to_angle(orientation)
        noisy_theta = theta + np.random.normal(0, self.MOTION_DISPERSION_THETA)

        # Create a new Odometry message
        noisy_odom_msg = Odometry()
        noisy_odom_msg.header.stamp = self.get_clock().now().to_msg()
        noisy_odom_msg.header.frame_id = msg.header.frame_id
        noisy_odom_msg.child_frame_id = msg.child_frame_id

        # Update the position
        noisy_odom_msg.pose.pose.position.x = noisy_x
        noisy_odom_msg.pose.pose.position.y = noisy_y
        noisy_odom_msg.pose.pose.position.z = msg.pose.pose.position.z  # Assuming z remains the same

        # Convert noisy theta back to quaternion
        noisy_orientation = angle_to_quaternion(noisy_theta)
        noisy_odom_msg.pose.pose.orientation = noisy_orientation

        # Publish the noisy odometry message
        self.noisy_odom_pub.publish(noisy_odom_msg)

def quaternion_to_angle(quat):
    # Convert quaternion to Euler angle (yaw)
    x, y, z, w = quat.x, quat.y, quat.z, quat.w
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def angle_to_quaternion(angle):
    # Convert Euler angle (yaw) back to quaternion
    q = Quaternion()  # Use Quaternion from geometry_msgs
    half_angle = angle / 2
    q.w = math.cos(half_angle)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half_angle)
    return q

def main(args=None):
    rclpy.init(args=args)
    node = NoisyOdometryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
