#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class RobotPosePublisher(Node):
    def __init__(self):
        super().__init__('slam')
        
        # Subscribe to the Gazebo model states topic
        self.subscription = self.create_subscription(ModelStates, '/model_states', self.model_callback, 10)
        
        # Publisher for PoseStamped (optional) and Path
        self.pose_publisher = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.path_publisher = self.create_publisher(Path, '/robot_gt_path', 10)
        
        # Initialize a Path message
        self.path = Path()
        self.path.header.frame_id = "map"  # Set the frame ID according to your setup

    def model_callback(self, model_data):
        # Find the index of the robot model in the model_data
        try:
            robot_index = model_data.name.index("my_bot")
            
            # Retrieve the pose of the robot
            robot_pose = PoseStamped()
            robot_pose.header.stamp = self.get_clock().now().to_msg()
            robot_pose.pose = model_data.pose[robot_index]
            
            # Print or publish the robot pose
            self.get_logger().info(f"Robot position: {robot_pose.pose.position}")
            self.get_logger().info(f"Robot orientation: {robot_pose.pose.orientation}")
            
            # Optionally publish the pose to a new topic
            self.pose_publisher.publish(robot_pose)
            
            # Add the current pose to the path and publish the path
            self.path.header.stamp = robot_pose.header.stamp
            self.path.poses.append(robot_pose)
            self.path_publisher.publish(self.path)
        
        except ValueError:
            self.get_logger().warn("Robot 'my_bot' not found in model states.")

def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)

    # Create the RobotPosePublisher node
    robot_pose_publisher = RobotPosePublisher()

    # Spin the node so it keeps receiving callbacks
    rclpy.spin(robot_pose_publisher)

    # Shutdown the ROS 2 Python client library
    robot_pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
