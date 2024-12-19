import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion
import csv
import os

class PathComparisonNode(Node):
    def __init__(self):
        super().__init__('path_comparison_node')
        
        # Paths for ground truth and odometry
        self.odom_path_msg = Path()
        self.odom_path_msg.header.frame_id = 'map'

        # Publishers for ground truth and odometry paths
        self.odom_path_pub = self.create_publisher(Path, 'odom_path', 10)

        # Subscriber for odometry
        self.create_subscription(Odometry, '/diff_cont/odom', self.odom_callback, 10)
        
        # Variables to store paths for CSV output
        self.odom_data = []

        # Path for saving CSV file
        self.csv_file_path = os.path.join(os.path.expanduser('~'), 'path_comparison.csv')

    def odom_callback(self, msg: Odometry):
        """Callback for odometry data."""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        # Append pose to odom path and publish
        self.odom_path_msg.poses.append(pose)
        self.odom_path_pub.publish(self.odom_path_msg)

        # Store data for CSV
        self.odom_data.append((pose.pose.position.x, pose.pose.position.y, pose.pose.orientation.z, pose.pose.orientation.w))

def main(args=None):
    rclpy.init(args=args)
    path_comparison_node = PathComparisonNode()

    try:
        rclpy.spin(path_comparison_node)
    except KeyboardInterrupt:
        pass
    finally:
        path_comparison_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
