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

        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = 'map'

        # Publishers for ground truth and odometry paths
        self.odom_path_pub = self.create_publisher(Path, 'odom_path', 10)
        self.gt_path_pub = self.create_publisher(Path, 'ground_truth_path', 10)

        # Subscriber for odometry
        self.create_subscription(Odometry, '/diff_cont/odom', self.odom_callback, 10)
        
        # TF listener for ground truth
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.tf_callback)  # Timer for ground truth at 10 Hz

        # Variables to store paths for CSV output
        self.odom_data = []
        self.gt_data = []

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

    def tf_callback(self):
        """Callback for ground truth using TF."""
        try:
            # Lookup transform from world to base_link for ground truth
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = self.get_clock().now().to_msg()

            # Set the position and orientation
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation

            # Append pose to ground truth path and publish
            self.gt_path_msg.poses.append(pose)
            self.gt_path_pub.publish(self.gt_path_msg)

            # Store data for CSV
            self.gt_data.append((pose.pose.position.x, pose.pose.position.y, pose.pose.orientation.z, pose.pose.orientation.w))

        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")

    def save_to_csv(self):
        """Save odometry and ground truth paths to CSV file."""
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Odom_X', 'Odom_Y', 'Odom_Orientation_Z', 'Odom_Orientation_W', 
                             'GT_X', 'GT_Y', 'GT_Orientation_Z', 'GT_Orientation_W'])
            
            # Write data row-by-row
            for odom, gt in zip(self.odom_data, self.gt_data):
                writer.writerow([*odom, *gt])

        self.get_logger().info(f"Path data saved to {self.csv_file_path}")

    def destroy_node(self):
        """Override destroy_node to save data before shutdown."""
        self.save_to_csv()
        super().destroy_node()

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
