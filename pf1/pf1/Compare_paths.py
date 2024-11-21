import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class OdometryPathPublisher(Node):
    def __init__(self):
        super().__init__('odometry_path_publisher')
        
        # Set the frame ID for the path
        self.odom_id = 'odom'

        # Initialize path message
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'
        self.noisy_path_msg = Path()
        self.noisy_path_msg.header.frame_id = 'map'

        # Variables to store the latest odometry pose
        self.latest_pose = None
        self.latest_noisy_pose = None

        # Publisher for the path
        self.path_publisher = self.create_publisher(Path, 'odom_path', 10)
        self.noisy_path_publisher = self.create_publisher(Path, 'noisy_odom_path', 10)

        # Subscriber to the odom topic
        self.odom_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.noisy_odom_subscriber = self.create_subscription(Odometry, '/ego_racecar/noisy_odom', self.noisy_odom_callback, 10)

        # Timer to publish path at regular intervals (e.g., every 100 ms)
        self.timer = self.create_timer(0.1, self.path_timer_callback)
        self.timer = self.create_timer(0.1, self.noisy_path_timer_callback)

    def odom_callback(self, msg: Odometry):
        """Callback to store the latest odometry data."""
        self.latest_pose = PoseStamped()
        self.latest_pose.header.frame_id = self.odom_id
        self.latest_pose.header.stamp = self.get_clock().now().to_msg()
        self.latest_pose.pose.position = msg.pose.pose.position
        self.latest_pose.pose.orientation = msg.pose.pose.orientation
        # Log the odometry message
        # self.get_logger().info(f"Received odometry message: position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z}), "
        #                    f"orientation=({msg.pose.pose.orientation.x}, {msg.pose.pose.orientation.y}, {msg.pose.pose.orientation.z}, {msg.pose.pose.orientation.w})")
        
    def noisy_odom_callback(self, msg: Odometry):
        """Callback to store the latest odometry data."""
        self.latest_noisy_pose = PoseStamped()
        self.latest_noisy_pose.header.frame_id = self.odom_id
        self.latest_noisy_pose.header.stamp = self.get_clock().now().to_msg()
        self.latest_noisy_pose.pose.position = msg.pose.pose.position
        self.latest_noisy_pose.pose.orientation = msg.pose.pose.orientation
        

    def path_timer_callback(self):
        """Callback to publish the path using the stored odometry data."""
        # Only add the latest pose to the path if we have received odometry data
        if self.latest_pose is not None:
            # Append the latest pose to the path message
            self.path_msg.poses.append(self.latest_pose)
            self.path_msg.header.stamp = self.latest_pose.header.stamp

            # Publish the updated path
            self.path_publisher.publish(self.path_msg)
            
             # Log the path message
            self.get_logger().info(f"Published path message with {len(self.path_msg.poses)} poses.")
            
    def noisy_path_timer_callback(self):
        """Callback to publish the path using the stored odometry data."""
        # Only add the latest pose to the path if we have received odometry data
        if self.latest_noisy_pose is not None:
            # Append the latest pose to the path message
            self.noisy_path_msg.poses.append(self.latest_noisy_pose)
            self.noisy_path_msg.header.stamp = self.latest_noisy_pose.header.stamp

            # Publish the updated path
            self.noisy_path_publisher.publish(self.noisy_path_msg)
            
             # Log the path message
            self.get_logger().info(f"Published noisy path message with {len(self.noisy_path_msg.poses)} poses.")

def main(args=None):
    rclpy.init(args=args)
    odometry_path_publisher = OdometryPathPublisher()

    try:
        rclpy.spin(odometry_path_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        odometry_path_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()