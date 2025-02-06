import open3d as o3d
import numpy as np
import rclpy
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry
import struct
import tf_transformations


from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Tests import Utils as Utils


class ScanToMapMatching(Node):
    def __init__(self):
        super().__init__('scan_to_map_matching')

        # Create subscribers
        self.subscription = self.create_subscription(
            PointCloud2,
            '/laser_controller/out',
            self.pointcloud_callback,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/diff_cont/odom',
            self.odom_callback,
            10
        )
        
        # Temporary subscriber for ground truth data
        self.gt_subscriber = self.create_subscription(ModelStates, '/model_states', self.model_callback, 10)

        # Initialize the global map and other variables
        self.global_map = o3d.geometry.PointCloud()
        self.prev_transform = np.eye(4)  # Start at identity transformation
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.real_transformation_matrix = np.eye(4)

        # Initialize odometry data
        self.odom_data = None

    def pointcloud_callback(self, msg):
        current_pcd = self.ros_to_open3d(msg)

        if len(self.global_map.points) > 0:
            # Use odometry for initial alignment
            if self.odom_data is not None:
                initial_transform = self.odom_to_transform(self.odom_data)
                # formatted_matrix = np.array2string(initial_transform, formatter={'float_kind': lambda x: f"{x:.2f}"})
                # self.get_logger().info(f'Transformation:\n{formatted_matrix}')
            else:
                initial_transform = np.eye(4)  # Use identity if no odom data available

            scan_down = current_pcd.voxel_down_sample(voxel_size=0.05)
            map_down = self.global_map.voxel_down_sample(voxel_size=0.05)
            
            # Align the new scan to the global map with initial transform
            transformation = self.match_scan_to_map(scan_down, map_down, initial_transform)
            localisation_transform = initial_transform @ transformation
            formatted_matrix = np.array2string(localisation_transform, formatter={'float_kind': lambda x: f"{x:.2f}"})
            self.get_logger().info(f'Transformation intial:\n{formatted_matrix}')

            # Apply the corrected pose before adding to the map
            current_pcd.transform(self.real_transformation_matrix)
            self.global_map += current_pcd
            self.global_map = self.global_map.voxel_down_sample(voxel_size=0.05)  # Downsample for efficiency
            
            # formatted_matrix = np.array2string(self.prev_transform, formatter={'float_kind': lambda x: f"{x:.2f}"})
            # self.get_logger().info(f'Transformation:\n{formatted_matrix}')

            # Publish corrected pose
            # self.publish_tf(self.prev_transform)

        else:
            # First scan: initialize the map
            self.global_map = current_pcd
            print("Initialized global map.")

    def odom_callback(self, msg):
        """Callback for odometry data."""
        self.odom_data = msg.pose.pose  # Store the latest odometry data
        
    def model_callback(self, model_data):
        # if self.message_counter_gt % self.process_interval != 0:
        #     return

        try:
            robot_index = model_data.name.index("my_bot")
            robot_pose = PoseStamped()
            robot_pose.header.stamp = self.get_clock().now().to_msg()
            robot_pose.pose = model_data.pose[robot_index]

            time_s = robot_pose.header.stamp.sec
            time_ns = robot_pose.header.stamp.nanosec
            x = robot_pose.pose.position.x
            y = robot_pose.pose.position.y
            z = robot_pose.pose.position.z
            yaw = Utils.quaternion_to_angle(robot_pose.pose.orientation)

            # Construct transformation matrix from position (x, y, z) and yaw
            transformation_matrix = Utils.pose_to_transform_matrix(x, y, z, yaw)
            self.real_transformation_matrix = transformation_matrix
            formatted_matrix = np.array2string(transformation_matrix, formatter={'float_kind': lambda x: f"{x:.2f}"})
            self.get_logger().info(f'Transformation intial:\n{formatted_matrix}')
            
            
        except Exception as e:
            self.get_logger().warn(f"Error in model callback: {e}")
        
    def ros_to_open3d(self, ros_msg):
        """Convert ROS 2 PointCloud2 to Open3D point cloud."""
        points = []
        fmt = "fff"  # XYZ format
        for i in range(0, len(ros_msg.data), ros_msg.point_step):
            x, y, z = struct.unpack_from(fmt, ros_msg.data, offset=i)
            points.append([x, y, z])

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def match_scan_to_map(self, scan, map, initial_transform):
        """Align scan to global map using ICP."""
     
        # Create a TransformationEstimationForGeneralizedICP instance
        threshold = 0.1
        map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        
        # Use the initial transform for ICP alignment
        reg_p2p = o3d.pipelines.registration.registration_icp(
            scan, map, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        return reg_p2p.transformation

    def odom_to_transform(self, odom_data):
        """Convert odometry (position + quaternion) to a 4x4 transformation matrix."""
        position = odom_data.position
        orientation = odom_data.orientation

        # Convert quaternion to rotation matrix
        rotation_matrix = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
        
        # Create a 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, 3] = [position.x, position.y, position.z]  # Set translation
        transform[:3, :3] = rotation_matrix[:3, :3]  # Set rotation

        return transform

    def publish_tf(self, transform):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        msg.transform.translation.x = transform[0, 3]
        msg.transform.translation.y = transform[1, 3]
        msg.transform.translation.z = transform[2, 3]

        # Convert rotation matrix to quaternion using tf_transformations
        quat = tf_transformations.quaternion_from_matrix(transform)
        msg.transform.rotation.x = quat[0]
        msg.transform.rotation.y = quat[1]
        msg.transform.rotation.z = quat[2]
        msg.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ScanToMapMatching()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
