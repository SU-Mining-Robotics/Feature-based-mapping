import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
import tf_transformations
import struct
import matplotlib.pyplot as plt
import scipy.spatial.transform

class ScanToMapMatching(Node):
    def __init__(self):
        super().__init__('scan_to_map_matching')
        
        self.subscription = self.create_subscription(PointCloud2, '/laser_controller/out', self.pointcloud_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/diff_cont/odom', self.odom_callback, 10)
        self.gt_subscriber = self.create_subscription(ModelStates, '/model_states', self.model_callback, 10)
        
        self.pcd_pub = self.create_publisher(PointCloud2, 'global_map_pcd', 10)
        self.timer = self.create_timer(1.0, self.publish_map)
        
        self.global_map = o3d.geometry.PointCloud()
        self.real_transformation_matrix = np.eye(4)
        self.odom_transform = np.eye(4)
        self.odom_transform_prev = np.eye(4)
        self.corrected_transform = np.eye(4)
        self.noisy_estimate = np.eye(4)
        
        self.init_visualization()
        
        self.voxel_size = 0.2
        self.voxel_centre_map_pcd = o3d.geometry.PointCloud()
        
        self.get_logger().info("Scan-to-Map Matching node initialized.")
    
    def init_visualization(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Scan-to-Map Matching")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)
    
    def update_visualization(self):
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_title("Scan-to-Map Matching")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        # Extract x, y from transformations
        x_real, y_real = self.real_transformation_matrix[:2, 3]
        x_odom, y_odom = self.odom_transform[:2, 3]
        x_corrected, y_corrected = self.corrected_transform[:2, 3]
        x_noisy, y_noisy = self.noisy_estimate[:2, 3]

        # Extract yaw (orientation) from transformations
        yaw_real = np.arctan2(self.real_transformation_matrix[1, 0], self.real_transformation_matrix[0, 0])
        yaw_odom = np.arctan2(self.odom_transform[1, 0], self.odom_transform[0, 0])
        yaw_corrected = np.arctan2(self.corrected_transform[1, 0], self.corrected_transform[0, 0])
        yaw_noisy = np.arctan2(self.noisy_estimate[1, 0], self.noisy_estimate[0, 0])

        # Plot positions
        self.ax.scatter(x_real, y_real, c='g', label='Real Pose')
        self.ax.scatter(x_odom, y_odom, c='b', label='Odom Pose')
        self.ax.scatter(x_corrected, y_corrected, c='r', label='Corrected Pose')
        self.ax.scatter(x_noisy, y_noisy, c='y', label='Noisy Pose')

        # Plot orientation as arrows
        arrow_length = 0.5
        self.ax.arrow(x_real, y_real, arrow_length * np.cos(yaw_real), arrow_length * np.sin(yaw_real),
                    head_width=0.1, head_length=0.15, fc='g', ec='g')
        self.ax.arrow(x_odom, y_odom, arrow_length * np.cos(yaw_odom), arrow_length * np.sin(yaw_odom),
                    head_width=0.1, head_length=0.15, fc='b', ec='b')
        self.ax.arrow(x_corrected, y_corrected, arrow_length * np.cos(yaw_corrected), arrow_length * np.sin(yaw_corrected),
                    head_width=0.1, head_length=0.15, fc='r', ec='r')
        self.ax.arrow(x_noisy, y_noisy, arrow_length * np.cos(yaw_noisy), arrow_length * np.sin(yaw_noisy),
                    head_width=0.1, head_length=0.15, fc='y', ec='y')

        self.ax.legend()
        plt.pause(0.1)

    def pointcloud_callback(self, msg):
        current_pcd = self.ros_to_open3d(msg)
        self.get_logger().info("Point cloud received with %d points." % len(current_pcd.points))
        current_pcd, _ = self.remove_ground_plane(current_pcd)
        # map_down, _ = self.remove_ground_plane(map_down)
        
        if len(self.global_map.points) > 0 and self.odom_data is not None:
    
            self.odom_transform = self.odom_to_transform(self.odom_data)
            self.noisy_estimate = self.add_noise_to_transform(self.real_transformation_matrix)
            
            #Scan matching
            scan_down = current_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            map_down = self.global_map
           
            transformation = self.match_scan_to_map(scan_down, map_down, self.noisy_estimate )
            self.corrected_transform = transformation
            # self.corrected_transform = initial_transform @ transformation
            
            #Map update
            current_pcd.transform(self.real_transformation_matrix)
            self.global_map += current_pcd
           
            self.global_map = self.global_map.voxel_down_sample(voxel_size=0.2)
            self.get_logger().info("Current point cloud map of %d points." % len(self.global_map.points))
            
            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.global_map, voxel_size=self.voxel_size)
            # self.get_logger().info("Voxel grid map created with %d voxels." % len(voxel_grid.get_voxels()))
            # voxel_centres = np.array([voxel.grid_index * self.voxel_size + voxel_grid.origin for voxel in voxel_grid.get_voxels()])
            # self.voxel_centre_map_pcd.points = o3d.utility.Vector3dVector(voxel_centres)
            # self.get_logger().info("Downsampled map point cloud to %d points." % len(self.voxel_centre_map_pcd.points))
        else:
            self.global_map = current_pcd
            self.voxel_centre_map_pcd = current_pcd
            
        # self.pcd_pub.publish(self.open3d_to_ros(self.voxel_centre_map_pcd))
        self.pcd_pub.publish(self.open3d_to_ros( self.global_map))
        # self.get_logger().info("Current point cloud map of %d points." % len(self.global_map.points))
        # self.get_logger().info("Current point cloud map of %d points." % len(self.voxel_centre_map_pcd.points))
        # self.global_map = self.voxel_centre_map_pcd
        # self.global_map.points = self.voxel_centre_map_pcd.points
            
    
    def odom_callback(self, msg):
        self.odom_data = msg.pose.pose
    
    def model_callback(self, model_data):
        try:
            robot_index = model_data.name.index("my_bot")
            pose = model_data.pose[robot_index]
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            yaw = tf_transformations.euler_from_quaternion([
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]
            self.real_transformation_matrix = self.pose_to_transform_matrix(x, y, z, yaw)
        except Exception as e:
            self.get_logger().warn(f"Error in model callback: {e}")
    
    def publish_map(self):
        if len(self.global_map.points) == 0:
            return

        self.pcd_pub.publish(self.open3d_to_ros(self.global_map))
        # self.marker_pub.publish(self.open3d_to_marker_array(self.global_map))
    
    def ros_to_open3d(self, ros_msg):
        points = [struct.unpack_from("fff", ros_msg.data, i) for i in range(0, len(ros_msg.data), ros_msg.point_step)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def open3d_to_ros(self, cloud):
        points = np.asarray(cloud.points)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = np.array(points, dtype=np.float32).tobytes()
        return msg
    
    def match_scan_to_map(self, scan, map, initial_transform):
        threshold = 0.01
        map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        scan.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            scan, map, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        return reg_p2p.transformation

    def remove_ground_plane(self, pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, 
                                                ransac_n=ransac_n, 
                                                num_iterations=num_iterations)
        ground = pcd.select_by_index(inliers)
        filtered_pcd = pcd.select_by_index(inliers, invert=True)
        return filtered_pcd, ground

    def odom_to_transform(self, odom_data):
        position = odom_data.position
        orientation = odom_data.orientation
        rotation_matrix = tf_transformations.quaternion_matrix([
            orientation.x, orientation.y, orientation.z, orientation.w])
        transform = np.eye(4)
        transform[:3, 3] = [position.x, position.y, position.z]
        transform[:3, :3] = rotation_matrix[:3, :3]
        return transform
    
    def pose_to_transform_matrix(self, x, y, z, yaw):
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        transform[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        return transform
    
    # def add_noise_to_transform(self, transform, trans_noise=0.05, rot_noise=0.2):
    #     # Extract rotation (3x3) and translation (3x1)
    #     R = transform[:3, :3]
    #     t = transform[:3, 3]
    #     # Add Gaussian noise to translation
    #     t_noisy = t + np.random.normal(0, trans_noise, size=3)
    #     # Generate small random rotation perturbation using axis-angle representation
    #     random_axis = np.random.normal(0, 1, size=3)  # Random 3D axis
    #     random_axis /= np.linalg.norm(random_axis)  # Normalize to unit vector
    #     angle = np.random.normal(0, rot_noise)  # Small random angle
    #     perturbation = scipy.spatial.transform.Rotation.from_rotvec(angle * random_axis)
    #     R_noisy = perturbation.as_matrix() @ R  # Apply noise to rotation
    #     # Construct the noisy transformation matrix
    #     noisy_transform = np.eye(4)
    #     noisy_transform[:3, :3] = R_noisy
    #     noisy_transform[:3, 3] = t_noisy

    #     return noisy_transform
    
    def add_noise_to_transform(self, transform, trans_noise=0.05, rot_noise=0.2):
        """
        Adds Gaussian noise to the x, y translation and yaw rotation of a 4x4 transformation matrix.

        Args:
            transform (np.array): 4x4 transformation matrix.
            trans_noise (float): Standard deviation of translation noise in meters.
            rot_noise (float): Standard deviation of rotation noise in radians.

        Returns:
            np.array: Noisy 4x4 transformation matrix.
        """
        # Extract rotation matrix and translation vector
        R = transform[:3, :3]
        t = transform[:3, 3]

        # Add noise only to x and y translation (t[0] and t[1])
        t_noisy = t.copy()
        t_noisy[:2] += np.random.normal(0, trans_noise, size=2)  # Add noise to x, y only

        # Convert rotation matrix to yaw angle
        yaw = np.arctan2(R[1, 0], R[0, 0])  # Extract yaw from rotation matrix

        # Add noise to yaw angle
        yaw_noisy = yaw + np.random.normal(0, rot_noise)

        # Convert back to rotation matrix (assume no roll or pitch changes)
        R_noisy = np.array([
            [np.cos(yaw_noisy), -np.sin(yaw_noisy), 0],
            [np.sin(yaw_noisy),  np.cos(yaw_noisy), 0],
            [0,                 0,                 1]
        ])

        # Construct the noisy transformation matrix
        noisy_transform = np.eye(4)
        noisy_transform[:3, :3] = R_noisy
        noisy_transform[:3, 3] = t_noisy

        return noisy_transform

def main(args=None):
    rclpy.init(args=args)
    
    node = ScanToMapMatching()

    # Run ROS and update visualization in the main thread
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.update_visualization()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
