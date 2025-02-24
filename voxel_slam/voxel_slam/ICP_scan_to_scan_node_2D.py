import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from scipy.spatial import KDTree
from scipy.linalg import svd
import json

import sys
import os

# Get the absolute path of the parent directory
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from Tests.ICP_scan_to_scan import ICPScanToScan
from Tests.ICP_scan_to_scan import ICPScanToScan

class ScanToScanMatching(Node):
    def __init__(self):
        super().__init__('scan_to_scan_matching')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laserscan_callback,
            10
        )
        self.prev_pcd = None
        self.prev_transform = np.eye(3)
        self.scan_matching = ICPScanToScan(max_iterations=20, tolerance=1e-5, visualize=False)

    def laserscan_callback(self, msg):
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = []

        for r, theta in zip(msg.ranges, angles):
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y])  # Z = 0 for 2D lidar scans
        
        current_pcd = np.array(points)
        if self.prev_pcd is not None:
            # transformation = self.icp_manual(self.prev_pcd, current_pcd, max_icp_iterations=20, tolerance=1e-5)
            transformation = self.scan_matching.fit(self.prev_pcd, current_pcd, np.eye(3))
            print(transformation)
            self.prev_transform = self.prev_transform @ transformation
            
             # Extract x, y, and yaw from the transformation matrix
            x, y = self.prev_transform[0, 2], self.prev_transform[1, 2]
            yaw = np.arctan2(self.prev_transform[1, 0], self.prev_transform[0, 0])  # Extract rotation in radians

            # Format values to two decimal places
            translation_str = f"x: {x:.2f}, y: {y:.2f}"
            yaw_str = f"yaw: {yaw:.2f} rad"

            # Print the transformation matrix in a readable format
            formatted_matrix = np.array2string(self.prev_transform, formatter={'float_kind': lambda v: f"{v:.2f}"})

            # Log the values
            self.get_logger().info(f'Pose:\nTranslation: [{translation_str}]\nRotation: [{yaw_str}]')
            # self.get_logger().info(f'Transformation:\n{formatted_matrix}')
        
        self.prev_pcd = current_pcd
        
        def measurement_CB(self, msg):
            """
            Callback to process the received message.
            """
            try:
                # Deserialize the JSON data
                combined_data = json.loads(msg.data)
                self.measurement_data = []
                self.measurement_data = combined_data

                # Process each entry in the combined data
                for entry in combined_data:
                    matrix_id = entry.get('id')
                    rows = entry.get('rows')
                    cols = entry.get('cols')
                    matrix_data = entry.get('data')
                    range_bearing_data = entry.get('range_bearing_data')

                    self.get_logger().info(f"Received matrix ID: {matrix_id}")
                    # self.get_logger().info(f"Matrix dimensions: {rows}x{cols}")
                    # self.get_logger().info(f"Matrix data: {matrix_data}")
                    # self.get_logger().info(f"Range-bearing data: {range_bearing_data}")

                    # If needed, reconstruct the matrix from the flattened data
                    if rows > 0 and cols > 0:
                        matrix = np.array(matrix_data).reshape(rows, cols)
                        # self.get_logger().info(f"Reconstructed matrix:\n{matrix}")

            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to decode JSON: {e}")
                
    
    
    def icp_manual(self, source_pts, target_pts, max_icp_iterations=20, tolerance=1e-5):
        """
        Perform ICP (Iterative Closest Point) manually without Open3D.

        Args:
            source_pts (numpy.ndarray): Source point cloud (Nx3).
            target_pts (numpy.ndarray): Target point cloud (Nx3).
            max_iterations (int): Maximum iterations for convergence.
            tolerance (float): Convergence tolerance.

        Returns:
            np.ndarray: 4x4 transformation matrix aligning source to target.
        """
        # Initialize transformation matrix
        transformation = np.eye(4)
        
        for i in range(max_icp_iterations):
            # Step 1: Find the closest points using KDTree
            kdtree = KDTree(target_pts)
            distances, indices = kdtree.query(source_pts)
            closest_pts = target_pts[indices]

            # Step 2: Compute centroids
            source_centroid = np.mean(source_pts, axis=0)
            target_centroid = np.mean(closest_pts, axis=0)

            # Step 3: Compute centered points
            source_centered = source_pts - source_centroid
            target_centered = closest_pts - target_centroid

            # Step 4: Compute rotation using Singular Value Decomposition (SVD)
            H = source_centered.T @ target_centered
            U, _, Vt = svd(H)
            R = Vt.T @ U.T

            # Ensure R is a valid rotation matrix (i.e., no reflection)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Step 5: Compute translation
            t = target_centroid - R @ source_centroid

            # Step 6: Apply transformation
            source_pts = (R @ source_pts.T).T + t

            # Step 7: Update the transformation matrix
            T_new = np.eye(4)
            T_new[:3, :3] = R
            T_new[:3, 3] = t
            transformation = T_new @ transformation  # Accumulate transformation

            # Step 8: Check for convergence
            mean_error = np.mean(distances)
            if mean_error < tolerance:
                break

        return transformation

def main(args=None):
    rclpy.init(args=args)
    node = ScanToScanMatching()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
