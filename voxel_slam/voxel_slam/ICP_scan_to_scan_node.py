import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import struct

class ScanToScanMatching(Node):
    def __init__(self):
        super().__init__('scan_to_scan_matching')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/laser_controller/out',
            self.pointcloud_callback,
            10
        )
        self.prev_pcd = None
        self.prev_transform = np.eye(4)

    def pointcloud_callback(self, msg):
        current_pcd = self.ros_to_open3d(msg)
        if self.prev_pcd is not None:
            transformation = self.match_scans(self.prev_pcd, current_pcd)
            self.prev_transform = self.prev_transform @ transformation
            # self.get_logger().info(f'Transformation:\n{self.prev_transform}')
            formatted_matrix = np.array2string(self.prev_transform, formatter={'float_kind': lambda x: f"{x:.2f}"})
            self.get_logger().info(f'Transformation:\n{formatted_matrix}')
        self.prev_pcd = current_pcd

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

    def match_scans(self, source, target):
        source_down = source.voxel_down_sample(voxel_size=0.05)
        target_down = target.voxel_down_sample(voxel_size=0.05)
        
        threshold = 0.1
        trans_init = np.eye(4)
        
        # #Generalised ICP (Does not work)
        # source_down.estimate_covariances()
        # target_down.estimate_covariances()
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        # )
        
        # #Point_to_plane ICP
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        # #Point_to_point ICP
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # )
        return reg_p2p.transformation

    def compute_velocity(self, dt):
        p_k = self.prev_transform[:3, 3]
        p_k_minus_1 = self.prev_pcd.points[0] if self.prev_pcd else np.zeros(3)
        velocity = (p_k - p_k_minus_1) / dt
        return velocity

def main(args=None):
    rclpy.init(args=args)
    node = ScanToScanMatching()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()