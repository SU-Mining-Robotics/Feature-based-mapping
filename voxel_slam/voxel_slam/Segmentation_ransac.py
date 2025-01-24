import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import open3d as o3d
import numpy as np
import struct
from std_msgs.msg import Header


class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__("pointcloud_processor")
        self.subscription = self.create_subscription(
            PointCloud2,
            "/laser_controller/out",
            self.pointcloud_callback,
            10,
        )
        self.marker_publisher = self.create_publisher(MarkerArray, "/segmented_markers", 10)
        self.voxel_size = 0.3  # Adjustable voxel size
        self.ransac_distance_threshold = 0.1  # Threshold for ground plane segmentation
        self.get_logger().info("PointCloudProcessor node initialized.")

    def pointcloud_callback(self, msg):
        # Convert ROS 2 PointCloud2 to Open3D point cloud
        pcd = self.ros_to_open3d(msg)
        self.get_logger().info("Point cloud received with %d points." % len(pcd.points))

        # Downsample the point cloud
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        self.get_logger().info("Downsampled point cloud to %d points." % len(downsampled_pcd.points))

        # Segment ground plane
        ground_plane, walls = self.segment_ground_plane(downsampled_pcd)

        # Publish segmented scans as markers
        self.publish_segmented_markers(ground_plane, walls, msg.header)

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

    def segment_ground_plane(self, pcd):
        """Segment the ground plane using RANSAC."""
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.ransac_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        ground_plane = pcd.select_by_index(inliers)
        walls = pcd.select_by_index(inliers, invert=True)
        self.get_logger().info("Ground plane segmented with %d points." % len(ground_plane.points))
        self.get_logger().info("Walls segmented with %d points." % len(walls.points))
        return ground_plane, walls

    def publish_segmented_markers(self, ground_plane, walls, header):
        """Publish segmented scans as markers in RViz."""
        marker_array = MarkerArray()

        # Ground plane markers
        marker_array.markers.extend(self.create_markers(ground_plane, header, [0.0, 1.0, 0.0], "ground_plane"))

        # Walls markers
        marker_array.markers.extend(self.create_markers(walls, header, [1.0, 0.0, 0.0], "walls"))

        self.marker_publisher.publish(marker_array)

    def create_markers(self, pcd, header, color, ns):
        """Create markers for a given point cloud."""
        markers = []
        for i, point in enumerate(np.asarray(pcd.points)):
            marker = Marker()
            marker.header = header
            marker.id = i
            marker.ns = ns
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.voxel_size / 2
            marker.scale.y = self.voxel_size / 2
            marker.scale.z = self.voxel_size / 2
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8  # Semi-transparent
            markers.append(marker)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
