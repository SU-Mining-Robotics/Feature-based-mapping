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
        self.marker_publisher = self.create_publisher(MarkerArray, "/voxel_grid_markers", 10)
        self.voxel_size = 0.3  # Adjustable voxel size
        self.get_logger().info("PointCloudProcessor node initialized.")

    def pointcloud_callback(self, msg):
        # Convert ROS 2 PointCloud2 to Open3D point cloud
        pcd = self.ros_to_open3d(msg)
        self.get_logger().info("Point cloud received with %d points." % len(pcd.points))

        # Downsample the point cloud
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        self.get_logger().info("Downsampled point cloud to %d points." % len(downsampled_pcd.points))

        # Segment ground plane using RANSAC
        ground_plane, walls = self.segment_ground_plane(downsampled_pcd)

        # Create voxel grids for ground and walls
        ground_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(ground_plane, voxel_size=self.voxel_size)
        wall_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(walls, voxel_size=self.voxel_size)

       # Convert the voxel grids to markers
        marker_array = MarkerArray()
        marker_array.markers.extend(self.voxel_grid_to_markers(ground_voxels, msg.header, color="green").markers)
        marker_array.markers.extend(self.voxel_grid_to_markers(wall_voxels, msg.header, color="blue").markers)


        # Publish the voxel grid as markers
        self.marker_publisher.publish(marker_array)
        self.get_logger().info("Voxel grid published as markers.")

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
        """Segment the ground plane using RANSAC and return ground and wall segments."""
        # Apply RANSAC to segment the ground plane (assume Z=0 is the ground)
        plane_model, inlier_indices = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
        ground_plane = pcd.select_by_index(inlier_indices)
        walls = pcd.select_by_index(inlier_indices, invert=True)  # The rest are walls
        return ground_plane, walls

    def voxel_grid_to_markers(self, voxel_grid, header, color="green"):
        """Convert Open3D VoxelGrid to a MarkerArray for RViz."""
        marker_array = MarkerArray()
        marker_id = 0
        color_dict = {"green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}

        for voxel in voxel_grid.get_voxels():
            cube_center = voxel.grid_index * voxel_grid.voxel_size + voxel_grid.origin
            marker = Marker()
            marker.header = header
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(cube_center[0])
            marker.pose.position.y = float(cube_center[1])
            marker.pose.position.z = float(cube_center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.voxel_size
            marker.scale.y = self.voxel_size
            marker.scale.z = self.voxel_size

            # Set color based on ground or wall
            marker.color.r, marker.color.g, marker.color.b = color_dict.get(color, (0.0, 1.0, 0.0))
            marker.color.a = 0.6  # Semi-transparent
            marker_array.markers.append(marker)
            marker_id += 1

        return marker_array
    

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
