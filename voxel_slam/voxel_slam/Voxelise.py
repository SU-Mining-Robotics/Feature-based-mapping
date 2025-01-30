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
        self.transformed_marker_publisher = self.create_publisher(MarkerArray, "/transformed_voxel_grid_markers", 10)
        self.orignal_marker_publisher = self.create_publisher(MarkerArray, "/original_voxel_grid_markers", 10)
        self.voxel_size = 0.3  # Adjustable voxel size
        
        self.first_scan = None
        self.first_voxel_grid = None

        self.get_logger().info("PointCloudProcessor node initialized.")

    def pointcloud_callback(self, msg):
        # Convert ROS 2 PointCloud2 to Open3D point cloud
        pcd = self.ros_to_open3d(msg)
        # self.get_logger().info("Point cloud received with %d points." % len(pcd.points))

        # Downsample the point cloud
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)  # Gives centroid of points in a voxel
        # self.get_logger().info("Downsampled point cloud to %d points." % len(downsampled_pcd.points))

        # Create a voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(downsampled_pcd, voxel_size=self.voxel_size)
        # self.get_logger().info("Voxel grid created with %d voxels." % len(voxel_grid.get_voxels()))
        
        # Voxel point cloud from grid centres
        voxel_centres = np.array([voxel.grid_index * self.voxel_size + voxel_grid.origin for voxel in voxel_grid.get_voxels()])
        voxel_centre_pcd = o3d.geometry.PointCloud()
        voxel_centre_pcd.points = o3d.utility.Vector3dVector(voxel_centres)
        
        # Save the first scan as an Open3D PointCloud
        if self.first_scan is None:
            self.first_scan = voxel_centre_pcd  # Save the entire PointCloud, not just the points
            self.first_voxel_grid = voxel_grid
            self.get_logger().info("First scan saved.")
        else:
            # Perform ICP
            icp = o3d.pipelines.registration.registration_icp(
                voxel_centre_pcd,             # Source: Open3D PointCloud
                self.first_scan,              # Target: Open3D PointCloud
                self.voxel_size,              # Max correspondence distance
                np.eye(4),                    # Initial transformation
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Apply the transformation to the point cloud
            transformed_pcd = downsampled_pcd.transform(icp.transformation)
            # self.get_logger().info("ICP performed.")
            self.get_logger().info("Transformation matrix:\n" + str(icp.transformation))
            
            # Create a new voxel grid from the transformed point cloud
            transformed_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(transformed_pcd, voxel_size=self.voxel_size)
            # self.get_logger().info("New voxel grid created after applying transformation.")
            
            original_marker_array = self.voxel_grid_to_markers(self.first_voxel_grid, msg.header, colour=[1.0, 0.0, 0.0])
            transformed_marker_array = self.voxel_grid_to_markers(transformed_voxel_grid, msg.header, colour=[0.0, 0.0, 1.0])
            
            
            self.transformed_marker_publisher.publish(transformed_marker_array)
            self.orignal_marker_publisher.publish(original_marker_array)
        
        # Convert the voxel grid to a MarkerArray for RViz
        marker_array = self.voxel_grid_to_markers(voxel_grid, msg.header)
        

        # Publish the voxel grid as markers
        self.marker_publisher.publish(marker_array)
        
        # self.get_logger().info("Voxel grid published as markers.")

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

    def voxel_grid_to_markers(self, voxel_grid, header, colour=[0.0, 1.0, 0.0]):
        """Convert Open3D VoxelGrid to a MarkerArray for RViz."""
        marker_array = MarkerArray()
        marker_id = 0

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
            marker.color.r = colour[0]
            marker.color.g = colour[1]
            marker.color.b = colour[2]
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
