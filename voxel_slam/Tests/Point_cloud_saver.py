import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d
import struct

class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')
        
        # Subscribe to the point cloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/laser_controller/out',
            self.pointcloud_callback,
            10  # QoS profile
        )
        self.subscription  # Prevent unused variable warning
        self.save_flag = False

    def pointcloud_callback(self, msg):
        """ Callback function that processes the incoming PointCloud2 message. """
        self.get_logger().info("Received point cloud message. Converting and saving...")

        # Convert PointCloud2 message to numpy array
        points = self.pointcloud2_to_array(msg)

        # Convert to Open3D point cloud format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Save as a PLY file
        if not self.save_flag:
            filename = "saved_pointcloud.ply"
            o3d.io.write_point_cloud(filename, pcd)
            self.get_logger().info(f"Point cloud saved as {filename}")

       
    def pointcloud2_to_array(self, msg):
        """ Converts a ROS2 PointCloud2 message to a NumPy array of (x, y, z) points. """
        points = []
        point_step = msg.point_step
        data = msg.data

        for i in range(0, len(data), point_step):
            x, y, z = struct.unpack_from('fff', data, i)  # Extract XYZ coordinates
            points.append([x, y, z])

        return np.array(points)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
