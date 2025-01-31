import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import numpy as np
from tf_transformations import quaternion_matrix

class OdomTransformListener(Node):
    def __init__(self):
        super().__init__('odom_tf_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.get_transform)  # Query at 10Hz
        
        self.map_to_odom_tf = None

    def get_transform(self):
        try:
            # Lookup transform from 'odom' to 'base_link' (change as needed)
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time())

            # Extract translation
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z

            # Extract rotation (quaternion)
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w

            # Convert quaternion to 4x4 transformation matrix
            transformation_matrix = quaternion_matrix([qx, qy, qz, qw])
            transformation_matrix[:3, 3] = [tx, ty, tz]  # Set translation part
            
            self.map_to_odom_tf = np.asarray(transformation_matrix)

            self.get_logger().info(f"Transformation Matrix:\n{transformation_matrix}")

        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = OdomTransformListener()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
