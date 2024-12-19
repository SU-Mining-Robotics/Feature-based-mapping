#PARAMETERS:
#PUBLISHES:
#SUBSCRIBES:
#SERVICE CLIENTS:

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from landmark_extract import utils as Utils
     
     
class EKF_SLAM(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("EKF_SLAM_node") # MODIFY NAME
        
        # Declare parameters
        self.declare_parameter('odometry_topic', "/diff_cont/odom")
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, self.get_parameter('odometry_topic').value, self.odomCB, 10)
        
    def odomCB(self, msg):
        '''
        Store deltas between consecutive odometry messages in the coordinate space of the car.
        Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.
        '''
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        self.current_speed = msg.twist.twist.linear.x

        if isinstance(self.last_pose, np.ndarray):
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array([position - self.last_pose[0:2]]).transpose()
            local_delta = (rot*delta).transpose()
            
            self.odometry_data = np.array([local_delta[0,0], local_delta[0,1], orientation - self.last_pose[2]])
            self.last_pose = pose
            self.last_stamp = msg.header.stamp
            self.odom_initialized = True
        else:
            self.get_logger().info('...Received first Odometry message')
            self.last_pose = pose
            
    
    
    def ekf_slam(xEst, PEst, u, z):
        # Predict
        G, Fx = jacob_motion(xEst, u)
        xEst[0:STATE_SIZE] = motion_model(xEst[0:STATE_SIZE], u)
        PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
        initP = np.eye(2)
        
        #Intermediate matrices to store the predicted state and covariance matrix
        xPredicted = xEst
        pPredicted = PEst

        # Update
        
        return xEst, PEst, xPredicted, pPredicted


def main(args=None):
    rclpy.init(args=args)
    node = EKF_SLAM() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()