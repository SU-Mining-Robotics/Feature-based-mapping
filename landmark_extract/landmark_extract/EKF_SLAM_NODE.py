# PARAMETERS:
# PUBLISHES:
# SUBSCRIBES:
# SERVICE CLIENTS:

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from landmark_extract import utils as Utils
from std_msgs.msg import String
import json

import time
import sys
import os

# Add the directory containing the module to sys.path
# This is neccessary since the module is in the same directroy as the node, but we will probably be in the workspace directory when running it
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# EKF state covariance
M = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 2  # Landmark size [x, y]


class EKF_SLAM(Node):  # MODIFY NAME
    def __init__(self):
        super().__init__("EKF_SLAM_node")  # MODIFY NAME

        # Declare parameters
        self.declare_parameter('odometry_topic', "/diff_cont/odom")

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, self.get_parameter('odometry_topic').value, self.odomCB, 10)
        self.subscription = self.create_subscription(String, 'Pseudo_matrices', self.matrixCB, 10)

        # State initialization
        self.xEst = np.zeros(STATE_SIZE)  # Initial state [x, y, yaw]
        self.PEst = np.eye(STATE_SIZE)  # Initial covariance matrix
        self.odom_initialized = False
        self.last_pose = None
        self.last_stamp = None
        
        # various data containers used in the EKF algorithm
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        
        self.get_logger().info('EKF SLAM node has been initialized.')

    def odomCB(self, msg):
        '''
        Store deltas between consecutive odometry messages in the coordinate space of the car.
        Apply the velocity motion model to predict the new state.
        '''
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z

        if self.odom_initialized:
            # dt = (msg.header.stamp.sec - self.last_stamp.sec) + \
            #      (msg.header.stamp.nanosec - self.last_stamp.nanosec) * 1e-9
            # if dt > 0:
            #     u = np.array([linear_velocity, angular_velocity])  # Input [v, ω]
            #     self.xEst[0:STATE_SIZE] = velocity_motion_model(self.xEst[0:STATE_SIZE], u, dt)
            #     self.get_logger().info(f"Predicted state: {self.xEst}")
                
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array([position - self.last_pose[0:2]]).transpose()
            local_delta = (rot*delta).transpose()
            self.odometry_data = np.array([local_delta[0,0], local_delta[0,1], orientation - self.last_pose[2]])

            self.last_pose = pose
            self.last_stamp = msg.header.stamp

        else:
            self.get_logger().info('...Received first Odometry message')
            self.last_pose = pose
            self.last_stamp = msg.header.stamp
            self.odom_initialized = True
            
        self.xEst, self.PEst = EKF_SLAM_step(self.xEst, self.PEst, self.odometry_data, None)
        # self.get_logger().info(f"Predicted state: {self.xEst}")
        
    def matrixCB(self, msg):
            # Deserialize the JSON data
            matrices_data = json.loads(msg.data)
            self.get_logger().info(f'Received {len(matrices_data)} matrices.')

            for matrix_info in matrices_data:
                matrix_id = matrix_info['id']
                rows = matrix_info['rows']
                cols = matrix_info['cols']
                data = matrix_info['data']

                if rows > 0 and cols > 0:
                    matrix = np.array(data).reshape(rows, cols)
                    self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}\n')
                    # self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}\n{matrix}')
                else:
                    self.get_logger().info(f'Matrix ID: {matrix_id} is a placeholder.')

        
        
            
def EKF_SLAM_step(xEst, PEst, u, z):
    '''
    Main EKF SLAM algorithm
    '''
    #Prediction step
    Gt, Vt, F = compute_dd_motion_model_jacobians(xEst, u)
    xEst[0:STATE_SIZE] = differential_drive_motion_model(xEst[0:STATE_SIZE], u)
    PEst = Gt.T @ PEst @ Gt + Vt.T @ M @ Vt
    # print(f"Predicted state: {xEst}")
    # print(f"Predicted covariance: {PEst}")
    
    #Intermediate matrices to store the predicted state and covariance matrix
    xPredicted = xEst
    pPredicted = PEst
    
    return xEst, PEst
    


def velocity_motion_model(state, u, dt):
    """
    Predicts the new state [x, y, yaw] given the current state, control inputs [v, ω],
    and time interval dt.

    :param state: Current state [x, y, yaw]
    :param u: Control input [v, ω] (linear and angular velocities)
    :param dt: Time step
    :return: Predicted state [x, y, yaw]
    """
    x, y, yaw = state
    v, omega = u

    if abs(omega) > 1e-6:  # Avoid division by zero
        x += v / omega * (np.sin(yaw + omega * dt) - np.sin(yaw))
        y += v / omega * (-np.cos(yaw + omega * dt) + np.cos(yaw))
        yaw += omega * dt
    else:
        x += v * dt * np.cos(yaw)
        y += v * dt * np.sin(yaw)
        yaw += omega * dt  # Still update yaw in case of drift

    return np.array([x, y, yaw])

def differential_drive_motion_model(state, delta_pose):
    """
    Apply the differential drive motion model.

    :param state: Current state [xr, yr, φr]
    :param delta_pose: Local odometry increments [∆xr, ∆yr, ∆φr]
    :return: Predicted state [xr, yr, φr]
    """
    xr, yr, phi_r = state
    delta_xr, delta_yr, delta_phi_r = delta_pose

    # Apply the motion model
    xr_new = xr + delta_xr * np.cos(phi_r) - delta_yr * np.sin(phi_r)
    yr_new = yr + delta_xr * np.sin(phi_r) + delta_yr * np.cos(phi_r)
    phi_r_new = phi_r + delta_phi_r

    return np.array([xr_new, yr_new, phi_r_new])

def compute_dd_motion_model_jacobians(state, delta_pose):
    """
    Compute the Jacobians G and Fx for the motion model.
    
    :param state: Current state [xr, yr, φr, ...landmarks...]
    :param delta_pose: Local odometry increments [∆xr, ∆yr, ∆φr]
    :return: Jacobians G (w.r.t. state) and Fx (w.r.t. control inputs)
    """
    # Extract the robot state
    xr, yr, phi_r = state[:STATE_SIZE]
    delta_xr, delta_yr, delta_phi_r = delta_pose
    
    # Reshaping matrix F to match the dimensions of the state
    F = np.hstack([np.eye(STATE_SIZE), np.zeros((STATE_SIZE,LM_SIZE * calc_n_lm(state)))])

    # Jacobian w.r.t. state (Jg)
    Jg = np.array([
        [1, 0, -delta_xr * np.sin(phi_r) - delta_yr * np.cos(phi_r)],
        [0, 1,  delta_xr * np.cos(phi_r) - delta_yr * np.sin(phi_r)],
        [0, 0, 1]
    ])

    # Jacobian w.r.t. control inputs (Vt)
    Vt = np.array([
        [np.cos(phi_r), -np.sin(phi_r), 0],
        [np.sin(phi_r),  np.cos(phi_r), 0],
        [0, 0, 1]
    ])
    
    Gt = Jg @ F

    return Gt, Vt, F

def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n



def main(args=None):
    rclpy.init(args=args)
    node = EKF_SLAM()  # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
