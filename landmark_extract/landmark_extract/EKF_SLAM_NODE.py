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
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from Test_scipts.Observation_model import SplineLaserPredictor
from Test_scipts.Bspline_data_association_test import SplineDataAssociation
from Spline_map_visualiser import SplineMapVisualiser
from scipy.interpolate import BSpline
import time
import sys
import os

# Add the directory containing the module to sys.path
# This is neccessary since the module is in the same directroy as the node, but we will probably be in the workspace directory when running it
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# EKF state covariance
M = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
M = np.diag([0.5]) 

# EKF measurement covariance of a single lidar beam
R = np.diag([0.1, np.deg2rad(1.0)]) ** 2

STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 2  # Landmark size [x, y]

predictor = SplineLaserPredictor()
data_associator = SplineDataAssociation()

class EKF_SLAM(Node):  # MODIFY NAME
    def __init__(self):
        super().__init__("EKF_SLAM_node")  # MODIFY NAME

        # Declare parameters
        self.declare_parameter('odometry_topic', "/diff_cont/odom")
        self.declare_parameter('measurement_topic', "/measurement_data")

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, self.get_parameter('odometry_topic').value, self.odomCB, 10)
        # self.subscription = self.create_subscription(String, 'pseudo_matrices', self.matrixCB, 10)
        # self.range_bearing_subscriber = self.create_subscription(String, "range_bearing_segments", self.range_bearingCB, 10)
        self.subscription = self.create_subscription(String, self.get_parameter('measurement_topic').value, self.measurement_CB, 10)

        # State initialization
        self.xEst = np.zeros(STATE_SIZE)  # Initial state [x, y, yaw]
        self.PEst = np.eye(STATE_SIZE)  # Initial covariance matrix
        self.odom_initialized = False
        self.last_pose = None
        self.last_stamp = None
        
        #Flags
        self.matrixes_recieved_flag = False
        self.range_bearing_recieved_flag = False
        self.EKF_test = True
        
        # various data containers used in the EKF algorithm
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.measurement_data = []
        self.feature_size_vector = []
        self.current_landmark_ids = -1
        
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
            
        # self.xEst, self.PEst = EKF_SLAM_step(self.xEst, self.PEst, self.odometry_data, None)
        # self.get_logger().info(f"Predicted state: {self.xEst}")
        
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
            
            
        if self.EKF_test == True:
            self.xEst, self.PEst = EKF_SLAM_step(self.xEst, self.PEst, self.odometry_data, self.measurement_data, self.feature_size_vector, self.current_landmark_ids)
            self.EKF_test = False
            
        self.xEst, self.PEst = EKF_SLAM_step(self.xEst, self.PEst, self.odometry_data, self.measurement_data, self.feature_size_vector, self.current_landmark_ids)
        
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
                self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}')
                # self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}\n{matrix}')
            else:
                self.get_logger().info(f'Matrix ID: {matrix_id} is a placeholder.')
                
        self.matrixes_recieved_flag = True
        
    def range_bearingCB(self, msg):
        # Parse the JSON data
        try:
            range_bearing_segments = json.loads(msg.data)
            # Process the received segments
            self.get_logger().info(f"Received range-bearing segments: {range_bearing_segments}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse JSON: {e}")
            
        self.range_bearing_recieved_flag = True

def EKF_SLAM_step(xEst, PEst, u, z, feature_size_vector, landmark_id):
    '''
    Main EKF SLAM algorithm
    '''
    #Prediction step
    Gt, Vt, F = compute_dd_motion_model_jacobians(xEst, u)
    xEst[0:STATE_SIZE] = differential_drive_motion_model(xEst[0:STATE_SIZE], u)
    # PEst = Gt.T @ PEst @ Gt + Vt.T @ M @ Vt
    # print(f"Predicted state: {xEst}")
    # print(f"Predicted covariance: {PEst}")
    
    
    
    #Intermediate matrices to store the predicted state and covariance matrix
    xPredicted = xEst
    pPredicted = PEst
    
    #Update
    for entry in z: #for each obeservation
        matrix_id = entry.get('id')
        rows = entry.get('rows')
        cols = entry.get('cols')
        range_bearing_data = np.array(entry.get('range_bearing_data', []))
        
        #Process if valid feature
        if rows > 0 and cols > 0:
            
            # landmark_id = search_correspond_LM_ID(xEst, PEst, entry)
            new_landmark = True
            
            # Get landmark positions
            
            # for each feature in map
                # Get map landmark popsition
                # Check for for association
            
            # If association, set ID to corresponding landmark
            # If no association, add new landmark
            
            if new_landmark is True:
                print("New LM")
                current_landmark_id = len(feature_size_vector) #Number of landmark +1
                x_coordinates, y_coordinates, feature_data_size = calc_landmark_positions(xEst, entry)
                feature_size_vector.append(feature_data_size)
               
                PAug = calc_augmented_covariance(xEst, PEst, entry) # Need to fix R
                xAug = np.hstack((xEst, x_coordinates, y_coordinates))
                xEst = xAug
                PEst = PAug
                
            #Predict measurement
            z_bar_list, tau_p_list, t_star_list, spline_tangents_list = predict_measurement(xEst, entry, feature_size_vector, current_landmark_id)
            H = calculate_measurement_jacobian(z_bar_list, tau_p_list, t_star_list, spline_tangents_list, feature_size_vector, current_landmark_id)
            # print(f"H shape: {H.shape}")
            # print(f"PEst shape: {PEst.shape}")
            
            # Extract range
            y = range_bearing_data[:, 0] - z_bar_list 
            # print(f"Ranges: {ranges}")
            # print(f"Ranges shape: {ranges.shape}")
            # print(f"Z_bar: {z_bar_list}")
            # print(f"Z_bar shape: {z_bar_list.shape}")
            # print(f"y: {y}")
            # print(f"y shape: {y.shape}")
      
            S = H @ PEst @ H.T # + R_new
            K = (PEst @ H.T) @ np.linalg.inv(S)
            
            print(f'xEst: {xEst}')
            visualiser = SplineMapVisualiser(xEst, feature_size_vector)
            visualiser.plot_splines()
            
            xEst = xEst + (K @ y)
            
            print(f'xEst: {xEst}')
            visualiser = SplineMapVisualiser(xEst, feature_size_vector)
            visualiser.plot_splines()
            
            PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst
            
            
          
               
                    
            # Jacobian of the observation model
            # H = calc_observation_jacobian(xEst, landmark_id)
            # B = calculate_basis_functions(0.0, feature_size_vector[-1], degree = 3)
            # print(f"B shape: {B.shape}")
            
            
            z = 0 #placeholder
            
            # plt.scatter(x_coordinates, y_coordinates)
            # plt.show()
            
            
            xEst = xEst
            # PEst = PAug
        else:
            # self.get_logger().info(f'Matrix ID: {matrix_id} is a placeholder.')
            print(f"Matrix ID: {matrix_id} is a placeholder.")
            
    # visualiser = SplineMapVisualiser(xEst, feature_size_vector)
    # visualiser.plot_splines()

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

def search_correspond_LM_ID(xEst, PEst, z):
    
    return True

def calc_landmark_positions(xEst, entry):
    
    # Extract robot pose
    x_r = xEst[0]  # Robot's x position
    y_r = xEst[1]  # Robot's y position
    phi_r = xEst[2]  # Robot's orientation (yaw)
    
    matrix_id = entry.get('id')
    print(f"Processing matrix ID: {matrix_id}")

    # Extract the transformation matrix Φ from the entry
    rows = entry.get('rows')
    cols = entry.get('cols')
    matrix_data = entry.get('data')
    
    if rows > 0 and cols > 0:
        phi_matrix = np.array(matrix_data).reshape(rows, cols)
        # self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}')
        # self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}\n{matrix}')
    else:
        # self.get_logger().info(f'Matrix ID: {matrix_id} is a placeholder.')
        print(f"Matrix ID: {matrix_id} is a placeholder.")

    # Extract range and bearing data
    range_bearing_data = np.array(entry.get('range_bearing_data', []))
    print(f"Range-bearing data: {range_bearing_data}")  

    # Initialize lists for landmark coordinates
    x_landmarks = []
    y_landmarks = []
    
    # Calculate the position of each landmark in the observation
    for data in range_bearing_data:
        z_p = data[0]  # Range to the landmark
        tau_p = data[1]  # Bearing to the landmark
        
        # Calculate the x and y coordinates
        x_lm = x_r + z_p * np.cos(phi_r + tau_p)
        y_lm = y_r + z_p * np.sin(phi_r + tau_p)
        
        x_landmarks.append(x_lm)
        y_landmarks.append(y_lm)
    
    # Convert lists to numpy arrays
    x_coordinates = np.array(x_landmarks)
    y_coordinates = np.array(y_landmarks)
    
    #Print matrix shape
    # print(f"Matrix shape: {phi_matrix.shape}")
    #Print landmark coordinates shape
    # print(f"Landmark x-coordinates shape: {x_coordinates.shape}")
    
    transformed_x_coordinates = phi_matrix @ x_coordinates
    transformed_y_coordinates = phi_matrix @ y_coordinates
    #Print transformed x-coordinates shape
    print(f"Transformed x-coordinates shape: {transformed_x_coordinates.shape}")
    #Print transformed y-coordinates shape
    print(f"Transformed y-coordinates shape: {transformed_y_coordinates.shape}")
    
    #Get number of rows and columns
    rows = len(transformed_x_coordinates)
    feature_data_size = rows 
   
    return transformed_x_coordinates, transformed_y_coordinates, feature_data_size

def calc_augmented_covariance(xEst, PEst, entry):
    # Extract robot pose
    x_r = xEst[0] 
    y_r = xEst[1]
    phi_r = xEst[2]  
    
    # Extract the transformation matrix Φ from the entry
    matrix_id = entry.get('id')
    rows = entry.get('rows')
    cols = entry.get('cols')
    matrix_data = entry.get('data')
    
    if rows > 0 and cols > 0:
        phi_matrix = np.array(matrix_data).reshape(rows, cols)
        # self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}')
        # self.get_logger().info(f'Matrix ID: {matrix_id}, Shape: {rows}x{cols}\n{matrix}')
    else:
        # self.get_logger().info(f'Matrix ID: {matrix_id} is a placeholder.')
        print(f"Matrix ID: {matrix_id} is a placeholder.")
        
    # Extract range and bearing data
    range_bearing_data = np.array(entry.get('range_bearing_data', []))
        
    # Partial derivatives of the transformation matrix
    
    # Placeholder for the matrix
    rows_x = []
    rows_y = []
    top_diag_list = []
    bottom_diag_list = []

    # Calculate the position of each landmark in the observation
    for data in range_bearing_data:
        z_p = data[0]  # Range to the landmark
        tau_p = data[1]  # Bearing to the landmark
        
        # Calculate mu_p (angle adjustment)
        mu_p = phi_r + tau_p
        
        sin_mu_p = np.sin(mu_p)
        cos_mu_p = np.cos(mu_p)
        
        # Calculate the components for the matrix row
        row_x = [1, 0, -z_p * sin_mu_p]
        row_y = [0, 1, z_p * cos_mu_p]
        
        rows_x.append(row_x)
        rows_y.append(row_y)
        top_diag_list.append(cos_mu_p)
        bottom_diag_list.append(sin_mu_p)

    # Convert the rows into a NumPy matrix
    top_matrix = np.array(rows_x)
    # print(f"Top matrix shape: {top_matrix.shape}")
    bottom_matrix = np.array(rows_y)
    # print(f"Bottom matrix shape: {bottom_matrix.shape}")
    dgs_dxr = np.vstack(((phi_matrix @ top_matrix), (phi_matrix @ bottom_matrix)))
    # print(f"DGs/Dxr shape: {dgs_dxr.shape}")
    
    top_diag_matrix = np.diag(top_diag_list)
    # print(f"Top diag matrix shape: {top_diag_matrix.shape}")
    bottom_diag_matrix = np.diag(bottom_diag_list)
    # print(f"Bottom diag matrix shape: {bottom_diag_matrix.shape}")
    dgs_dz = np.vstack(((phi_matrix @ top_diag_matrix), (phi_matrix @ bottom_diag_matrix)))
    print(f"DGs/Dz shape: {dgs_dz.shape}")
        
    # Jacobians
    # Calculating G_x
    top = np.eye(xEst.shape[0])
    # print(f"Top shape: {top.shape}")
    
    if dgs_dxr.shape[1] == xEst.shape[0]:
        G_x = np.vstack((top, dgs_dxr))     #Building matrix for first landmark
        # print(f"G_x shape: {G_x.shape}")
    else:
        bottom = np.zeros((dgs_dxr.shape[0], xEst.shape[0] - dgs_dxr.shape[1]))
        # print(f"Bottom shape: {bottom.shape}")
        G_x = np.vstack((top, np.hstack((dgs_dxr, bottom))))
        # print(f"G_x shape: {G_x.shape}")

    #Calculating G_z
    G_z = np.vstack((np.zeros((xEst.shape[0], dgs_dz.shape[1])), dgs_dz))

    # Expand M to fit the size of the augmented state vector
    size = G_z.shape[1]
    M = np.diag([0.5] * size)
    
    print(f"G_x shape: {G_x.shape}")
    print(f"PEst shape: {PEst.shape}")
    print(f"G_z shape: {G_z.shape}")
    print(f"M shape: {M.shape}")
    
    First = G_x @ PEst @ G_x.T
    Second = G_z @ M @ G_z.T
    PAug = First #+ Second
    # PAug = G_x @ PEst @ G_x.T + G_z @ M @ G_z.T
    print(f"First shape: {First.shape}")
    print(f"Second shape: {Second.shape}")
    print(f"PAug shape: {PAug.shape}")
    
    return PAug
   
def predict_measurement(xEst, entry, feature_size_vector, landmark_id):
    
    # Extract robot pose
    x_r = xEst[0] 
    y_r = xEst[1]  
    phi_r = xEst[2]  
    pose = np.array([x_r, y_r, phi_r])
    
    # Extract landmark control points
    feature_size = feature_size_vector[landmark_id]
    n = 2 * np.sum(feature_size_vector[:landmark_id]) 
    x_coordinates = xEst[int(n+STATE_SIZE):int(n+STATE_SIZE+feature_size)]
    y_coordinates = xEst[int(n+STATE_SIZE+feature_size):int(n+STATE_SIZE+2*feature_size)]
    control_points = np.array([x_coordinates, y_coordinates])
    
    # Extract the angles relevant to the current spline
    range_bearing_data = np.array(entry.get('range_bearing_data', []))   
    tau_p_list = range_bearing_data[:, 1]
    z , t_stars, tangent_angles = predictor.predict_distances(tau_p_list, pose, control_points.T)
    
    #Visualise imformation
    # print("f.tau_p_list", tau_p_list)
    # print(f"Predicted measurements: {z}")
    # predictor.visualize_lidar_beams(tau_p_list, pose, control_points.T)
    
    return z, tau_p_list, t_stars, tangent_angles

def calculate_measurement_jacobian(z_bar_list, tau_p_list, t_star_list, spline_tangents_list, feature_size_vector, landmark_id):
    
    H = None
    feature_size = feature_size_vector[landmark_id]
    n = 2 * np.sum(feature_size_vector[:landmark_id])  # Sum of values' components before landmark_id
    p = 2 * np.sum(feature_size_vector[landmark_id+1:]) 
    # print(f"tau_p_list:\n {len(tau_p_list)}")
    # print(f"t_star_list:\n {len(t_star_list)}")
    # print(f"T_star:\n {t_star_list}") #First t_start is wierd
    
    for tau_p, t_star, tangent_angle, z_bar in zip(tau_p_list, t_star_list, spline_tangents_list, z_bar_list):
        
        basis_vector_at_t_star = calculate_basis_functions(t_star, feature_size-1, degree = 3)
        # print(f"Basis vector at t_star:\n {basis_vector_at_t_star}")
        h_xi_vec = basis_vector_at_t_star * (np.cos(tau_p) + np.sin(tau_p)/np.tan(tangent_angle-tau_p))
        h_yi_vec = basis_vector_at_t_star * (np.sin(tau_p) - np.cos(tau_p)/np.tan(tangent_angle-tau_p))
        # print(f"h_xi_vec:\n {h_xi_vec}")
        # print(f"h_yi_vec shape:\n {h_xi_vec.shape}")
        
        h_xr = -np.cos(tau_p) - np.sin(tau_p)/np.tan(tangent_angle-tau_p)
        h_yr = -np.sin(tau_p) + np.cos(tau_p)/np.tan(tangent_angle-tau_p)
        h_phi_r = z_bar/np.tan(tangent_angle-tau_p)
        
        # Construct a row in jocobian matrix
        H_row = np.hstack([
            h_xr, h_yr, h_phi_r, 
            *( [np.zeros(n)] if n > 0 else [] ),
            h_xi_vec, h_yi_vec, 
            *( [np.zeros(p)] if p > 0 else [] )
        ])
        # print(f"H_row:\n {H_row}")
        # print(f"H_row shape: {H_row.shape}")
        
        if H is not None:
            H = np.vstack([H, H_row]) 
        else:
            H = H_row
        
    # print(f"H:\n {H}")
    # print(f"H shape: {H.shape}")
    
    return H
    
def calculate_basis_functions(t_star, num_points, degree=3):
    """Calculate the basis function values at a single point t_star."""
    
    n = num_points
    p = degree

    # Generate an open uniform knot vector
    knots = np.concatenate(([0] * (p + 1), np.linspace(0, 1, n - p), [1] * (p + 1)))

    # Number of basis functions
    num_basis = len(knots) - degree - 1

    # Compute the basis function values at t_star
    B = np.zeros(num_basis)
    for i in range(num_basis):
        coeff = np.zeros(num_basis)
        coeff[i] = 1
        basis_function = BSpline(knots, coeff, degree)
        B[i] = basis_function(t_star)  # Evaluate at single t_star
    
    print(B)
    print(f"B shape: {B.shape}")    

    return B  # Returns a 1D array (single row)


    
def main(args=None):
    rclpy.init(args=args)
    node = EKF_SLAM()  # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
