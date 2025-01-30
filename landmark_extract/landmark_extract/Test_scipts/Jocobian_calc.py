import numpy as np

STATE_SIZE = 3
LANDMARK_SIZE = 2
Number_of_landmarks = 2

F = np.hstack([np.eye(STATE_SIZE), np.zeros((STATE_SIZE, LANDMARK_SIZE * Number_of_landmarks))])


def compute_dd_motion_model_jacobians(state, delta_pose):
    """
    Compute the Jacobians G and Fx for the motion model.
    
    :param state: Current state [xr, yr, φr]
    :param delta_pose: Local odometry increments [∆xr, ∆yr, ∆φr]
    :return: Jacobians G (w.r.t. state) and Fx (w.r.t. control inputs)
    """
    xr, yr, phi_r = state
    delta_xr, delta_yr, delta_phi_r = delta_pose

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
    
    # Reshaping matrix F to match the dimensions of the state
    F = np.hstack([np.eye(STATE_SIZE), np.zeros((STATE_SIZE, LANDMARK_SIZE * Number_of_landmarks))])

    return Jg, Vt




if __name__ == "__main__":
    print(F)


