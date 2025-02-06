import numpy as np

from utils import wrapAngle, normalDistribution

class MotionModel(object):
    def __init__(self, config):
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.alpha3 = config['alpha3']
        self.alpha4 = config['alpha4']

    def sample_motion_model(self, prev_odo, curr_odo, prev_pose):
        rot1 = np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2]
        rot1 = wrapAngle(rot1)
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        rot2 = curr_odo[2] - prev_odo[2] - rot1
        rot2 = wrapAngle(rot2)

        rot1 = rot1 - np.random.normal(0, self.alpha1 * rot1 ** 2 + self.alpha2 * trans ** 2)
        rot1 = wrapAngle(rot1)
        trans = trans - np.random.normal(0, self.alpha3 * trans ** 2 + self.alpha4 * (rot1 ** 2 + rot2 ** 2))
        rot2 = rot2 - np.random.normal(0, self.alpha1 * rot2 ** 2 + self.alpha2 * trans ** 2)
        rot2 = wrapAngle(rot2)

        x = prev_pose[0] + trans * np.cos(prev_pose[2] + rot1)
        y = prev_pose[1] + trans * np.sin(prev_pose[2] + rot1)
        theta = prev_pose[2] + rot1 + rot2

        return (x, y, theta)
    
    def motion_model(self, prev_odo, curr_odo, prev_pose, curr_pose):
        rot1 = np.arctan2(curr_odo[1] - prev_odo[1], curr_odo[0] - prev_odo[0]) - prev_odo[2]
        rot1 = wrapAngle(rot1)
        trans = np.sqrt((curr_odo[0] - prev_odo[0]) ** 2 + (curr_odo[1] - prev_odo[1]) ** 2)
        rot2 = curr_odo[2] - prev_odo[2] - rot1
        rot2 = wrapAngle(rot2)

        rot1_prime = np.arctan2(curr_pose[1] - prev_pose[1], curr_pose[0] - prev_pose[0]) - prev_pose[2]
        rot1_prime = wrapAngle(rot1_prime)
        trans_prime = np.sqrt((curr_pose[0] - prev_pose[0]) ** 2 + (curr_pose[1] - prev_pose[1]) ** 2)
        rot2_prime = curr_pose[2] - prev_pose[2] - rot1_prime
        rot2_prime = wrapAngle(rot2_prime)
        
        p1 = normalDistribution(wrapAngle(rot1 - rot1_prime), self.alpha1 * rot1_prime ** 2 + self.alpha2 * trans_prime ** 2)
        p2 = normalDistribution(trans - trans_prime, self.alpha3 * trans_prime ** 2 + self.alpha4 * (rot1_prime ** 2 + rot2_prime ** 2))
        p3 = normalDistribution(wrapAngle(rot2 - rot2_prime), self.alpha1 * rot2_prime ** 2 + self.alpha2 * trans_prime ** 2)

        return p1 * p2 * p3
    
    def simple_velocity_motion_model(state, u, dt):
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