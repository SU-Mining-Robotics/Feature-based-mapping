import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
import tf_transformations


def wrapAngle(radian):
    radian = radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))
    return radian

def pose_to_transform_matrix(x, y, z, yaw):
    """Generate a 4x4 transformation matrix from position (x, y, z) and yaw."""
    # Create a 4x4 transformation matrix
    transform_matrix = np.eye(4)

    # Translation (x, y, z)
    transform_matrix[0, 3] = x
    transform_matrix[1, 3] = y
    transform_matrix[2, 3] = z

    # Rotation (yaw)
    cos_yaw = cos(yaw)
    sin_yaw = sin(yaw)

    # 2D rotation matrix (for yaw)
    transform_matrix[0, 0] = cos_yaw
    transform_matrix[0, 1] = -sin_yaw
    transform_matrix[1, 0] = sin_yaw
    transform_matrix[1, 1] = cos_yaw

    return transform_matrix

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf_transformations.euler_from_quaternion((x, y, z, w))
    return yaw

def wrapAngle(radian):
    radian = radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))
    return radian

def normalDistribution(mean, variance):
    return np.exp(-(np.power(mean, 2) / variance / 2.0) / np.sqrt(2.0 * np.pi * variance))
