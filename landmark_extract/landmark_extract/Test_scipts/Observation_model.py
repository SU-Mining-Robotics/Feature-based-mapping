import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

def rotate_and_translate(control_points, laser_angle, robot_pose):
    """Transform control points of the spline to the laser beam's local frame."""
    # Robot pose: [xr, yr, theta] (position and orientation in global frame)
    xr, yr, theta = robot_pose
    # Laser beam angle relative to the robot's orientation
    mu_p = theta + laser_angle

    # Rotation matrix for aligning with the laser beam
    rotation_matrix = np.array([[np.cos(mu_p), np.sin(mu_p)],
                                 [-np.sin(mu_p), np.cos(mu_p)]])
    # Translate and rotate control points
    transformed_points = []
    for x, y in control_points:
        translated = np.array([x - xr, y - yr])
        transformed = rotation_matrix @ translated
        transformed_points.append(transformed)

    return np.array(transformed_points)

def spline_function(control_points, t):
    """Evaluate the spline function at parameter t."""
    n = len(control_points) - 1  # Degree of the spline
    sx = sum((1 - t) ** (n - i) * t ** i * control_points[i][0] for i in range(n + 1))
    sy = sum((1 - t) ** (n - i) * t ** i * control_points[i][1] for i in range(n + 1))
    return np.array([sx, sy])

def sy_function(t, control_points):
    """Extract the y-component of the spline at parameter t."""
    return spline_function(control_points, t)[1]

def predict_measurement(control_points, laser_angle, robot_pose):
    """Predict the laser measurement for a single laser beam."""
    # Transform the spline to the laser beam's frame
    transformed_points = rotate_and_translate(control_points, laser_angle, robot_pose)

    # Define a function for finding the root (sy = 0)
    def sy_root(t):
        return sy_function(t, transformed_points)

    # Use Newton-Raphson to find the root (t*)
    t_star = newton(sy_root, 0.5)  # Start with an initial guess (e.g., t = 0.5)

    # Predicted measurement (x-component at t*)
    predicted_distance = spline_function(transformed_points, t_star)[0]
    return predicted_distance, t_star, transformed_points

def visualize_prediction(control_points, laser_angle, robot_pose):
    """Visualize the spline, robot, laser beam, and predicted intersection in two frames."""

    # Predict the laser measurement
    predicted_distance, t_star, transformed_points = predict_measurement(control_points, laser_angle, robot_pose)

    # Generate points along the spline for plotting
    t_values = np.linspace(0, 1, 100)  # Parameter values for spline
    original_spline_points = np.array([spline_function(control_points, t) for t in t_values])
    transformed_spline_points = np.array([spline_function(transformed_points, t) for t in t_values])

    # Laser beam in global frame
    laser_origin = np.array([robot_pose[0], robot_pose[1]])  # Robot position
    laser_direction = np.array([np.cos(laser_angle + robot_pose[2]), np.sin(laser_angle + robot_pose[2])])
    laser_end = laser_origin + 5 * laser_direction  # Extend laser for visualization

    # Laser beam in the local frame
    laser_line_x = [0, predicted_distance]
    laser_line_y = [0, 0]

    # --- Plot 1: Original Spline in Global Frame ---
    plt.figure(figsize=(8, 6))
    plt.plot(original_spline_points[:, 0], original_spline_points[:, 1], label="Original Spline", color="blue")
    plt.scatter(control_points[:, 0], control_points[:, 1], color="green", label="Control Points")
    plt.plot([laser_origin[0], laser_end[0]], [laser_origin[1], laser_end[1]], 
             label="Laser Beam", color="orange", linestyle="--")
    plt.title("Original Spline and Laser Beam in Global Frame")
    plt.xlabel("x (global frame)")
    plt.ylabel("y (global frame)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # --- Plot 2: Transformed Spline in Local Frame ---
    plt.figure(figsize=(8, 6))
    plt.plot(transformed_spline_points[:, 0], transformed_spline_points[:, 1], 
             label="Transformed Spline", color="blue")
    plt.plot(laser_line_x, laser_line_y, label="Laser Beam", color="orange", linestyle="--")
    plt.scatter(predicted_distance, 0, color="red", label="Intersection Point", zorder=5)
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color="green", label="Transformed Control Points")
    plt.axhline(0, color="gray", linestyle=":", linewidth=0.5)  # Horizontal line for reference
    plt.title("Transformed Spline and Laser Beam in Local Frame")
    plt.xlabel("x (local frame)")
    plt.ylabel("y (local frame)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()

# Example usage
control_points = np.array([[0, 0], [1, 1], [2, 1], [3, 0]])  # Example spline control points
laser_angle = np.radians(45)  # Laser beam angle (e.g., 45 degrees)
robot_pose = [0, 0, np.radians(0)]  # Robot at origin, facing along x-axis

visualize_prediction(control_points, laser_angle, robot_pose)
