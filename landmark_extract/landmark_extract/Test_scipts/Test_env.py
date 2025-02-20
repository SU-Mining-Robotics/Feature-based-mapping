import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to calculate curvature at each point of a 2D curve using second derivative
def curvature(points):
    x = points[:, 0]
    y = points[:, 1]
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curv = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curv

# Function to align two curves based on curvature matching
def curvature_matching_error(params, curve1, curve2):
    dx, dy, dtheta = params
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
    
    # Apply rotation and translation to the second curve
    transformed_curve2 = (curve2 @ R.T) + np.array([dx, dy])
    
    # Compute curvatures of both curves
    curv1 = curvature(curve1)
    curv2 = curvature(transformed_curve2)
    
    # Minimize the difference in curvatures
    error = np.sum((curv1 - curv2) ** 2)
    return error

# Define two curves (a sine curve and a rotated/shifted sine curve)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
curve1 = np.vstack([x, y1]).T

# Create a transformed version of curve1 (shifted and rotated)
theta = np.radians(30)  # Rotation angle
translation = np.array([1, 0.5])
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
curve2 = (curve1 @ R.T) + translation

# Optimize the alignment (translation + rotation)
initial_guess = [0, 0, 0]  # Initial guess for dx, dy, dtheta
result = minimize(curvature_matching_error, initial_guess, args=(curve1, curve2), method="BFGS")
dx_opt, dy_opt, dtheta_opt = result.x

# Apply the optimal transformation
R_opt = np.array([[np.cos(dtheta_opt), -np.sin(dtheta_opt)], [np.sin(dtheta_opt), np.cos(dtheta_opt)]])
aligned_curve2 = (curve2 @ R_opt.T) + np.array([dx_opt, dy_opt])

# Plot the original and aligned curves
plt.plot(curve1[:, 0], curve1[:, 1], 'b-', label="Original Curve")

plt.plot(aligned_curve2[:, 0], aligned_curve2[:, 1], 'g-', label="Aligned Curve (After Matching)")
plt.plot(curve2[:, 0], curve2[:, 1], 'r--', label="Transformed Curve (Before Matching)")
plt.legend()
plt.show()
