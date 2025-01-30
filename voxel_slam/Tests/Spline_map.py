import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define control points (x, y)
control_points = np.array([[0, 0], [1, 2], [3, 1], [4, 3]])

# Separate control points into x and y
x = control_points[:, 0]
y = control_points[:, 1]

# Fit a cubic spline to the control points
spline = CubicSpline(x, y)

# Generate a smooth curve between the control points
x_vals = np.linspace(min(x), max(x), 100)
y_vals = spline(x_vals)

# Plot the control points and spline
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Spline")
plt.scatter(x, y, color='red', label="Control Points")
plt.title("Spline-based Occupancy Grid Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
