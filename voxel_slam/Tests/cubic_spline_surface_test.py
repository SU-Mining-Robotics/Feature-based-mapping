import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cubic_spline_surface import CubicSplineSurface

# Initialize the cubic spline surface
spline_surface = CubicSplineSurface(knot_space=0.5, surface_size=np.array([10., 10.]))

# Extract grid size
grid_x, grid_y = spline_surface.grid_size.flatten()

# Create mesh grid for visualization
x = np.linspace(spline_surface.map_lower_limits[0], spline_surface.map_upper_limits[0], grid_x)
y = np.linspace(spline_surface.map_lower_limits[1], spline_surface.map_upper_limits[1], grid_y)
X, Y = np.meshgrid(x, y)

# Control points reshaped into 2D
Z_original = spline_surface.ctrl_pts.reshape(grid_x, grid_y).copy()

# Modify a specific control point (e.g., increase height at index [5, 5])
spline_surface.ctrl_pts[5 * grid_y + 5] += 0.5  # Increase height
# spline_surface.ctrl_pts[5 * grid_y + 6] -= 1  # Increase height
spline_surface.ctrl_pts[10 * grid_x + 10] -= 1  # Increase height

Z_modified = spline_surface.ctrl_pts.reshape(grid_x, grid_y)

# Plot original and modified surfaces side by side
fig = plt.figure(figsize=(12, 6))

# Original Surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_original, cmap='viridis', edgecolor='k')
ax1.set_title("Original Surface")

# Modified Surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_modified, cmap='viridis', edgecolor='k')
ax2.set_title("Modified Surface (Control Point Changed)")

# Labels
for ax in [ax1, ax2]:
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Control Point Values")

plt.show()
