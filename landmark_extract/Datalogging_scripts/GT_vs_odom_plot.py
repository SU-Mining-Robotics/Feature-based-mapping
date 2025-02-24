import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
data = np.loadtxt('/home/ruan/dev_ws/src/landmark_extract/GT_vs_Odom.csv', delimiter=',', skiprows=1)

# Extract ground truth and odometry data
gt_x = data[:, 2]  # True X coordinates (ground truth)
gt_y = data[:, 3]  # True Y coordinates (ground truth)
gt_w = data[:, 5]  # True W (orientation) - from ground truth

odom_x = data[:, 6]  # Odometry X coordinates
odom_y = data[:, 7]  # Odometry Y coordinates
odom_w = data[:, 9]  # Odometry W (orientation)

# Plot the paths
plt.figure(figsize=(10, 6))

# Plot ground truth path
plt.plot(gt_x, gt_y, label='Ground Truth Path', color='blue', marker='o', markersize=3, linestyle='-', alpha=0.7)

# Plot odometry path
plt.plot(odom_x, odom_y, label='Odometry Path', color='red', marker='x', markersize=3, linestyle='-', alpha=0.7)

# Adding orientation as arrows
for i in range(0, len(gt_x), 20):  # Sample every 20th point for clarity
    plt.arrow(gt_x[i], gt_y[i], 0.1*np.cos(gt_w[i]), 0.1*np.sin(gt_w[i]), color='blue', head_width=0.1, head_length=0.1)
    plt.arrow(odom_x[i], odom_y[i], 0.1*np.cos(odom_w[i]), 0.1*np.sin(odom_w[i]), color='red', head_width=0.1, head_length=0.1)

# Labels and title
plt.title("Ground Truth vs Odometry Path with Orientation")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
