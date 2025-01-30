import numpy as np
import matplotlib.pyplot as plt

# Load scan parameters and initialize scan data
scanParams = np.loadtxt(f'/home/ruan/dev_ws/src/landmark_extract/Simple_test_data/scanParameters.csv', delimiter=',')
num_scans = int(scanParams[3])
scans = np.zeros((num_scans, 5, 5))

# Load the scan data into each column
for i in range(5):
    scans[:, i] = np.loadtxt(f'/home/ruan/dev_ws/src/landmark_extract/Simple_test_data/scanData_0.csv', delimiter=',')

# Generate the angles for each scan point based on scan parameters
angles = np.arange(scanParams[0], scanParams[1], scanParams[2])

print(scanParams)

# Set up colors for plotting
colors = ['r', 'g', 'b', 'y', 'm']

# Polar plot with points
plt.figure()
for i in range(5):
    plt.polar(angles[0:360], scans[0:360, i, 0], 'o', color=colors[i], markersize=2, label=f'Scan {i+1}')

plt.title('Local Scan (Polar Coordinates)', va='bottom')
plt.gca().set_rlabel_position(-22.5)
plt.gca().set_ylabel('Distance (m)', labelpad=30)
plt.legend()
plt.show()

# Cartesian plot
plt.figure()
for i in range(5):
    x = scans[0:360, i, 0] * np.cos(angles[0:360])  # Convert range and angle to x
    y = scans[0:360, i, 0] * np.sin(angles[0:360])  # Convert range and angle to y
    plt.scatter(x, y, color=colors[i], s=5, label=f'Scan {i+1}')

plt.title('Local Scan (Cartesian Coordinates)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.legend()
plt.axis('equal')
plt.show()
