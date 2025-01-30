import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

# === Simulated Global Map ===
def generate_map():
    walls = np.array([
        [0, 0], [10, 0], [10, 10], [0, 10], [0, 0]  # Rectangle
    ])
    return walls

# === Simulated LiDAR Scan ===
def simulate_lidar(pose, map_points, noise_std=0.05):
    x, y, theta = pose
    rotation = R.from_euler('z', theta).as_matrix()[:2, :2]
    translation = np.array([x, y])
    map_local = (map_points - translation) @ rotation.T
    distances = np.linalg.norm(map_local, axis=1)
    angles = np.arctan2(map_local[:, 1], map_local[:, 0])
    distances += np.random.normal(0, noise_std, size=distances.shape)
    return np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))

# === ICP Algorithm ===
def icp(actual_scan, predicted_scan, max_iterations=20, tolerance=1e-4):
    for iteration in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1).fit(predicted_scan)
        distances, indices = nbrs.kneighbors(actual_scan)
        matched_points = predicted_scan[indices.flatten()]
        actual_mean = np.mean(actual_scan, axis=0)
        matched_mean = np.mean(matched_points, axis=0)
        H = (actual_scan - actual_mean).T @ (matched_points - matched_mean)
        U, _, Vt = np.linalg.svd(H)
        R_icp = U @ Vt
        t_icp = matched_mean - R_icp @ actual_mean
        actual_scan = actual_scan @ R_icp.T + t_icp
        error = np.linalg.norm(distances)
        if error < tolerance:
            break
    return R_icp, t_icp, error

# === Visualization Helpers ===
def plot_pose(pose, label, color, scale=0.5):
    x, y, theta = pose
    dx = scale * np.cos(theta)
    dy = scale * np.sin(theta)
    plt.arrow(x, y, dx, dy, color=color, head_width=0.3, length_includes_head=True, label=label)

def visualize_icp(global_map, actual_scan, predicted_scan, corrected_scan, true_pose, estimated_pose, corrected_pose):
    plt.figure(figsize=(12, 12))
    
    # Global map
    plt.plot(global_map[:, 0], global_map[:, 1], 'k-', label="Global Map")

    # Robot poses
    plot_pose(true_pose, "True Pose", "green")
    plot_pose(estimated_pose, "Initial Estimated Pose", "blue")
    plot_pose(corrected_pose, "Corrected Pose (ICP)", "red")

    # Scans
    plt.scatter(actual_scan[:, 0], actual_scan[:, 1], label="Actual Scan (LiDAR)", color="orange", alpha=0.7)
    plt.scatter(predicted_scan[:, 0], predicted_scan[:, 1], label="Predicted Scan (From Pose)", color="purple", alpha=0.7)
    plt.scatter(corrected_scan[:, 0], corrected_scan[:, 1], label="Corrected Scan (ICP Aligned)", color="cyan", alpha=0.7)

    # Formatting
    plt.title("ICP Scan Matching Visualization")
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.grid()

# === Main Simulation ===
def main():
    global_map = generate_map()
    true_pose = np.array([5.0, 3.0, np.pi / 6])
    estimated_pose = np.array([5.2, 3.1, np.pi / 6 + 0.1])
    actual_scan = simulate_lidar(true_pose, global_map)
    predicted_scan = simulate_lidar(estimated_pose, global_map, noise_std=0)
    R_icp, t_icp, error = icp(actual_scan, predicted_scan)
    corrected_scan = actual_scan @ R_icp.T + t_icp
    corrected_pose = estimated_pose + np.array([t_icp[0], t_icp[1], np.arctan2(R_icp[1, 0], R_icp[0, 0])])
    print("True Pose:", true_pose)
    print("Initial Estimated Pose:", estimated_pose)
    print("Corrected Pose (after ICP):", corrected_pose)
    print("Alignment Error:", error)
    visualize_icp(global_map, actual_scan, predicted_scan, corrected_scan, true_pose, estimated_pose, corrected_pose)
    plt.show()

# Run the simulation
main()
