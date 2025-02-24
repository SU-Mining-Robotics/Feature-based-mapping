import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Constants
NUM_PARTICLES = 50
MAP_SIZE = 200  # Occupancy grid size
MAP_RESOLUTION = 0.05  # Grid resolution (meters per cell)

class Particle:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0
        self.map = np.zeros((MAP_SIZE, MAP_SIZE))  # Occupancy grid

    def move(self, delta_x, delta_y, delta_theta):
        """Apply motion model with Gaussian noise"""
        self.x += delta_x + np.random.normal(0, 0.1)
        self.y += delta_y + np.random.normal(0, 0.1)
        self.theta += delta_theta + np.random.normal(0, 0.05)

def initialize_particles():
    """Initialize particles randomly in a small area"""
    return [Particle(np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-np.pi, np.pi)) for _ in range(NUM_PARTICLES)]

def transform_scan(scan, particle):
    """Transform Lidar scan to world frame based on particle pose"""
    rot = R.from_euler('z', particle.theta).as_matrix()[:2, :2]
    transformed_scan = (rot @ scan.T).T + np.array([particle.x, particle.y])
    return transformed_scan

def icp_scan_matching(local_scan, global_map):
    """Perform ICP scan matching and return alignment error"""
    # Ensure the scan has (N, 3) shape
    if local_scan.shape[1] == 2:
        local_scan = np.hstack((local_scan, np.zeros((local_scan.shape[0], 1))))  

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(local_scan)

    if global_map.shape[1] == 2:
        global_map = np.hstack((global_map, np.zeros((global_map.shape[0], 1))))  
        
    print(f"global_map type: {type(global_map)}")
    print(f"global_map shape: {global_map.shape}")
    print(f"global_map dtype: {global_map.dtype}")
    print(f"First 5 points of global_map:\n{global_map[:5]}")


    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(global_map)

    threshold = 0.2
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result.fitness  # Higher means better match


def measurement_update(particles, scan, global_map):
    """Update particle weights based on scan matching"""
    for particle in particles:
        transformed_scan = transform_scan(scan, particle)
        match_score = icp_scan_matching(transformed_scan, global_map)
        particle.weight = match_score + 1e-6  # Avoid zero weights

def resample(particles):
    """Resample particles based on weights"""
    weights = np.array([p.weight for p in particles])
    weights /= np.sum(weights)  # Normalize

    indices = np.random.choice(len(particles), size=NUM_PARTICLES, p=weights)
    return [particles[i] for i in indices]

def update_map(particle, scan):
    """Update the particle’s occupancy grid"""
    transformed_scan = transform_scan(scan, particle)
    for point in transformed_scan:
        grid_x, grid_y = int(point[0] / MAP_RESOLUTION), int(point[1] / MAP_RESOLUTION)
        if 0 <= grid_x < MAP_SIZE and 0 <= grid_y < MAP_SIZE:
            particle.map[grid_x, grid_y] += 1  # Mark occupied cells

# ---- Simulation ----
particles = initialize_particles()

# Simulated robot motion (move forward + slight turn)
motion = [(0.2, 0.0, np.pi / 18)] * 20  # (dx, dy, dtheta) for each step

# Simulated Lidar scan (simple 2D point cloud in local frame)
scan = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 100)]) * 3

# Global Map (for visualization)
global_map = np.zeros((MAP_SIZE, MAP_SIZE))

for step, (dx, dy, dtheta) in enumerate(motion):
    # 1. Move particles
    for p in particles:
        p.move(dx, dy, dtheta)

    # 2. Measurement update (scan matching)
    measurement_update(particles, scan, global_map)

    # 3. Resampling
    particles = resample(particles)

    # 4. Map update
    for p in particles:
        update_map(p, scan)

    # Visualization
    plt.clf()
    for p in particles:
        plt.scatter(p.x, p.y, s=5, color='blue', alpha=0.3)
    plt.scatter(0, 0, s=50, color='red', marker="x")  # Initial position
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.pause(0.1)

plt.show()
