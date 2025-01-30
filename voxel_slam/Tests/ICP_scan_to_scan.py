import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


class ICPScanToScan:
    def __init__(self, max_iterations=50, tolerance=1e-5, visualize=False):
        """
        Initializes the ICPScanToScan class.

        Args:
            max_iterations (int): Maximum number of ICP iterations.
            tolerance (float): Convergence tolerance for transformation error.
            visualize (bool): Whether to visualize the alignment process.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.visualize = visualize

    @staticmethod
    def nearest_neighbor_association(source, target):
        """Finds the nearest neighbors in the target for each point in the source."""
        tree = KDTree(target)
        distances, indices = tree.query(source)
        return target[indices], distances

    @staticmethod
    def svd_motion_estimation(source, target):
        """Computes the optimal rotation and translation using SVD."""
        # Compute centroids
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)

        # Center the point sets
        source_centered = source - centroid_source
        target_centered = target - centroid_target

        # Compute covariance matrix
        H = source_centered.T @ target_centered

        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        T = centroid_target - R @ centroid_source

        return R, T

    def fit(self, prev_scan, curr_scan, initial_pose):
        """
        Performs ICP to align the current scan to the previous scan.

        Args:
            prev_scan (ndarray): Point cloud from the previous scan, shape (N, 2).
            curr_scan (ndarray): Point cloud from the current scan, shape (M, 2).
            initial_pose (ndarray): Initial guess for the transformation (3x3 matrix).

        Returns:
            R (ndarray): Final rotation matrix (2x2).
            T (ndarray): Final translation vector (2x1).
            transformed_scan (ndarray): Transformed current scan after alignment.
        """
        # Initialize transformation
        H = initial_pose  # Start with initial guess (3x3)
        curr_scan_homo = np.hstack((curr_scan, np.ones((curr_scan.shape[0], 1))))  # Homogeneous coords

        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            # Transform the current scan using the current transformation
            transformed_scan = (H @ curr_scan_homo.T).T[:, :2]

            # Find correspondences between transformed scan and previous scan
            matched_points, distances = self.nearest_neighbor_association(transformed_scan, prev_scan)

            # Compute new transformation using matched points
            R, T = self.svd_motion_estimation(transformed_scan, matched_points)

            # Update the total transformation
            H_update = np.eye(3)
            H_update[:2, :2] = R
            H_update[:2, 2] = T
            H = H_update @ H

            # Check for convergence (change in error)
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

            # Visualize the process
            if self.visualize:
                self._visualize_iteration(prev_scan, transformed_scan, iteration)

        # Extract final R and T from H
        R_final = H[:2, :2]
        T_final = H[:2, 2]

        # Final transformed scan
        transformed_scan = (H @ curr_scan_homo.T).T[:, :2]

        # Final visualization
        if self.visualize:
            self._visualize_final(prev_scan, transformed_scan)

        return R_final, T_final, transformed_scan

    def _visualize_iteration(self, prev_scan, transformed_scan, iteration):
        """Visualizes the alignment process for each iteration."""
        plt.figure(figsize=(8, 6))
        plt.scatter(prev_scan[:, 0], prev_scan[:, 1], c='blue', label='Previous Scan', alpha=0.5)
        plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], c='red', label='Current Scan (Transformed)', alpha=0.5)
        plt.title(f"Iteration {iteration + 1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis('equal')
        plt.show()

    def _visualize_final(self, prev_scan, transformed_scan):
        """Visualizes the final alignment result."""
        plt.figure(figsize=(8, 6))
        plt.scatter(prev_scan[:, 0], prev_scan[:, 1], c='blue', label='Previous Scan', alpha=0.5)
        plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], c='red', label='Current Scan (Transformed)', alpha=0.5)
        plt.title("Final Alignment")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis('equal')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example scans
    prev_scan = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Square as reference
    curr_scan = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])  # Shifted square

    # Initial guess (identity matrix, no transformation)
    initial_pose = np.eye(3)

    # Instantiate ICP class with visualization enabled
    icp = ICPScanToScan(max_iterations=50, tolerance=1e-5, visualize=True)

    # Run ICP for scan-to-scan matching
    R, T, transformed_scan = icp.fit(prev_scan, curr_scan, initial_pose)

    # Output results
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
