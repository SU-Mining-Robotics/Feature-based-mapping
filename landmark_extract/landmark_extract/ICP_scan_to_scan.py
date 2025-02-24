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
        
        source = np.asarray(source)
        target = np.asarray(target)

        # Ensure both source and target are 2D (N, 2)
        if source.shape[1] != 2 or target.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) shape for source and target, got {source.shape} and {target.shape}")
        
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

    def icp_scanmatching(self, target, source, initial_pose):
        """
        Performs ICP to align the source point cloud to the target point cloud.

        Args:
            target (ndarray): Reference point cloud (N, 2).
            source (ndarray): Point cloud to be aligned (M, 2).
            initial_pose (ndarray): Initial guess for the transformation (3x3 matrix).

        Returns:
            H_final (ndarray): Final total transformation matrix (3x3).
            H_correction (ndarray): Correction transformation from initial to final pose (3x3).
            transformed_source (ndarray): Transformed source point cloud after alignment.
        """
        # Initialize transformation
        H = initial_pose  # Start with initial guess (3x3)
        source_homo = np.hstack((source, np.ones((source.shape[0], 1))))  # Homogeneous coords

        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            # Transform the source using the current transformation
            transformed_source = (H @ source_homo.T).T[:, :2]

            # Find correspondences between transformed source and target
            matched_points, distances = self.nearest_neighbor_association(transformed_source, target)

            # Compute new transformation using matched points
            R, T = self.svd_motion_estimation(transformed_source, matched_points)

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
                self._visualize_iteration(target, transformed_source, iteration)

        # Compute the correction transformation
        # print("Initial pose matrix:\n", initial_pose)
        # print("Determinant:", np.linalg.det(initial_pose))
        H_correction = H @ np.linalg.inv(initial_pose)
        # print("Correction matrix:\n", H_correction)
        # print("Determinant:", np.linalg.det(H_correction))
        

        # Final transformed source
        transformed_source = (H @ source_homo.T).T[:, :2]

        # Final visualization
        if self.visualize:
            self._visualize_final(target, transformed_source)

        return H, H_correction, transformed_source

    def icp_scanmatching_map(self, target, source, initial_pose):
        """
        Performs ICP to align the source point cloud to the target point cloud.

        Args:
            target (ndarray): Reference point cloud (N, 2).
            source (ndarray): Point cloud to be aligned (M, 2).
            initial_pose (ndarray): Initial guess for the transformation (3x3 matrix).

        Returns:
            H_final (ndarray): Final total transformation matrix (3x3).
            H_correction (ndarray): Correction transformation from initial to final pose (3x3).
            transformed_source (ndarray): Transformed source point cloud after alignment.
            final_error (float): Mean squared error of the final alignment.
        """
        # Initialize transformation
        H = initial_pose  # Start with initial guess (3x3)
        source_homo = np.hstack((source, np.ones((source.shape[0], 1))))  # Homogeneous coords

        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            # Transform the source using the current transformation
            transformed_source = (H @ source_homo.T).T[:, :2]

            # Find correspondences between transformed source and target
            matched_points, distances = self.nearest_neighbor_association(transformed_source, target)

            # Compute new transformation using matched points
            R, T = self.svd_motion_estimation(transformed_source, matched_points)

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
                self._visualize_iteration(target, transformed_source, iteration)

        # Compute the correction transformation
        H_correction = H @ np.linalg.inv(initial_pose)
        # print("Correction matrix:\n", H_correction)
        # print("Determinant:", np.linalg.det(H_correction))

        # Final transformed source
        transformed_source = (H @ source_homo.T).T[:, :2]

        # Compute final mean squared error
        _, final_distances = self.nearest_neighbor_association(transformed_source, target)
        final_error = np.mean(final_distances)  # Mean squared error

        # Final visualization
        if self.visualize:
            self._visualize_final(target, transformed_source)

        return H, H_correction, transformed_source, final_error

    
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
