import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.interpolate import BSpline
from matplotlib.patches import Ellipse

class SplineMapVisualiser:
    def __init__(self, state_vector=None, feature_sizes=None, covariance_matrix=None):
        self.state_vector = state_vector
        self.feature_sizes = feature_sizes
        self.covariance_matrix = covariance_matrix
        # self.robot_pose = state_vector[:3]  # (x_r, y_r, theta_r)
        # self.features = self.extract_features()
        self.robot_trajec = []
        
         # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.ion()  # Turn on interactive mode
    
    def extract_features(self):
        """Extracts features (spline control points) from the state vector using feature_sizes."""
        features = []
        idx = 3  # Start after robot pose
        for size in self.feature_sizes:
            x_points = self.state_vector[idx:idx+size]
            y_points = self.state_vector[idx+size:idx+2*size]
            control_points = np.column_stack((x_points, y_points))
            features.append(control_points)
            idx += 2 * size
        return features
    
    @staticmethod
    def bspline_function(control_points, t):
        """Evaluate the cubic B-spline at parameter t."""
        degree = 3  # Cubic B-spline
        n = len(control_points)  # Number of control points
        
        if n <= degree:
            raise ValueError("Not enough control points for a cubic B-spline")
        
        # Knot vector with uniform spacing
        knots = np.concatenate(([0] * degree, np.linspace(0, 1, n - degree + 1), [1] * degree))
        
        # Create B-spline
        spline = BSpline(knots, control_points, degree)
        
        return spline(t)  # Evaluate spline at t
    
    # def plot_covariance_ellipse(self, mean, covariance, ax, edgecolor='r'):
    #     """Plot a covariance ellipse given the mean and covariance matrix."""
    #     if covariance.shape != (2, 2):
    #         return  # Ignore non-2D covariance matrices
        
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
    #     # Compute ellipse properties
    #     angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    #     width, height = 2 * np.sqrt(eigenvalues)  # 1-sigma (68%) confidence
        
    #     ellipse = Ellipse(mean, width, height, np.degrees(angle), edgecolor=edgecolor, facecolor='none', lw=1.5)
    #     ax.add_patch(ellipse)
        
    def plot_covariance_ellipse(self, mean, covariance):
        """Plot a covariance ellipse given the mean and covariance matrix."""
        if covariance.shape != (2, 2):
            return  # Ignore non-2D covariance matrices
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Compute ellipse properties
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = 2 * np.sqrt(eigenvalues)  # 1-sigma (68%) confidence

        ellipse = Ellipse(mean, width, height, np.degrees(angle), edgecolor='g', facecolor='none', lw=1.5)
        self.ax.add_patch(ellipse)
    
    def plot_splines(self):
        """Visualize the robot pose, spline features, and covariance ellipses."""
        fig, ax = plt.subplots(figsize=(8, 6))
       
        # Plot robot pose as an arrow
        ax.quiver(
            self.robot_pose[0], self.robot_pose[1], 
            np.cos(self.robot_pose[2]), 
            np.sin(self.robot_pose[2]), 
            angles='xy', scale_units='xy', scale=1, 
            color='r', width=0.01, label="Robot"
        )
        
        # Plot covariance ellipse for the robot pose
        if self.covariance_matrix is not None:
                idx_offset = 3  # Start after robot pose
                for i, control_points in enumerate(self.features):
                    for j, point in enumerate(control_points):
                        base_idx = idx_offset + j * 2  # Get the correct index for (x, y)
                        covariance_2x2 = self.covariance_matrix[base_idx:base_idx+2, base_idx:base_idx+2]
                        self.plot_covariance_ellipse(point, covariance_2x2, ax, edgecolor='g')
                    idx_offset += 2 * self.feature_sizes[i]  # Move to next feature block

        
        # Plot splines
        t_values = np.linspace(0, 1, 100)
        for i, control_points in enumerate(self.features):
            try:
                spline_curve = np.array([self.bspline_function(control_points, t) for t in t_values])
                ax.plot(control_points[:, 0], control_points[:, 1], 'ko-', label=f'Control Points {i+1}')
                ax.plot(spline_curve[:, 0], spline_curve[:, 1], 'b-', label=f'Spline {i+1}')
            except ValueError as e:
                print(f"Skipping spline {i+1}: {e}")
                continue
            
            # Plot covariance ellipses for feature points
            if self.covariance_matrix is not None:
                for j, point in enumerate(control_points):
                    idx = 3 + sum(self.feature_sizes[:i]) * 2 + j * 2
                    self.plot_covariance_ellipse(point, self.covariance_matrix[idx:idx+2, idx:idx+2], ax, edgecolor='g')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.set_title('Spline Map Visualisation with Covariance Ellipses')
        ax.axis('equal')
        ax.grid()
        plt.show()
        
    def update_plot(self, state_vector, feature_sizes, covariance_matrix=None):
        """Update and redraw the splines based on new inputs."""
        self.state_vector = state_vector
        self.feature_sizes = feature_sizes
        self.covariance_matrix = covariance_matrix
        self.robot_pose = state_vector[:3]
        self.features = self.extract_features()
        
        # Clear previous plot
        self.ax.clear()

        # Plot robot pose as an arrow
        self.ax.quiver(
            self.robot_pose[0], self.robot_pose[1], 
            np.cos(self.robot_pose[2]), 
            np.sin(self.robot_pose[2]), 
            angles='xy', scale_units='xy', scale=1, 
            color='r', width=0.01, label="Robot"
        )
        self.robot_trajec.append(self.robot_pose)
        for i in range(len(self.robot_trajec)-1):
            plt.plot([self.robot_trajec[i][0], self.robot_trajec[i+1][0]], [self.robot_trajec[i][1], self.robot_trajec[i+1][1]], 'g')
        
        # Plot splines
        t_values = np.linspace(0, 1, 100)
        for i, control_points in enumerate(self.features):
            try:
                spline_curve = np.array([self.bspline_function(control_points, t) for t in t_values])
                self.ax.plot(control_points[:, 0], control_points[:, 1], 'ko-', label=f'Control Points {i+1}')
                self.ax.plot(spline_curve[:, 0], spline_curve[:, 1], 'b-', label=f'Spline {i+1}')
            except ValueError as e:
                print(f"Skipping spline {i+1}: {e}")
                continue

            # Plot covariance ellipses for feature points
            if self.covariance_matrix is not None:
                idx_offset = 3 + sum(self.feature_sizes[:i]) * 2
                for j, point in enumerate(control_points):
                    base_idx = idx_offset + j * 2
                    covariance_2x2 = self.covariance_matrix[base_idx:base_idx+2, base_idx:base_idx+2]
                    self.plot_covariance_ellipse(point, covariance_2x2)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()
        self.ax.set_title('Continuous Spline Map Visualization')
        self.ax.axis('equal')
        self.ax.grid()
        plt.pause(0.001)  # Small pause to update the plot without blocking execution


# Example usage:
def main():
    state_vector = np.array([
        2.0, 3.0, np.pi/4,  # Robot pose (x_r, y_r, theta_r)
        1.0, 2.0, 3.0, 4.0,  # Feature 1: x-coordinates
        1.0, 1.5, 2.0, 2.5,  # Feature 1: y-coordinates
        4.0, 5.0, 6.0, 7.0,  # Feature 2: x-coordinates
        3.0, 4.0, 5.0, 6.0   # Feature 2: y-coordinates
    ])
    
    feature_sizes = [4, 4]
    covariance_matrix = np.eye(len(state_vector)) * 0.1  # Example covariance matrix
    
    visualiser = SplineMapVisualiser(state_vector, feature_sizes, covariance_matrix)
    visualiser.plot_splines()

if __name__ == "__main__":
    main()
