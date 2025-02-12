import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

class SplineMapVisualiser:
    def __init__(self, state_vector, feature_sizes):
        self.state_vector = state_vector
        self.feature_sizes = feature_sizes
        self.robot_pose = state_vector[:3]  # (x_r, y_r, theta_r)
        self.features = self.extract_features()
    
    def extract_features(self):
        """Extracts features (spline control points) from the state vector using feature_sizes."""
        features = []
        idx = 3  # Start after robot pose
        for size in self.feature_sizes:
            x_points = self.state_vector[idx:idx+size]
            y_points = self.state_vector[idx+len(x_points):idx+2*size]
            control_points = np.column_stack((x_points, y_points))
            features.append(control_points)
            idx += 2 * size
        return features

    @staticmethod
    def bernstein_basis(n, i, t):
        """Compute the Bernstein basis polynomial of degree n."""
        return comb(n, i) * (1 - t)**(n - i) * t**i

    @staticmethod
    def spline_function(control_points, t):
        """Evaluate a Bézier curve for parameter t."""
        n = len(control_points) - 1  # Degree of the spline
        point = np.zeros(2)  # To store the evaluated point
        for i in range(n + 1):
            bernstein = SplineMapVisualiser.bernstein_basis(n, i, t)
            point += bernstein * control_points[i]
        return point

    def plot_splines(self):
        """Visualize the robot pose and spline features."""
        plt.figure(figsize=(8, 6))
       
        # Plot robot pose as an arrow
        plt.quiver(
            self.robot_pose[0], self.robot_pose[1], 
            np.cos(self.robot_pose[2]), 
            np.sin(self.robot_pose[2]), 
            angles='xy', scale_units='xy', scale=1, 
            color='r', width=0.01, label="Robot"
            )
        

        # Plot splines
        t_values = np.linspace(0, 1, 100)
        for i, control_points in enumerate(self.features):
            spline_curve = np.array([self.spline_function(control_points, t) for t in t_values])
            plt.plot(control_points[:, 0], control_points[:, 1], 'ko-', label=f'Control Points {i+1}')
            plt.plot(spline_curve[:, 0], spline_curve[:, 1], 'b-', label=f'Spline {i+1}')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Spline Map Visualisation')
        plt.axis('equal')
        plt.grid()
        plt.show()

# Example usage:
# state_vector = np.array([x_r, y_r, theta_r, x1_1, x1_2, y1_1, y1_2, x2_1, y2_1])
# feature_sizes = [2, 1]  # Two features, one with 2 control points, another with 1
# visualiser = SplineMapVisualiser(state_vector, feature_sizes)
# visualiser.plot_splines()

def main(args=None):
 # Example state vector:
    # (x_r, y_r, theta_r, x1_1, x1_2, x1_3, y1_1, y1_2, y1_3, x2_1, x2_2, y2_1, y2_2)
    state_vector = np.array([
        2.0, 3.0, np.pi/4,  # Robot pose (x_r, y_r, theta_r)
        1.0, 2.0, 3.0,  # Feature 1: x-coordinates
        1.0, 1.5, 2.0,  # Feature 1: y-coordinates
        4.0, 5.0,  # Feature 2: x-coordinates
        3.0, 4.0   # Feature 2: y-coordinates
    ])

    # Feature sizes: First feature has 3 control points, second feature has 2
    feature_sizes = [3, 2]

    # Create visualizer instance and plot
    visualiser = SplineMapVisualiser(state_vector, feature_sizes)
    visualiser.plot_splines()


if __name__ == "__main__":
    main()