import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class BezierCurveFitter:
    def __init__(self, lidar_segments, segment_length=None):
        """
        Initialize the BezierCurveFitter with a list of lidar segments.
        Each segment should be a list of points (numpy arrays of shape (N, 2)).
        If segment_length is specified, lidar segments will be subdivided accordingly.
        """
        self.original_segments = lidar_segments
        self.segment_length = segment_length
        self.lidar_segments = self.subdivide_segments() if segment_length else lidar_segments
        self.bezier_curves = []
        self.control_points = []
        self.centroids = []

    @staticmethod
    def cubic_bezier(p0, p1, p2, p3, num_points=100):
        """Calculate a cubic Bezier curve from four control points."""
        t = np.linspace(0, 1, num_points)[:, None]
        bezier = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        return bezier

    @staticmethod
    def bezier_loss(control_points, lidar_points, p0, p3):
        """Loss function to minimize the distance between lidar points and the Bezier curve."""
        p1, p2 = np.array(control_points[:2]), np.array(control_points[2:])
        bezier = BezierCurveFitter.cubic_bezier(p0, p1, p2, p3, num_points=len(lidar_points))
        loss = np.sum(np.linalg.norm(bezier - lidar_points, axis=1)**2)
        return loss

    def subdivide_segments(self):
        """Subdivide lidar segments into smaller segments of specified length."""
        subdivided_segments = []
        for segment in self.original_segments:
            distances = np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1))
            cumulative_distance = np.cumsum(distances)
            start_idx = 0
            while start_idx < len(segment) - 1:
                end_idx = np.searchsorted(cumulative_distance, cumulative_distance[start_idx] + self.segment_length)
                end_idx = min(end_idx, len(segment) - 1)  # Ensure index doesn't exceed segment length
                subdivided_segments.append(segment[start_idx:end_idx + 1])
                start_idx = end_idx
        return subdivided_segments

    def fit_bezier_to_lidar(self, lidar_segment):
        """Fit a Bezier curve to a lidar segment."""
        p0, p3 = lidar_segment[0], lidar_segment[-1]
        initial_guess = np.concatenate([p0, p3])
        result = minimize(self.bezier_loss, initial_guess, args=(lidar_segment, p0, p3), method='BFGS')
        p1, p2 = np.array(result.x[:2]), np.array(result.x[2:])
        return p0, p1, p2, p3

    def fit_all_segments(self):
        """
        Fit Bezier curves to all lidar segments and return:
        - A list of Bezier curves
        - A list of control points for each segment
        - A list of centroids of control points for each segment
        """
        bezier_curves = []
        control_points_list = []
        centroids_list = []

        for segment in self.lidar_segments:
            control_points = self.fit_bezier_to_lidar(segment)
            bezier_curve = self.cubic_bezier(*control_points)
            centroid = np.mean(control_points, axis=0)

            self.bezier_curves.append(bezier_curve)
            self.control_points.append(control_points)
            self.centroids.append(centroid)

            bezier_curves.append(bezier_curve)
            control_points_list.append(control_points)
            centroids_list.append(centroid)

        return bezier_curves, control_points_list, centroids_list

    def visualize(self):
        """Visualize lidar segments, fitted Bezier curves, control points, and centroids."""
        plt.figure(figsize=(10, 6))

        for i, (lidar_segment) in enumerate(self.lidar_segments):
            bezier_curve = self.bezier_curves[i]
            control_points = self.control_points[i]
            centroid = self.centroids[i]

            # Plot lidar segment points
            plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'o', markersize=5, label=f'Segment {i+1} Points')

            # Plot Bezier curve
            plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], '-', linewidth=2, label=f'Bezier Segment {i+1}')

            # Plot control points
            control_points_array = np.array(control_points)
            plt.plot(control_points_array[:, 0], control_points_array[:, 1], 'kx-', markersize=8, label=f'Control Points {i+1}')

            # Plot centroid of control points
            plt.plot(centroid[0], centroid[1], 'r*', markersize=10, label=f'Centroid {i+1}')

        plt.title('Cubic Bezier Curves Fitted to Lidar Segments with Control Point Centroids')
        plt.legend()
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

    def visualize_continues(self):
        """Continuously visualize lidar segments, fitted Bezier curves, control points, and centroids."""
        plt.clf()
        for i, (lidar_segment) in enumerate(self.lidar_segments):
            bezier_curve = self.bezier_curves[i]
            control_points = self.control_points[i]
            centroid = self.centroids[i]

            # Plot lidar segment points
            plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'o', markersize=5, label=f'Segment {i+1} Points')

            # Plot Bezier curve
            plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], '-', linewidth=2, label=f'Bezier Segment {i+1}')

            # Plot control points
            control_points_array = np.array(control_points)
            plt.plot(control_points_array[:, 0], control_points_array[:, 1], 'kx-', markersize=8, label=f'Control Points {i+1}')

            # Plot centroid of control points
            plt.plot(centroid[0], centroid[1], 'r*', markersize=10, label=f'Centroid {i+1}')

        plt.title('Cubic Bezier Curves Fitted to Lidar Segments with Control Point Centroids')
        plt.legend()
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot


# Example usage
def main():
    lidar_segments = [
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),  # Example straight segment
        np.array([[4, 0], [5, 0.5], [5.5, 1], [5.9, 1.5], [6, 2]])  # Example curved segment
    ]

    bezier_fitter = BezierCurveFitter(lidar_segments, segment_length=2)
    bezier_curves, control_points, centroids = bezier_fitter.fit_all_segments()
    bezier_fitter.visualize()


if __name__ == "__main__":
    main()
