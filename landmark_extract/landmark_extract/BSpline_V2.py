import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import logging

logging.basicConfig(level=logging.WARNING)

class BSplineFitter:
    def __init__(self, lidar_segments):
        """
        Initialize the BSplineFitter with a list of lidar segments.
        Each segment should be a list of points (numpy arrays of shape (N, 2)).
        """
        self.lidar_segments = lidar_segments
        self.bspline_curves = []
        self.knot_points = []
        self.control_points = []
        self.centroids = []


    def fit_bspline_to_lidar(self, lidar_segment, knot_distance=1.0, smoothness=0):
        """
        Fit a B-spline to a lidar segment with a specified knot distance.
        """
        x, y = lidar_segment[:, 0], lidar_segment[:, 1]

        # Interpolate to ensure the required knot distance
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)
        num_points = int(total_distance / knot_distance) + 1
        interp_param = np.linspace(0, 1, num_points)
        interp_x = np.interp(interp_param, np.linspace(0, 1, len(x)), x)
        interp_y = np.interp(interp_param, np.linspace(0, 1, len(y)), y)

        # Check if the number of points is sufficient to fit a spline
        num_points = len(interp_x)
        if num_points < 2:
            logging.warning(f"Skipping segment: not enough points ({num_points}) to fit a B-spline.")
            return None, None, None, None

        
        spline_degree = max(1, min(3, num_points - 1))  # Ensure 1 <= k <= 3

        # Fit the B-spline using splprep
        tck, u = splprep([interp_x, interp_y], s=smoothness, k=spline_degree)

        # Generate the B-spline curve
        spline_points = splev(np.linspace(0, 1, 100), tck)
        bspline_curve = np.vstack(spline_points).T

        # Evaluate knot positions
        knot_positions = splev(tck[0], tck)  # Evaluate spline at knot parameter values
        knot_points = np.vstack(knot_positions).T

        # Extract control points
        control_points = np.array(tck[1]).T

        # Compute the centroid of the control points
        centroid = np.mean(control_points, axis=0)

        return bspline_curve, knot_points, control_points, centroid



    def fit_all_segments(self, knot_distance=1.0, smoothness=0):
        """
        Fit B-splines to all lidar segments and store:
        - A list of B-spline curves
        - A list of control points for each segment
        - A list of centroids of control points for each segment
        """
        bspline_curves = []
        knot_points_list = []
        control_points_list = []
        centroids_list = []

        for segment in self.lidar_segments:
            bspline_curve, knot_points, control_points, centroid = self.fit_bspline_to_lidar(
                segment, knot_distance, smoothness
            )
            
            if bspline_curve is not None:
                # Store results
                self.bspline_curves.append(bspline_curve)
                self.knot_points.append(knot_points)
                self.control_points.append(control_points)
                self.centroids.append(centroid)
                
                bspline_curves.append(bspline_curve)
                knot_points_list.append(knot_points)
                control_points_list.append(control_points)
                centroids_list.append(centroid)

        return bspline_curves, knot_points_list, control_points_list, centroids_list

    def visualize(self):
        """Visualize lidar segments, fitted B-spline curves, control points, and centroids."""
        plt.figure(figsize=(10, 6))

        for i, lidar_segment in enumerate(self.lidar_segments):
            bspline_curve = self.bspline_curves[i]
            knot_points = self.knot_points[i]
            control_points = self.control_points[i]
            centroid = self.centroids[i]

            # Plot lidar segment points with transparency
            plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'o', markersize=5, alpha=0.5, label=f'Segment {i+1} Points')

            # Plot B-spline curve
            plt.plot(bspline_curve[:, 0], bspline_curve[:, 1], '-', linewidth=2, label=f'B-spline Segment {i+1}')
            
            # Plot knot points
            plt.plot(knot_points[:, 0], knot_points[:, 1], 'rx-', markersize=8, label=f'Knot Points {i+1}')

            # Plot control points
            plt.plot(control_points[:, 0], control_points[:, 1], 'kx-', markersize=8, label=f'Control Points {i+1}')

        # Plot centroid of control points
        plt.plot(centroid[0], centroid[1], 'r*', markersize=10, label=f'Centroid {i+1}')
        plt.title('B-Spline Curves Fitted to Lidar Segments with Control Point Centroids')
        plt.legend()
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

            
    def visualize_continues(self):
        """
        Visualize lidar segments, fitted B-spline curves, control points, and centroids.
        Skips visualization for segments without valid B-spline data.
        """
        plt.clf()

        # Find the minimum length across all attributes to avoid index out of range errors
        min_length = min(
            len(self.lidar_segments),
            len(self.bspline_curves),
            len(self.knot_points),
            len(self.control_points),
            len(self.centroids),
        )

        # Check if there is any valid data to plot
        if min_length == 0:
            print("No valid segments to visualize.")
            return

        for i in range(min_length):
            try:
                # Get the data for the current segment
                lidar_segment = self.lidar_segments[i]
                bspline_curve = self.bspline_curves[i]
                knot_points = self.knot_points[i]
                control_points = self.control_points[i]
                centroid = self.centroids[i]

                # Plot lidar segment points with transparency (background layer)
                plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'o', markersize=5, alpha=0.7, label=f'Segment {i+1} Points')

                # # Plot B-spline curve (foreground layer)
                # plt.plot(bspline_curve[:, 0], bspline_curve[:, 1], '-', linewidth=2, label=f'B-spline Segment {i+1}')

                # Plot knot points (foreground layer)
                plt.plot(knot_points[:, 0], knot_points[:, 1], 'rx-', markersize=8, label=f'Knot Points {i+1}')

                # Plot control points (foreground layer)
                plt.plot(control_points[:, 0], control_points[:, 1], 'kx-', markersize=8, label=f'Control Points {i+1}')

                # Plot centroid of control points (foreground layer)
                # plt.plot(centroid[0], centroid[1], 'r*', markersize=10, label=f'Centroid {i+1}')

            except IndexError:
                # Handle cases where data might be missing for a segment
                print(f"Skipping segment {i+1}: incomplete data.")
                # plt.clf()
                continue

        # Plot an arrow at the origin
        plt.arrow(0, 0, 0.3, 0, head_width=0.15, head_length=0.3, fc='k', ec='k')

        # Customize plot appearance
        plt.title('B-Spline Curves Fitted to Lidar Segments with Control Point Centroids')
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
        np.array([[4, 0], [4.5, 0.2], [5.5, 0.9], [5.9, 1.5], [6, 2]])  # Example curved segment
    ]

    bspline_fitter = BSplineFitter(lidar_segments)  
    bspline_curves, knot_points_list, control_points_list, centroids_list = bspline_fitter.fit_all_segments(knot_distance=0.5)
    bspline_fitter.visualize()

if __name__ == "__main__":
    main()
