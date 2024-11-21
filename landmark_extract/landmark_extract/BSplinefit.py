import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy import interpolate

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
        self.spline_order = []
        self.centroids = []

    def fit_bspline_to_lidar(self, lidar_segment, smoothness=0):
        """
        Fit a B-spline to a lidar segment and return the evaluated curve,
        control points, and centroid of control points.
        """
        # Extract x and y coordinates from the segment points
        x, y = lidar_segment[:, 0], lidar_segment[:, 1]
        
        # Fit the B-spline using splprep
        tck, u = splprep([x, y], s=smoothness)
        
        # Generate the B-spline curve by evaluating at a range of points
        spline_points = splev(np.linspace(0, 1, 100), tck)
        bspline_curve = np.vstack(spline_points).T
        
        # Evaluate knot positions
        knot_positions = splev(tck[0], tck)  # Evaluate spline at knot parameter values
        knot_points = np.vstack(knot_positions).T
        
        # Extract control points from the tck object
        control_points = np.array(tck[1]).T  # Control points are in tc k[1]
        
        # Compute the centroid of the control points
        centroid = np.mean(control_points, axis=0)
        
        return bspline_curve, knot_points, control_points, centroid

    def fit_all_segments(self, smoothness=0):
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
            bspline_curve, knot_points, control_points, centroid = self.fit_bspline_to_lidar(segment, smoothness)
            
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

            # Plot lidar segment points
            plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'o', markersize=5, label=f'Segment {i+1} Points')

            # Plot B-spline curve
            plt.plot(bspline_curve[:, 0], bspline_curve[:, 1], '-', linewidth=2, label=f'B-spline Segment {i+1}')
            
            #Plot knot points
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
        """Visualize lidar segments, fitted B-spline curves, control points, and centroids."""
        plt.clf()

        for i, lidar_segment in enumerate(self.lidar_segments):
            bspline_curve = self.bspline_curves[i]
            knot_points = self.knot_points[i]
            control_points = self.control_points[i]
            centroid = self.centroids[i]

            # Plot lidar segment points
            plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'o', markersize=5, label=f'Segment {i+1} Points')

            # Plot B-spline curve
            plt.plot(bspline_curve[:, 0], bspline_curve[:, 1], '-', linewidth=2, label=f'B-spline Segment {i+1}')
            
            #Plot knot points
            plt.plot(knot_points[:, 0], knot_points[:, 1], 'rx-', markersize=8, label=f'Knot Points {i+1}')
            
            # Plot contorl points
            plt.plot(control_points[:, 0], control_points[:, 1], 'kx-', markersize=8, label=f'Control Points {i+1}')

            # Plot centroid of control points
            plt.plot(centroid[0], centroid[1], 'r*', markersize=10, label=f'Centroid {i+1}')
            
            #Plot arrow at origin
            plt.arrow(0, 0, 0.3, 0, head_width=0.15, head_length=0.3, fc='k', ec='k')

        plt.title('B-Spline Curves Fitted to Lidar Segments with Control Point Centroids')
        plt.legend()
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot
    
    
def interpolate_track_new(points, n_points=None, s=0):
    if len(points) <= 1:
        return points
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
    return track

def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
    if points[0, 0] > points[-1, 0]:
        points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / seperation_distance), 2)
    smooth_line = interpolate_track_new(points, None, smoothing)
    resampled_points = interpolate_track_new(smooth_line, n_pts, 0)

    return resampled_points, smooth_line

# Example usage
def main():
    lidar_segments = [
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),  # Example straight segment
        np.array([[4, 0], [4.5, 0.5], [5, 1], [5.5, 1.5], [6, 2]])  # Example curved segment
    ]

    # bspline_fitter = BSplineFitter(lidar_segments)
    # bspline_curves, control_points, centroids = bspline_fitter.fit_all_segments()
    # bspline_fitter.visualize()
    
    bspline_fitter = BSplineFitter(lidar_segments)
    bspline_fitter.plot_live(interval=1.0)  # Update every 1 second

if __name__ == "__main__":
    main()
