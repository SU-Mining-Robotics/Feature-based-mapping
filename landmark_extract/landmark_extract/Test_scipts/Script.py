import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


class SplineDataAssociation:
    def __init__(self, threshold=0.5, smoothing=0, k=3):
        """
        Initialize the SplineDataAssociation class.
        
        Args:
            threshold: Distance threshold for data association.
            smoothing: Smoothing factor for spline generation.
            k: Degree of the spline.
        """
        self.threshold = threshold
        self.smoothing = smoothing
        self.k = k
        self.map_splines = []
        self.obs_splines = []

    def generate_spline(self, points):
        """
        Generate a spline representation from a set of points.
        
        Args:
            points: A 2D array of shape (2, n_points) representing x and y coordinates.
        
        Returns:
            tck: The spline representation.
        """
        if points.shape[1] < self.k + 1:
            k = points.shape[1] - 1  # Reduce k to fit the number of points
        else:
            k = self.k
        tck, _ = splprep(points, s=self.smoothing, k=k)
        return tck

    def data_association(self):
        """
        Associates observed splines to map splines based on control point distances.
        
        Returns:
            associations: List of tuples representing matched spline pairs.
        """
        associations = []
        for obs_tck in self.obs_splines:        # For all observations
            obs_cp = np.vstack(obs_tck[1]).T  # Control points (n_points, 2)
            
            for map_tck in self.map_splines:    # For all map splines
                map_cp = np.vstack(map_tck[1]).T  # Control points (n_points, 2)
                # Calculate pairwise distances 
                # obs_cp[:, None, :] - Shape (N, 1, 2)
                # map_cp[None, :, :] - Shape (1, M, 2)
                # obs_cp[:, None, :] - map_cp[None, :, :] - Shape (N, M, 2)
                # distances - Shape (N, M) 
                # Each element (i, j) represents the distance between the ith observation and jth map control point
                distances = np.linalg.norm(obs_cp[:, None, :] - map_cp[None, :, :], axis=2)
                min_distance = np.min(distances)
                # Check if the distance is within the threshold
                if min_distance < self.threshold:
                    # Features are close (control point comparison)
                    # initial points matching
                    associations.append((obs_tck, map_tck))

        return associations

    def visualize(self, associations):
        """
        Visualizes the map splines, observed splines, control points, and their associations.
        
        Args:
            associations: List of associated splines.
        """
        plt.figure(figsize=(10, 8))

        # Plot map splines and control points
        for map_tck in self.map_splines:
            u = np.linspace(0, 1, 100)
            x, y = splev(u, map_tck)
            plt.plot(x, y, label="Map Spline", color="blue")
            map_cp = np.vstack(map_tck[1]).T
            plt.scatter(map_cp[:, 0], map_cp[:, 1], color="blue", marker='o', label="Map Control Points")

        # Plot observed splines and control points
        for obs_tck in self.obs_splines:
            u = np.linspace(0, 1, 100)
            x, y = splev(u, obs_tck)
            plt.plot(x, y, label="Observed Spline", color="green")
            obs_cp = np.vstack(obs_tck[1]).T
            plt.scatter(obs_cp[:, 0], obs_cp[:, 1], color="green", marker='x', label="Obs Control Points")

        # Highlight associations
        for obs_tck, map_tck in associations:
            obs_cp = np.vstack(obs_tck[1]).T
            map_cp = np.vstack(map_tck[1]).T
            for i in range(len(obs_cp)):
                plt.plot(
                    [obs_cp[i, 0], map_cp[i, 0]], [obs_cp[i, 1], map_cp[i, 1]],
                    color="red", linestyle="--", label="Association" if i == 0 else None
                )

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Spline Data Association with Control Points")
        plt.grid()
        plt.show()

    def process(self, map_points, obs_points):
        """
        Processes the map and observed points to generate splines and perform data association.
        
        Args:
            map_points: List of 2D arrays of map points.
            obs_points: List of 2D arrays of observed points.
        """
        self.map_splines = [self.generate_spline(points) for points in map_points]
        self.obs_splines = [self.generate_spline(points) for points in obs_points]

        # Perform data association
        associations = self.data_association()

        # Visualize the results
        self.visualize(associations)


if __name__ == "__main__":
    # Example data for map and observed points
    map_points = [
        # np.array([[0, 1, 2, 3], [0, 1, 0, -1]]),  # Spline 1
        # np.array([[2, 3, 4, 5], [1, 2, 1, 0]])   # Spline 2
        # np.array([[0, 1, 2, 3], [0, 1, 0, -1]])
         np.array([[0, 1, 4, 4.5, 6.5 , 7], [0, 0.5, 1.0, 0.8, 0.8, 0.9]])
    ]

    x_offset = -0.2
    y_offset = -0.2
    obs_points = [
        # np.array([[0.5, 1.5, 2.5], [0.5, 0.5, -0.5]]),  # Spline 1
        # np.array([[2.5, 3.5, 4.5], [1.5, 1.5, 0.5]]),    # Spline 2
        np.array([[4+x_offset, 4.5+x_offset, 6.5+x_offset , 7+x_offset], [1.0+y_offset, 0.8+y_offset, 0.8+y_offset, 0.9+y_offset]]),
        # np.array([[4, 4.5, 6.5 , 7], [1.0, 0.8, 0.8, 0.9]]),
        np.array([[6, 6.2, 7.8 , 8], [2, 2.1, 2.1, 2]])
    ]

    # Initialize the class and process data
    sda = SplineDataAssociation(threshold=0.5, smoothing=0, k=3)
    sda.process(map_points, obs_points)
