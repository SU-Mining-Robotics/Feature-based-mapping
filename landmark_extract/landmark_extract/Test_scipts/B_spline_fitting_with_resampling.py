import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import interpolate

class SplineFitting:
    def __init__(self, phi_range=(0, np.pi), num_points=500, noise_std=5/1000, degree=3):
        """
        Initialize the SplineFitting class.
        
        Args:
        - phi_range: The range of the angle phi for generating the curve (default is from 0 to pi).
        - num_points: Number of points to generate for the curve.
        - noise_std: Standard deviation of the noise (in meters).
        - degree: Degree of the B-spline (default is cubic, degree=3).
        """
        self.phi_range = phi_range
        self.num_points = num_points
        self.noise_std = noise_std
        self.degree = degree
        
        self.x = None
        self.y = None
        self.x_noisy = None
        self.y_noisy = None
    
    def generate_data(self):
        """Generate the data points using the polar equation."""
        # Define the range of phi from 0 to pi
        phi = np.linspace(self.phi_range[0], self.phi_range[1], self.num_points)

        # Polar equation z = 5 + 0.5 * sin(5 * phi)
        r = 5 + 0.5 * np.sin(5 * phi)

        # Convert polar coordinates to Cartesian coordinates
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)

        # Add synthetic noise to the points
        self.x_noisy = self.x + np.random.normal(0, self.noise_std, size=self.x.shape)
        self.y_noisy = self.y + np.random.normal(0, self.noise_std, size=self.y.shape)
        
        # # Plot the result
        # plt.figure(figsize=(6, 6))
        # plt.plot(self.x, self.y, label=r'$z = 5 + 0.5 \sin(5\phi)$')
        # # plt.scatter(self.x, self.y, s=10, color='red')  # Add points
        # plt.scatter( self.x_noisy,  self.y_noisy, s=10, color='orange')  # Add synthetic noisy points
        # plt.title('Plot of z = 5 + 0.5sin(5ϕ)')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid(True)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.legend()
        # plt.show()
    
    def calculate_curve_length(self):
        """Calculate the total length of the curve using the Euclidean norm between consecutive points."""
        if self.x_noisy is None or self.y_noisy is None:
            raise ValueError("No data points available. Generate data first.")
        
        # points = np.column_stack([self.x_noisy, self.y_noisy])
        points = np.column_stack([self.x, self.y])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = np.sum(distances)
        return total_length
      
def interpolate_track_new(points, n_points=None, s=0):
    # if len(points) <= 1:
    #     return points
    order_k = min(3, len(points) - 1)
    # order_k = 3
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
    return track, tck
    
def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
    # if points[0, 0] > points[-1, 0]:
    #     points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / seperation_distance), 2)
    smooth_line, _ = interpolate_track_new(points, None, smoothing)
    resampled_points, re_tck = interpolate_track_new(smooth_line, n_pts, 0)
    resampled_points_lenght = np.sum(np.linalg.norm(np.diff(resampled_points, axis=0), axis=1))

    return resampled_points, smooth_line, line_length, resampled_points_lenght, re_tck

# Example usage:
spline_fitter = SplineFitting()
spline_fitter.generate_data()  # Generate the data
points = np.column_stack([spline_fitter.x_noisy, spline_fitter.y_noisy])
resampled_points, smooth_line, line_lenght, resampled_lenght , re_tck = resample_track_points(points, seperation_distance=2., smoothing=0.2)


# Calculate and print the total length of the curve
total_length = spline_fitter.calculate_curve_length()
print("Total length of the curve:", total_length)
print("Line Length:", line_lenght)
print("Resampled Line Length:", resampled_lenght)

# Plot the result
plt.figure(figsize=(6, 6))
# plt.plot(spline_fitter.x, spline_fitter.y, label=r'$z = 5 + 0.5 \sin(5\phi)$')
plt.scatter(spline_fitter.x_noisy, spline_fitter.y_noisy, s=10, color='orange')  # Add synthetic noisy points
plt.scatter(resampled_points[:, 0], resampled_points[:, 1], label='Resampled Points', s=10 ,color='red')
# plt.plot(re_tck[1][0], re_tck[1][1], 'x', label='Knot Points', color='blue')
plt.plot(smooth_line[:, 0], smooth_line[:, 1], label='Smooth Line', color='green')
plt.title('Plot of z = 5 + 0.5sin(5ϕ)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()

print(re_tck[1])


