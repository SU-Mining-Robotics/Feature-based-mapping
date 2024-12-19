import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev

class BSplineFittingTest:
    def __init__(self, spline_order=3, knot_spacing=2.92, noise_std=0.005):
        """
        Initialize the test parameters.
        - spline_order: Order of the spline (e.g., 2 for quadratic, 3 for cubic).
        - knot_spacing: Distance between consecutive knots.
        - noise_std: Standard deviation of synthetic noise added to the measurements.
        """
        self.spline_order = spline_order
        self.knot_spacing = knot_spacing
        self.noise_std = noise_std

    def generate_wall_data(self, angle_range, resolution):
        """
        Simulate noisy laser range data for a curved wall.
        - angle_range: Tuple of (min_angle, max_angle) in radians.
        - resolution: Angular resolution in radians.
        """
        angles = np.arange(angle_range[0], angle_range[1], resolution)
        true_distances = 5 + 0.5 * np.sin(5 * angles)  # Polar equation of the wall
        noisy_distances = true_distances + np.random.normal(0, self.noise_std, size=true_distances.shape)
        
        x = noisy_distances * np.cos(angles)
        y = noisy_distances * np.sin(angles)
        return np.vstack((x, y)).T, np.vstack((angles, true_distances)).T

    def chord_length_parametrization(self, points):
        """
        Compute chord length parameterization for the given points.
        """
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        return cumulative_distances / cumulative_distances[-1]  # Normalize to [0, 1]

    def fit_bspline(self, data_points):
        """
        Fit a B-spline to the given data points using predefined knot spacing and spline order.
        """
        t = self.chord_length_parametrization(data_points)

        # Ensure `t` is strictly increasing
        if not np.all(np.diff(t) > 0):
            raise ValueError("Parameterization t is not strictly increasing.")

        # Fit splines for x and y separately
        tck_x = splrep(t, data_points[:, 0], k=self.spline_order, s=0)  # Let splrep handle knots automatically
        tck_y = splrep(t, data_points[:, 1], k=self.spline_order, s=0)

        return tck_x, tck_y

    def evaluate_bspline(self, tck_x, tck_y, num_points=100):
        """
        Evaluate the fitted B-spline at a set of points.
        """
        u = np.linspace(0, 1, num_points)
        x_vals = splev(u, tck_x)
        y_vals = splev(u, tck_y)
        return np.vstack((x_vals, y_vals)).T

    def run_monte_carlo(self, num_runs=50, angle_range=(-np.pi / 2, np.pi / 2), resolution=np.pi / 180):
        """
        Run Monte Carlo experiments to test the fitting process.
        """
        mse_list = []
        for _ in range(num_runs):
            data_points, true_points = self.generate_wall_data(angle_range, resolution)
            tck_x, tck_y = self.fit_bspline(data_points)
            fitted_points = self.evaluate_bspline(tck_x, tck_y, num_points=len(data_points))
            
            residuals = data_points - fitted_points
            mse = np.mean(np.sum(residuals**2, axis=1))
            mse_list.append(mse)
        
        return np.mean(mse_list), np.std(mse_list)

    def visualize(self, angle_range=(-np.pi / 2, np.pi / 2), resolution=np.pi / 180):
        """
        Visualize the noisy data, true curve, and fitted spline.
        """
        data_points, true_points = self.generate_wall_data(angle_range, resolution)
        tck_x, tck_y = self.fit_bspline(data_points)
        fitted_points = self.evaluate_bspline(tck_x, tck_y)
        
        plt.figure(figsize=(10, 6))
        plt.plot(data_points[:, 0], data_points[:, 1], 'o', label='Noisy Data', alpha=0.6)
        plt.plot(fitted_points[:, 0], fitted_points[:, 1], '-', label='Fitted Spline', linewidth=2)
        plt.title(f'B-Spline Fitting (Order: {self.spline_order}, Knot Spacing: {self.knot_spacing})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


# Test the script
if __name__ == "__main__":
    tester = BSplineFittingTest(spline_order=3, knot_spacing=6, noise_std=0.05)
    
    # Run Monte Carlo and print results
    mean_mse, std_mse = tester.run_monte_carlo()
    print(f"Mean MSE: {mean_mse:.6f}, Std MSE: {std_mse:.6f}")
    
    # Visualize
    tester.visualize()
