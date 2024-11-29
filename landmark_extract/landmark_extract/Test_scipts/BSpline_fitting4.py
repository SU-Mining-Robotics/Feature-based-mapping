import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import interpolate

class SplineFitting:
    def __init__(self, phi_range=(0, np.pi), num_points=500, noise_std=5 / 1000, degree=3, knot_spacing=2):
        self.phi_range = phi_range
        self.num_points = num_points
        self.noise_std = noise_std
        self.degree = degree
        self.knot_spacing = knot_spacing

        self.x = None
        self.y = None
        self.x_noisy = None
        self.y_noisy = None
        self.curve_length = None
        self.spline = None
        self.u = None

    def generate_data(self):
        """Generate the data points using the polar equation."""
        phi = np.linspace(self.phi_range[0], self.phi_range[1], self.num_points)
        r = 5 + 0.5 * np.sin(5 * phi)
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)
        self.x_noisy = self.x + np.random.normal(0, self.noise_std, size=self.x.shape)
        self.y_noisy = self.y + np.random.normal(0, self.noise_std, size=self.y.shape)

        points = np.column_stack([self.x_noisy, self.y_noisy])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.curve_length = np.sum(distances)

    def get_uniform_knots(self):
        """Generate a knot vector with uniform spacing."""
        num_knots = int(self.curve_length // self.knot_spacing) + 1
        knots = np.linspace(0, self.curve_length, num_knots)
        return knots

    def fit_bspline(self):
        """Fit a B-spline to the noisy data."""
        # Generate cumulative distances as parameter t for the spline
        points = np.column_stack([self.x_noisy, self.y_noisy])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        t = np.concatenate(([0], np.cumsum(distances)))

        # Normalize t to match the range of the knot vector
        # t /= t[-1]

        # Uniform knot vector with specified spacing
        knots = self.get_uniform_knots()
        print("Knots:", knots)
        # knots /= knots[-1]  # Normalize knots to [0, 1]

        # Fit the spline
        tck, u = interpolate.splprep([self.x_noisy, self.y_noisy], s = 1, u=t, k=self.degree, t=knots[1:-1])
        
        # tck1, u1 = interpolate.splprep([self.x_noisy, self.y_noisy], s = 0.2)
        # spline_test = interpolate.splev(u1, tck1)
        # plt.plot(spline_test[0], spline_test[1], label="Fitted B-spline", color="red")
        # plt.scatter(self.x_noisy, self.y_noisy, s=10, color="orange", label="Noisy Data")
        # plt.show()
        
        self.spline = tck
        self.u = knots
        print("Knots _vector:", self.u)  

    def plot_bspline(self):
        """Plot the original data, noisy data, fitted B-spline, knots, and control points."""
        if self.spline is None:
            raise ValueError("You must fit the B-spline before plotting.")

        # Generate points on the spline
        u = np.linspace(0, self.u[-1], 1000)
        spline_points = interpolate.splev(self.u, self.spline)
        spline_points1 = interpolate.splev(u, self.spline)

        # Extract control points and knots
        control_points = np.array(self.spline[1])  # Control points
        knots = self.spline[0]  # Knot vector
        print("Knots _vector_wrong:", knots)  

        # Generate knot points on the spline
        knot_points = interpolate.splev(self.u, self.spline)

        plt.figure(figsize=(8, 8))
        # plt.plot(self.x, self.y, label="Original Curve", color="blue")
        plt.scatter(self.x_noisy, self.y_noisy, s=10, color="orange", label="Noisy Data")
        plt.plot(spline_points[0], spline_points[1], label="Fitted B-spline(Knot points)", color="red")
        plt.plot(spline_points1[0], spline_points1[1], label="Fitted B-spline(More points)", color="green")
        plt.scatter(control_points[0], control_points[1], color="green", s=50, label="Control Points")
        plt.scatter(knot_points[0], knot_points[1], color="purple", s=50, label="Knots")
        plt.title("B-spline Fitting with Knots and Control Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()


if __name__ == "__main__":
    spline_fitter = SplineFitting(knot_spacing=1)
    spline_fitter.generate_data()  # Generate the data
    spline_fitter.fit_bspline()   # Fit the B-spline
    spline_fitter.plot_bspline()  # Plot the results
