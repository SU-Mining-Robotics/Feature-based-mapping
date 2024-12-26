import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.integrate import quad


class SplineFitting:
    def __init__(self, phi_range=(np.pi , 0), num_points=500, noise_std=5 / 1000, degree=3, knot_spacing=2):
    # def __init__(self, phi_range=(np.pi/2 - 0.5, np.pi/2 + 0.5), num_points=20, noise_std=5 / 1000, degree=3, knot_spacing=0.5):
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
        self.control_points = None
        self.knots = None
        self.spline = None
        self.collocation_matrix = None

        self.B_pseudoinverse = None
        self.reversed_control_points = None
        self.r_spline = None

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
        print("Curve Length: ", self.curve_length)

    def calculate_control_points(self):
        """Calculate control points and knots proportional to the curve length and knot spacing."""
        num_segments = int(self.curve_length // self.knot_spacing)
        print(f'Num Segments: {num_segments}')
        segment_lengths = np.linspace(0, self.curve_length, num_segments + 3 )

        distances = np.cumsum(
            np.linalg.norm(np.diff(np.column_stack((self.x_noisy, self.y_noisy)), axis=0), axis=1)
        )
        distances = np.insert(distances, 0, 0)  # Include the starting point

        x_control = np.interp(segment_lengths, distances, self.x_noisy)
        y_control = np.interp(segment_lengths, distances, self.y_noisy)
        self.control_points = np.column_stack((x_control, y_control))
        print(f'Control Points\n{self.control_points}')

        # Knot vector
        num_knots = len(self.control_points) + self.degree + 1
        self.knots = np.zeros(num_knots)

        # Define the boundary knots
        self.knots[:self.degree + 1] = 0
        self.knots[-(self.degree + 1):] = self.curve_length

        # Define the interior knots
        num_interior_knots = num_knots - 2 * (self.degree + 1)
        if num_interior_knots > 0:
            interior_knots = np.linspace(0, self.curve_length, num_interior_knots + 2)[1:-1]
            self.knots[self.degree + 1:-(self.degree + 1)] = interior_knots

        print("Knots\n", self.knots)
        
        # self.knots[self.degree+1] = 0.001
        # self.knots[self.degree] = 0.12
        # print("Knots\n", self.knots)


    def fit_bspline(self):
        """Fit a B-spline to the control points."""
        if self.control_points is None or self.knots is None:
            raise ValueError("Control points and knots must be calculated first.")

        # Use BSpline to create the spline
        self.spline = BSpline(self.knots, self.control_points, self.degree)

    def calculate_collocation_matrix(self):
        """Calculate the collocation matrix B for the B-spline."""
        if self.knots is None or self.control_points is None:
            raise ValueError("Control points and knots must be calculated first.")

        # Parameter values corresponding to noisy data
        t = np.linspace(0, self.curve_length, self.num_points)

        # Evaluate basis functions for all t
        num_basis = len(self.knots) - self.degree - 1
        B = np.zeros((self.num_points, num_basis))
        for i in range(num_basis):
            coeff = np.zeros(num_basis)
            coeff[i] = 1
            basis_function = BSpline(self.knots, coeff, self.degree)
            B[:, i] = basis_function(t)
        self.collocation_matrix = B
        
        print("Collocation Matrix (B):\n")
        print(B)

        self.B_pseudoinverse = np.linalg.pinv(B)  # Use pseudoinverse directly
        self.reversed_control_points = self.B_pseudoinverse @ np.column_stack((self.x_noisy, self.y_noisy))
        # print(f'Reversed Control Points\n{self.reversed_control_points}')
        self.r_spline = BSpline(self.knots, self.reversed_control_points, self.degree)

    def calculate_mse(self):
        """Calculate the mean squared error between the spline and the noisy control points."""
        if self.spline is None:
            raise ValueError("You must fit the B-spline before calculating the MSE.")
        t = np.linspace(0, self.curve_length, self.num_points)
        fitted_points = self.spline(t)
        noisy_points = np.column_stack((self.x_noisy, self.y_noisy))
        mse = np.mean(np.sum((fitted_points - noisy_points) ** 2, axis=1))
        return mse
    
    def calculate_segment_lengths(self):
        """Calculate the length of the reversed B-spline between each pair of consecutive knots."""
        if self.r_spline is None:
            raise ValueError("You must fit the reversed B-spline before calculating segment lengths.")
        
        segment_lengths = []
        for i in range(len(self.knots) - 1):
            if self.knots[i] == self.knots[i + 1]:  # Skip duplicate knots
                continue
            
            # Define the integrand for arc length calculation
            def integrand(t):
                dxdt, dydt = self.r_spline(t, nu=1).T  # Derivative of the spline
                return np.sqrt(dxdt**2 + dydt**2)
            
            # Integrate between consecutive knots
            length, _ = quad(integrand, self.knots[i], self.knots[i + 1])
            segment_lengths.append(length)
        
        print("Segment Lengths between Knots:", segment_lengths)
        return segment_lengths

    def plot_bspline(self):
        """Plot the original data, noisy data, fitted B-spline, control points, and knots."""
        if self.spline is None:
            raise ValueError("You must fit the B-spline before plotting.")

        # Generate points on the B-spline
        t = np.linspace(0, self.curve_length, 1000)
        spline_points = self.spline(t)

        plt.figure(figsize=(8, 8))
        plt.scatter(self.x_noisy, self.y_noisy, s=10, color="orange", label="Noisy Data")
        plt.plot(spline_points[:, 0], spline_points[:, 1], label="Fitted B-spline", color="red")
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color="green", s=50, label="Control Points")
        
        # Plot the knots
        knot_positions = self.spline(self.knots[self.degree:-self.degree])  # Interior knots
        plt.scatter(knot_positions[:, 0], knot_positions[:, 1], color="blue", s=20, label="Knots", zorder=5)
        
        # Plot the reversed control points
        t = np.linspace(0, self.curve_length, 1000)
        plt.plot(self.r_spline(t)[:, 0], self.r_spline(t)[:, 1], label="Reversed B-spline", color="purple")
        plt.scatter(self.reversed_control_points[:, 0], self.reversed_control_points[:, 1], color="black", s=30, label="Reversed Control Points")
        
        # Plot the reversed knots
        reversed_knot_positions = self.r_spline(self.knots[self.degree:-self.degree])
        plt.scatter(reversed_knot_positions[:, 0], reversed_knot_positions[:, 1], color="yellow", s=20, label="Reversed Knots", zorder=5)

        plt.title("B-spline Fitting with Knots and Collocation Matrix")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()


if __name__ == "__main__":
    spline_fitter = SplineFitting()
    spline_fitter.generate_data()  # Generate the data
    spline_fitter.calculate_control_points()  # Calculate control points and knots
    spline_fitter.fit_bspline()  # Fit the B-spline
    spline_fitter.calculate_collocation_matrix()  # Compute the collocation matrix
    mse = spline_fitter.calculate_mse()  # Calculate the MSE
    print(f"Mean Squared Error: {mse}")
    segment_lengths = spline_fitter.calculate_segment_lengths()  # Calculate segment lengths
    spline_fitter.plot_bspline()  # Plot the results
