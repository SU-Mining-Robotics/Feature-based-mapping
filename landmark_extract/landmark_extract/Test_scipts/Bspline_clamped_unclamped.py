import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Function to plot a B-spline and its knots
def plot_bspline_with_knots(control_points, knot_vector, degree, label, color):
    # Create the B-spline
    spline = BSpline(knot_vector, control_points, degree)
    # Evaluate the spline
    t = np.linspace(knot_vector[degree], knot_vector[-degree - 1], 500)  # Valid range
    spline_points = spline(t)
    # Plot the spline curve
    plt.plot(spline_points[:, 0], spline_points[:, 1], color, label=label)
    # Plot the control polygon
    plt.plot(control_points[:, 0], control_points[:, 1], 'o--', color=color, alpha=0.5)
    # Plot the knots on the curve
    knots = knot_vector[degree:-degree]  # Knots influencing the curve
    knot_points = spline(knots)
    plt.plot(knot_points[:, 0], knot_points[:, 1], 'o', color='black', label="Knots")

# Function to plot basis functions
def plot_basis_functions(knot_vector, degree, label):
    t = np.linspace(knot_vector[0], knot_vector[-1], 500)
    n_basis = len(knot_vector) - degree - 1  # Number of basis functions
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1
        basis = BSpline(knot_vector, coeffs, degree)
        # Plot only the valid range of the basis function
        valid_range = (t >= knot_vector[i]) & (t <= knot_vector[i + degree + 1])
        plt.plot(t[valid_range], basis(t[valid_range]), label=f"{label} Basis {i}")
    plt.title(f"{label} Basis Functions")
    plt.xlabel("t")
    plt.ylabel("N(t)")
    plt.grid(True)

# Control points for the B-spline
control_points = np.array([
    [1.0, 2.0],
    [3.0, 3.0],
    [5.0, 3.0],
    [7.0, 2.0],
    [6.0, 1.0],
    [4.0, 1.0]#,
    #[2.0, 1.0],
    #[1.0, 2.0]
])

# Degree of the B-spline (cubic)
degree = 3  # Order of the polynomial

# Clamped knot vector
n_control_points = len(control_points)
n_knots = n_control_points + degree + 1
clamped_knot_vector = (
    [0] * degree +  # Fully repeated at the start
    list(range(n_control_points - degree + 1)) +  # Internal knots
    [n_control_points - degree] * degree  # Fully repeated at the end
)
print("Clamped Knot Vector:", clamped_knot_vector)

# Periodic (unclamped) knot vector
unclamped_knot_vector = np.arange(-degree, n_control_points + degree - degree + 1)
print("Unclamped Knot Vector (Periodic):", unclamped_knot_vector)

# Plot both B-splines
plt.figure(figsize=(12, 12))

# Clamped B-spline
plt.subplot(3, 2, 1)
plot_bspline_with_knots(control_points, clamped_knot_vector, degree, "Clamped B-spline", "blue")
plt.title("Clamped Cubic B-spline")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0, 8)
plt.ylim(0, 4)
plt.legend()
plt.grid(True)

# Periodic Unclamped B-spline
plt.subplot(3, 2, 3)
plot_bspline_with_knots(control_points, unclamped_knot_vector, degree, "Periodic B-spline", "red")
plt.title("Periodic Cubic B-spline")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0, 8)
plt.ylim(0, 4)
plt.legend()
plt.grid(True)

# Basis functions for Clamped B-spline
plt.subplot(3, 2, 2)
plot_basis_functions(clamped_knot_vector, degree, "Clamped")

# Basis functions for Periodic B-spline
plt.subplot(3, 2, 4)
plot_basis_functions(unclamped_knot_vector, degree, "Periodic")

# Adjust layout
plt.tight_layout()
plt.show()
