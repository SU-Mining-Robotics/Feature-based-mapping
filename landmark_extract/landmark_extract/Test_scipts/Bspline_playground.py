import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Function to plot the B-spline curve
def plot_bspline(control_points, knot_vector, degree, label, color):
    # Create the B-spline
    spline = BSpline(knot_vector, control_points, degree)
    # Evaluate the spline
    t = np.linspace(knot_vector[0], knot_vector[-1], 500)
    spline_points = spline(t)
    # Plot the spline curve
    plt.plot(spline_points[:, 0], spline_points[:, 1], color, label=label)
    # Plot the control polygon
    plt.plot(control_points[:, 0], control_points[:, 1], 'o--', color=color, alpha=0.5)
    # Plot the knots on the curve
    knots = knot_vector[degree:-degree]  # Internal knots influencing the curve
    knot_points = spline(knots)
    plt.plot(knot_points[:, 0], knot_points[:, 1], 'o', color='black', label="Knots")

# Function to plot basis functions
def plot_basis_functions(knot_vector, degree, label, color):
    t = np.linspace(knot_vector[0], knot_vector[-1], 500)
    n_basis = len(knot_vector) - degree - 1  # Number of basis functions
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1
        basis = BSpline(knot_vector, coeffs, degree)
        # Plot the basis function
        plt.plot(t, basis(t), color=color, label=f"{label} Basis {i}" if i == 0 else None)

# Initial setup for a cubic B-spline
degree = 3
control_points = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, -1.0],
    [4.0, 3.0],
    [5.0, 0.0]
])

# knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 4]  # Corrected clamped knot vector
knot_vector = [0, 0, 0, 0, 1, 2, 2, 2 ,2]  # Corrected clamped knot vector

# Plot the original B-spline and its basis functions
plt.figure(figsize=(12, 8))

# Plot the spline in the first subplot
plt.subplot(2, 1, 1)
plot_bspline(control_points, knot_vector, degree, "Original B-spline", "blue")
plt.title("Original B-spline with Knots")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.axis("equal")

# Plot the basis functions in the second subplot
plt.subplot(2, 1, 2)
plot_basis_functions(knot_vector, degree, "Original", "green")
plt.title("Basis Functions")
plt.xlabel("t")
plt.ylabel("N(t)")
plt.grid(True)
plt.legend()

# # Add a new control point (commented as requested)
# new_control_point = [6.0, 1.0]
# extended_control_points = np.vstack([control_points, new_control_point])

# # Create an extended knot vector
# extended_knot_vector = knot_vector[:-3] + [5, 5, 5]

# # Plot the extended B-spline
# plot_bspline(extended_control_points, extended_knot_vector, degree, "Extended B-spline", "red")

plt.tight_layout()
plt.show()
