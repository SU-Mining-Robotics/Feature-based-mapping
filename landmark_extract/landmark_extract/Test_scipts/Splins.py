import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Generate some sample data
x = np.linspace(0, 10, 8)
y = np.sin(x)
k = 3  # Degree of the spline

# Compute the knots and coefficients for the B-spline
# Standard uniform knots
t_uniform = np.linspace(x[0], x[-1], len(x) + k + 1)  # Knot vector
c = y  # Coefficients (control points)
print("Control points:\n", c)

# Clamped (open uniform) knot vector
n = len(x)  # Number of control points
t_clamped = np.concatenate((
    np.full(k, x[0]),  # Repeat the first knot (k times)
    np.linspace(x[0], x[-1], n - k + 1),  # Interior knots
    np.full(k, x[-1])  # Repeat the last knot (k times)
))

# Verify that the number of knots satisfies the condition
assert len(t_clamped) == len(c) + k + 1, "Knots, coefficients, and degree are inconsistent!"

print("Uniform Knots:\n", t_uniform)
print("Clamped Knots:\n", t_clamped)

# Create B-splines with and without extrapolation
bspline_with_extrap = BSpline(t_uniform, c, k, extrapolate=True)
bspline_no_extrap = BSpline(t_uniform, c, k, extrapolate=False)
bspline_clamped = BSpline(t_clamped, c, k, extrapolate=True)  # Clamped spline

# Generate a dense range of x values for plotting
x_dense = np.linspace(-5, 15, 500)

# Evaluate the B-splines
y_with_extrap = bspline_with_extrap(x_dense)
y_no_extrap = bspline_no_extrap(x_dense)
y_clamped = bspline_clamped(x_dense)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Data points', color='black')
plt.plot(x_dense, y_with_extrap, label='B-Spline (Extrapolate=True)', color='blue')
plt.plot(x_dense, y_no_extrap, label='B-Spline (Extrapolate=False)', color='red', linestyle='--')
plt.plot(x_dense, y_clamped, label='B-Spline (Clamped)', color='green', linestyle='-.')

plt.axvline(x[0], color='gray', linestyle=':', label='Extrapolation limits')
plt.axvline(x[-1], color='gray', linestyle=':')
plt.title('Effect of Extrapolate and Clamped Spline on B-Spline')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
