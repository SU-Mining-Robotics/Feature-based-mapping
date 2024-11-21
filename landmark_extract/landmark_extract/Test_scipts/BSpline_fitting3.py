import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate noisy data points
np.random.seed(0)  # For reproducibility
x_data = np.linspace(0, 10, 10)  # 10 data points
y_data = np.sin(x_data) + np.random.normal(0, 0.1, len(x_data))  # Noisy sine wave

# Step 2: Compute the parameter values t_j (using cumulative chord length)
t_data = np.zeros_like(x_data)
t_data[0] = 0
for i in range(1, len(x_data)):
    t_data[i] = t_data[i-1] + np.abs(x_data[i] - x_data[i-1])

# Step 3: Construct the collocation matrix B
def cubic_basis(t, t_i, t_i1, t_i2, t_i3, epsilon=1e-6):
    """Compute the cubic B-spline basis functions."""
    # Ensure that the denominator doesn't become too small to avoid division by zero
    if abs(t_i1 - t_i) < epsilon:
        return 0
    if t_i <= t < t_i1:
        return (t - t_i)**3 / ((t_i1 - t_i) * (t_i1 - t_i) * (t_i1 - t_i2))
    elif t_i1 <= t < t_i2:
        return (t - t_i1)**3 / ((t_i1 - t_i) * (t_i2 - t_i) * (t_i3 - t_i1))
    elif t_i2 <= t < t_i3:
        return (t - t_i2)**3 / ((t_i2 - t_i1) * (t_i3 - t_i2) * (t_i3 - t_i))
    else:
        return 0

# Number of data points (and control points)
n_data = len(x_data)
n_control = n_data - 1  # Number of control points (this could change based on spline order)

# Build the collocation matrix B
B = np.zeros((n_data, n_control))  # Collocation matrix B

# Example for cubic spline with a basic approach
for j in range(n_data):
    for i in range(n_control):
        # Ensure that we have four indices for the basis function (t_i, t_i1, t_i2, t_i3)
        if i + 3 < len(t_data):  # Make sure we don't go out of bounds
            B[j, i] = cubic_basis(t_data[j], t_data[i], t_data[i+1], t_data[i+2], t_data[i+3])

# Step 4: Solve for the control points using pseudoinverse of B
# This is x = (B^T B)^-1 B^T d
B_transpose = B.T
B_pseudoinverse = np.linalg.pinv(B)  # Use pseudoinverse directly
control_points = B_pseudoinverse @ y_data

# Step 5: Define the spline evaluation
def evaluate_spline(t, control_points):
    """Evaluate the spline at a given point."""
    result = 0
    for i, control_point in enumerate(control_points):
        if i + 3 < len(t_data):  # Ensure we don't exceed the array bounds
            result += control_point * cubic_basis(t, t_data[i], t_data[i+1], t_data[i+2], t_data[i+3])
    return result

# Step 6: Evaluate the spline at a higher resolution for plotting
x_fine = np.linspace(0, 10, 100)
y_fine = [evaluate_spline(t, control_points) for t in x_fine]

# Step 7: Plot the noisy data and the fitted spline
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='red', label='Noisy Data')
plt.plot(x_fine, y_fine, label='Fitted Spline', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Manually Fitted Cubic Spline to Noisy Data')
plt.show()

# Output the control points (coefficients)
print("Control Points (Spline Coefficients):")
print(control_points)
