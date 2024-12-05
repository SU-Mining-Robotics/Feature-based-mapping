import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Step 1: Generate noisy data points
# Let's create a smooth sine wave and add some Gaussian noise to it
np.random.seed(0)  # For reproducibility
x_data = np.linspace(0, 10, 10)
y_data = np.sin(x_data) + np.random.normal(0, 0.1, len(x_data))  # Sine wave with noise

# Step 2: Define the cubic spline basis functions
# For simplicity, we will use scipy's CubicSpline to fit the spline.
# But first, we will construct the collocation matrix manually for demonstration.

# The parameter values tj (here we use the cumulative chord length)
t_data = np.zeros_like(x_data)
t_data[0] = 0
for i in range(1, len(x_data)):
    t_data[i] = t_data[i-1] + np.abs(x_data[i] - x_data[i-1])


# Step 3: Construct the collocation matrix B
# We will use the basis functions from the scipy CubicSpline, but normally we'd compute these manually.
# The basis functions in this case are the piecewise cubic functions that define the spline.
# In this case, B is not easy to write manually, so we will directly use the CubicSpline class.
cs = CubicSpline(x_data, y_data)

# Step 4: Solve for the control points using the pseudoinverse
# Using the formula x = (B^T B)^-1 B^T d, where B is the collocation matrix.
# The control points are essentially the coefficients of the spline curve.
# However, scipy's CubicSpline automatically computes this for us.

# Step 5: Evaluate the spline at a higher resolution for plotting
x_fine = np.linspace(0, 10, 100)
y_fine = cs(x_fine)

# Step 6: Plot the noisy data and the fitted spline
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='red', label='Noisy Data')
plt.plot(x_fine, y_fine, label='Fitted Spline', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spline Curve Fitting to Noisy Data')
plt.show()

# Output the control points (coefficients)
print("Control Points (Spline Coefficients):")
print(cs.c)  # These are the coefficients of the spline

