import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import interpolate

u_values = np.linspace(0, 1, 10)  # Generate 10 values of u between 0 and 1
knots = np.array([[u**3, u**2, u, 1] for u in u_values])
# print("knots:\n", knots)

M = np.array([[-1, 3, -3, 1],
              [3, -6, 3, 0],
              [-3, 3, 0, 0],
              [1, 0, 0, 0]])
B = knots @ M
print("B:\n", B)

Control_points = np.array([[0, 0], [1, 1], [2, 1], [3,0]])

points = B  @ Control_points# Update to use the new knots array
# print("Points:", points)

B_transpose = B.T
B_pseudoinverse = np.linalg.pinv(B)  # Use pseudoinverse directly
print("B_pseudoinverse:\n", B_pseudoinverse)

data = B_pseudoinverse @ points
print("Data points:\n", data)


if __name__ == "__main__":
    plt.figure(figsize=(12, 12))
    plt.scatter(Control_points[:, 0], Control_points[:, 1], label='Control Points', color='green')
    plt.plot(points[:, 0], points[:, 1], label='Points', color='red')
    plt.title('Basics')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
