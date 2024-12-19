import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

class BSplineVisualizer:
    def __init__(self, control_points, degree, target_point, target_point2):
        self.control_points = control_points
        self.degree = degree
        self.n_control_points = len(control_points)
        self.clamped_knot_vector = self._generate_clamped_knot_vector()
        self.unclamped_knot_vector = self._generate_unclamped_knot_vector()
        self.target_point = target_point
        self.target_point2 = target_point2
        
        self.x_n_1 =[]
        self.x_n = []

    def _generate_clamped_knot_vector(self):
        """Generate a clamped knot vector."""
        clamped_knot_vector = (
            [0] * self.degree +  # Fully repeated at the start
            list(range(self.n_control_points - self.degree + 1)) +  # Internal knots
            [self.n_control_points - self.degree] * self.degree  # Fully repeated at the end
        )
        clamped_knot_vector = np.array(clamped_knot_vector)/clamped_knot_vector[-1]
        # print("Clamped Knot Vector:\n", clamped_knot_vector)
        return clamped_knot_vector

    def _generate_unclamped_knot_vector(self):
        """Generate a periodic (unclamped) knot vector."""
        unclamped_knot_vector = np.arange(-self.degree, self.n_control_points + 1)
        
        unclamped_knot_vector = ( [0] * self.degree + # Fully repeated at the start
        list(range(self.n_control_points  + 1)) # Fully repeated at the end
        )
        unclamped_knot_vector = np.array(unclamped_knot_vector)/unclamped_knot_vector[-1]
        # print("Right unclamped Vector:\n", unclamped_knot_vector)
        return unclamped_knot_vector

    def plot_bspline_with_knotsR(self, knot_vector, label, color):
        """Plot a B-spline along with its knots."""
        spline = BSpline(knot_vector, self.control_points, self.degree)
        t = np.linspace(knot_vector[self.degree], knot_vector[-self.degree - 1], 500)
        spline_points = spline(t)
        
        # Plot the spline curve
        plt.plot(spline_points[:, 0], spline_points[:, 1], color, label=label)
        # Plot the control polygon
        plt.plot(self.control_points[:, 0], self.control_points[:, 1], 'o--', color=color, alpha=0.5)
        # Plot the knots on the curve
        knots = knot_vector[self.degree:-self.degree]
        knot_points = spline(knots)
        plt.plot(knot_points[:, 0], knot_points[:, 1], 'o', color='black', label="Knots")
        plt.plot(self.target_point[0], self.target_point[1], 'o', color='red', label="Target Point")
        
        control_points_new , new_knot_vector = unclamp_right_side(knot_vector, self.control_points, self.target_point, self.degree)
        plt.plot(control_points_new[:, 0], control_points_new[:, 1], 'o--', color='purple', alpha=0.5)
        
        rspline = BSpline(new_knot_vector, control_points_new, self.degree)
        t = np.linspace(0, 1, 500)
        rspline_points = rspline(t)
        plt.plot(rspline_points[:, 0], rspline_points[:, 1], color = 'yellow', label="New")
        
    def plot_bspline_with_knotsL(self, knot_vector, label, color):
            """Plot a B-spline along with its knots."""
            spline = BSpline(knot_vector, self.control_points, self.degree)
            t = np.linspace(knot_vector[self.degree], knot_vector[-self.degree - 1], 500)
            spline_points = spline(t)
            
            # Plot the spline curve
            plt.plot(spline_points[:, 0], spline_points[:, 1], color, label=label)
            # Plot the control polygon
            plt.plot(self.control_points[:, 0], self.control_points[:, 1], 'o--', color=color, alpha=0.5)
            # Plot the knots on the curve
            knots = knot_vector[self.degree:-self.degree]
            knot_points = spline(knots)
            plt.plot(knot_points[:, 0], knot_points[:, 1], 'o', color='black', label="Knots")
            plt.plot(self.target_point2[0], self.target_point2[1], 'o', color='red', label="Target Point")
            
            control_points_newL, new_knot_vectorL = unclmap_left_side(knot_vector, self.control_points, self.target_point2, self.degree)
            plt.plot(control_points_newL[:, 0], control_points_newL[:, 1], 'o--', color='purple', alpha=0.5)
            
            lspline = BSpline(new_knot_vectorL, control_points_newL, self.degree)
            t = np.linspace(0, 1, 500)
            lspline_points = lspline(t)
            plt.plot(lspline_points[:, 0], lspline_points[:, 1], color = 'green', label="New")
            

    

    def plot_basis_functions(self, knot_vector, label):
        """Plot basis functions for the given knot vector."""
        t = np.linspace(knot_vector[0], knot_vector[-1], 500)
        n_basis = len(knot_vector) - self.degree - 1  # Number of basis functions
        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1
            basis = BSpline(knot_vector, coeffs, self.degree)
            # Plot only the valid range of the basis function
            valid_range = (t >= knot_vector[i]) & (t <= knot_vector[i + self.degree + 1])
            plt.plot(t[valid_range], basis(t[valid_range]), label=f"{label} Basis {i}")
        plt.title(f"{label} Basis Functions")
        plt.xlabel("t")
        plt.ylabel("N(t)")
        plt.legend()
        plt.grid(True)

    def visualize(self):
        """Visualize clamped and periodic B-splines and their basis functions."""
        plt.figure(figsize=(12, 12))

        # Clamped B-spline
        plt.subplot(3, 2, 1)
        self.plot_bspline_with_knotsR(self.clamped_knot_vector, "Clamped B-spline", "blue")
        plt.title("Right extention")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.xlim(0, 8)
        # plt.ylim(0, 4)
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        
        # Periodic (unclamped) B-spline
        plt.subplot(3, 2, 3)
        self.plot_bspline_with_knotsL(self.clamped_knot_vector, "Clamped B-spline", "red")
        plt.title("Left extention")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.xlim(0, 8)
        # plt.ylim(0, 4)
        plt.axis("equal")
        plt.legend()
        plt.grid(True)

        # # Periodic (unclamped) B-spline
        # plt.subplot(3, 2, 3)
        # self.plot_bspline_with_knotsR(self.unclamped_knot_vector, "Periodic B-spline", "red")
        # plt.title("Periodic Cubic B-spline")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # # plt.xlim(0, 8)
        # # plt.ylim(0, 4)
        # plt.axis("equal")
        # plt.legend()
        # plt.grid(True)

        # # Basis functions for Clamped B-spline
        # plt.subplot(3, 2, 2)
        # self.plot_basis_functions(self.clamped_knot_vector, "Clamped")

        # # Basis functions for Periodic B-spline
        # plt.subplot(3, 2, 4)
        # self.plot_basis_functions(self.unclamped_knot_vector, "Periodic")

        # Adjust layout
        plt.tight_layout()
        plt.show()
        
def find_u_distanceR(knot_vector, control_points, target_point, degree):
    k = degree + 1  # Spline order
    n = len(control_points) - 1  # Number of control points
        
    spline = BSpline(knot_vector, control_points, degree)
        
    R_distance = np.linalg.norm(control_points[-1] - target_point)
    Totalpoint_dist = 0
    
    for r in range(0, n - k  + 2): # Need to be 2 instead of 1 since range does not 
        P = spline(knot_vector[k + r])
        P_prev = spline(knot_vector[k + r - 1])
        Point_dist = np.linalg.norm(P - P_prev)
        # print(f"r_{r}:", Point_dist)
        Totalpoint_dist += Point_dist

    u = 1 + (R_distance / Totalpoint_dist)
    
    return u

def find_u_distanceL(knot_vector, control_points, target_point, degree):
    k = degree + 1  # Spline order
    n = len(control_points) - 1  # Number of control points
        
    spline = BSpline(knot_vector, control_points, degree)
        
    R_distance = np.linalg.norm(control_points[0] - target_point)
    Totalpoint_dist = 0
    
    for r in range(0, n - k  + 2): # Need to be 2 instead of 1 since range does not 
        P = spline(knot_vector[k + r])
        P_prev = spline(knot_vector[k + r - 1])
        Point_dist = np.linalg.norm(P - P_prev)
        # print(f"r_{r}:", Point_dist)
        Totalpoint_dist += Point_dist

    u = 0 - (R_distance / Totalpoint_dist)
    print("u:", u)  
    # u = -0.05
    
    return u

def unclmap_left_side(knot_vector, control_points, target_point, degree):
    k = degree + 1  # Spline order
    n = len(control_points) - 1  # Number of control points
    
    # u = find_u_distanceR(knot_vector, control_points, target_point, degree)
    u=find_u_distanceL(knot_vector, control_points, target_point, degree)
    
    #Modify knot vector
    T_2_temp = knot_vector[degree:]
    print("T_2_temp_funcL:", T_2_temp)
    T_2 = np.concatenate(([u]*degree, T_2_temp))
    print("T_2_funcL:", T_2)
    
    x_0 = 1/((omega(0, 1, k, T_2))*omega(0, 2, k, T_2)) * control_points[0] - ((omega_cap(0, 2, k, T_2)/omega(1, 2, k, T_2)) + (omega_cap(0, 1, k, T_2)/omega(0, 2, k, T_2))) * control_points[1] + omega_cap(0, 2, k, T_2)*omega_cap(1, 2, k, T_2) * control_points[2]
    print("x_0:", x_0)
    x_1 = (1/omega(1, 2, k, T_2)) * control_points[1] - omega_cap(1, 2, k, T_2) * control_points[2]
    # print("x_1:", x_1)
    x_i = control_points[2:]
    # print("x_i", x_i)
    
    # test = np.array([0.7,0.0])
    # x_0 = x_0-test
    # print("x_0:", x_0)
    
    control_points_new = np.concatenate(([target_point],[x_0], [x_1], x_i))
    print(control_points_new)
    
    test_knot_vector = _generate_clamped_knot_vector(degree, len(control_points_new))
    
    return control_points_new, test_knot_vector
    
def omega(i, j, k, knot_vector):
    omega = (knot_vector[k - 1] - knot_vector[i+k]) / (knot_vector[i + k - j - 1] - knot_vector[i + k])
    return omega

def omega_cap(i, j, k, knot_vector):
    omega_cap = (1 - omega(i, j, k, knot_vector))/omega(i, j, k, knot_vector)
    return omega_cap

def unclamp_right_side(knot_vector, control_points, target_point, degree):
    k = degree + 1  # Spline order
    n = len(control_points) - 1  # Number of control points
    
    u = find_u_distanceR(knot_vector, control_points, target_point, degree)
    # print("u_func:", u)
    
    #Modify knot vector
    print("Knot vector", knot_vector)
    T_2_temp = knot_vector[:-degree]
    print("T_2_temp_func:", T_2_temp)
    T_2 = np.concatenate((T_2_temp, [u]*degree))
    print("T_2_func:", T_2)
    
    #Apply unclampping algorithm (Right side)
    x_i = control_points[:n - 1] #Should be 2 but last element not included in syntax
    # print("x_i", x_i)
    x_n_1 = -gamma_cap(n-1, 2, n, T_2) * control_points[n-2] + 1/(gamma(n-1, 2, n, T_2)) * control_points[n-1]
    # print("x_n_1:", x_n_1)
    x_n = gamma_cap(n, 2, n, T_2) * gamma_cap(n-1, 2, n, T_2) * control_points[n-2] - ((gamma_cap(n, 2, n, T_2)/gamma(n-1, 2, n , T_2))+(gamma_cap(n, 1, n , T_2)/gamma(n, 2, n, T_2))) * control_points[n-1] + 1/(gamma(n, 1, n, T_2)*gamma(n, 2, n, T_2)) * control_points[n]
    # print("x_n:", x_n)
    
    control_points_new = np.concatenate((x_i, [x_n_1], [x_n], [target_point]))
    print(control_points_new)
    
    test_knot_vector = _generate_clamped_knot_vector(degree, len(control_points_new))
    
    return control_points_new, test_knot_vector
    
def gamma(i, j, n, knot_vector):
    y = (knot_vector[n+1] - knot_vector[i]) / (knot_vector[i + j + 1] - knot_vector[i])
    # print(f"E_{n+1}", knot_vector[n+1])
    # print(f"E_{i}", knot_vector[i])
    # print(f"E_{i + j + 1}", knot_vector[i + j + 1])
    # print(f"E_{i}", knot_vector[i])
    # print("y:", y)
    return y

def gamma_cap(i, j, n, knot_vector):
    gamma_cap = (1 - gamma(i , j, n , knot_vector))/gamma(i , j, n , knot_vector)
    # print("gamma_cap:", gamma_cap)
    return gamma_cap

def _generate_clamped_knot_vector(degree, control_points):
    """Generate a clamped knot vector."""
    clamped_knot_vector = (
        [0] * degree +  # Fully repeated at the start
        list(range(control_points - degree + 1)) +  # Internal knots
        [control_points - degree] * degree  # Fully repeated at the end
    )
    clamped_knot_vector = np.array(clamped_knot_vector)/clamped_knot_vector[-1]
    # print("Clamped Knot Vector:\n", clamped_knot_vector)
    return clamped_knot_vector


# Control points for the B-spline
control_points = np.array([
    [1.0, 0.0],
    [0.0, 3.0],
    [1.0, 6.0],
    [5.0, 6.0],
    [7.0, 4.0],
    [7.0, 3.0],
])

target_pointR = np.array([4.0, 3])
target_pointL = np.array([8.0, 0.0])
degree = 3

# Create a BSplineVisualizer instance and visualize
visualizer = BSplineVisualizer(control_points, degree, target_pointR, target_pointL)
visualizer.visualize()
