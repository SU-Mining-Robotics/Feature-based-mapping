import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.interpolate import BSpline
import logging

logging.basicConfig(level=logging.WARNING)

class BSplineFitter:
    def __init__(self):
        """
        Initialize the BSplineFitter with a list of lidar segments.
        Each segment should be a list of points (numpy arrays of shape (N, 2)).
        """
        self.lidar_segments = []
        self.bspline_curves = []
        self.knot_points = []
        self.control_points = []
        self.centroids = []
        self.degree = 3
        
        self.curve_length_list = []
        self.knots_list = []
        self.control_points_list = []
        self.spline_list = []
        self.Collocation_Matrix_list = []
        self.B_pseudoinverse_list = []
        self.reversed_control_points_list = []
        self.r_spline_list = []
        
    
    def feed_lidar_segments(self, lidar_segments):
        self.lidar_segments = lidar_segments


    def fit_bspline_to_lidar(self, lidar_segment, knot_spacing=1.0, degree=3):
        """
        Fit a B-spline to a lidar segment with a specified knot distance.
        """
        x, y = lidar_segment[:, 0], lidar_segment[:, 1]
        points = np.column_stack([x, y])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        curve_length = np.sum(distances)
        # print("Curve Length: ", curve_length)
        
        if curve_length < knot_spacing:
            logging.warning("Curve length is less than knot spacing. Skipping fitting.")
            placeholder = np.array([])  # Placeholder for skipped segments
            return curve_length, placeholder, placeholder, placeholder, placeholder, placeholder, placeholder, placeholder
        
        knots, control_points = self.calculate_knots_control_points(curve_length, x, y, degree=3, knot_spacing=knot_spacing)
        # print(f'Knots length: {len(knots)}')
        # print(f'Control Points length: {len(control_points)}')
        spline = BSpline(knots, control_points, degree)
        Collocation_Matrix, B_pseudoinverse, reversed_control_points, r_spline = self.calculate_collocation_matrix(curve_length, x, y, knots, degree=3, knot_spacing=knot_spacing)
        
        self.spline = spline
        self.knots = knots
        self.control_points = control_points
        self.curve_length = curve_length
        self.reversed_control_points = reversed_control_points
        self.r_spline = r_spline
        self.x_noisy = x
        self.y_noisy = y
        self.degree = degree
        
        return curve_length, knots, control_points, spline, Collocation_Matrix, B_pseudoinverse, reversed_control_points, r_spline

    def calculate_knots_control_points(self, curve_length, x_noisy, y_noisy, degree = 3, knot_spacing = 1):
        """Calculate control points and knots proportional to the curve length and knot spacing."""
        num_segments = int(curve_length // knot_spacing)
        # print(f'Num Segments: {num_segments}')
        segment_lengths = np.linspace(0, curve_length, num_segments + 3 )

        distances = np.cumsum(
            np.linalg.norm(np.diff(np.column_stack((x_noisy, y_noisy)), axis=0), axis=1)
        )
        distances = np.insert(distances, 0, 0)  # Include the starting point

        x_control = np.interp(segment_lengths, distances, x_noisy)
        y_control = np.interp(segment_lengths, distances, y_noisy)
        control_points = np.column_stack((x_control, y_control))
        # print(f'Control Points\n{control_points}')

        # Knot vector
        num_knots = len(control_points) + degree + 1
        knots = np.zeros(num_knots)

        # Define the boundary knots
        knots[:degree + 1] = 0
        knots[-(degree + 1):] = curve_length

        # Define the interior knots
        num_interior_knots = num_knots - 2 * (degree + 1)
        if num_interior_knots > 0:
            interior_knots = np.linspace(0, curve_length, num_interior_knots + 2)[1:-1]
            knots[degree + 1:-(degree + 1)] = interior_knots

        # print("Knots\n", knots)
        
        return knots, control_points
    
    def calculate_collocation_matrix(self, curve_length, x_noisy, y_noisy, knots, degree = 3, knot_spacing = 1):
            """Calculate the collocation matrix B for the B-spline."""
           
            num_points = len(x_noisy)
            # Parameter values corresponding to noisy data
            t = np.linspace(0, curve_length, num_points)

            # Evaluate basis functions for all t
            num_basis = len(knots) - degree - 1
            B = np.zeros((num_points, num_basis))
            for i in range(num_basis):
                coeff = np.zeros(num_basis)
                coeff[i] = 1
                basis_function = BSpline(knots, coeff, degree)
                B[:, i] = basis_function(t)
            
            # print("Collocation Matrix (B):\n")
            # print(B)

            B_pseudoinverse = np.linalg.pinv(B)  # Use pseudoinverse directly
            reversed_control_points = B_pseudoinverse @ np.column_stack((x_noisy, y_noisy))
            # print(f'Reversed Control Points\n{self.reversed_control_points}')
            r_spline = BSpline(knots, reversed_control_points, degree)
            
            return B, B_pseudoinverse, reversed_control_points, r_spline




    def fit_all_segments(self, knot_spacing=1.0):
        """
        Fit B-splines to all lidar segments and store:
        - A list of B-spline curves
        - A list of control points for each segment
        - A list of centroids of control points for each segment
        """
        # Clear previous results
        self.curve_length_list = []
        self.knots_list = []
        self.control_points_list = []
        self.spline_list = []
        self.Collocation_Matrix_list = []
        self.B_pseudoinverse_list = []
        self.reversed_control_points_list = []
        self.r_spline_list = []        
   
        for segment in self.lidar_segments:
            curve_length, knots, control_points, spline, Collocation_Matrix, B_pseudoinverse, reversed_control_points, r_spline = self.fit_bspline_to_lidar(
                segment, knot_spacing
            )
            # print(f'Curve Length: {curve_length}')
            
            # Store results
            self.curve_length_list.append(curve_length)
            self.knots_list.append(knots)
            self.control_points_list.append(control_points)
            self.spline_list.append(spline)
            self.Collocation_Matrix_list.append(Collocation_Matrix)
            self.B_pseudoinverse_list.append(B_pseudoinverse)
            self.reversed_control_points_list.append(reversed_control_points)
            self.r_spline_list.append(r_spline)
        
        # print(f'Curve Length List: {len(self.curve_length_list)}')
                
    def send_results(self):
        if  self.Collocation_Matrix_list is None:
            raise ValueError("You must fit the B-splines before getting results.")

        return self.curve_length_list, self.knots_list, self.control_points_list, self.spline_list, self.Collocation_Matrix_list, self.B_pseudoinverse_list, self.reversed_control_points_list, self.r_spline_list
                
    def plot_bspline(self):
        """Plot the original data, noisy data, fitted B-spline, control points, and knots."""
        if self.spline is None:
            raise ValueError("You must fit the B-spline before plotting.")

        # Generate points on the B-spline
        t = np.linspace(0, self.curve_length, 100)
        # spline_points = self.spline(t)

        # plt.figure(figsize=(8, 8))
        # plt.scatter(self.x_noisy, self.y_noisy, s=10, color="orange", label="Noisy Data")
        # plt.plot(spline_points[:, 0], spline_points[:, 1], label="Fitted B-spline", color="red")
        # plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color="green", s=50, label="Control Points")
        
        # # Plot the knots
        # knot_positions = self.spline(self.knots[self.degree:-self.degree])  # Interior knots
        # plt.scatter(knot_positions[:, 0], knot_positions[:, 1], color="blue", s=20, label="Knots", zorder=5)
        
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
        
    def visualize(self):
        """Visualize lidar segments, fitted B-spline curves, control points, and centroids."""
        
        if self.Collocation_Matrix_list is None:
            raise ValueError("You must fit the B-splines before getting results.")
    
        plt.figure(figsize=(10, 6))

        for i, lidar_segment in enumerate(self.lidar_segments):
            curve_length = self.curve_length_list[i]
            spline = self.spline_list[i]
            control_points = self.control_points_list[i]
            knots = self.knots_list[i]
            r_spline = self.r_spline_list[i]
            reversed_control_points = self.reversed_control_points_list[i]
            
            placeholder = np.array([])
            # Skip plotting if a placeholder is encountered
            if not isinstance(spline, BSpline):
                continue
            
            # Generate points on the B-spline
            t = np.linspace(0, curve_length, 1000)
            spline_points = spline(t)

            plt.scatter(lidar_segment[:,0], lidar_segment[:,1], s=10, color="orange", label="Noisy Data")
            plt.plot(spline_points[:, 0], spline_points[:, 1], label="Fitted B-spline", color="red")
            plt.scatter(control_points[:, 0], control_points[:, 1], color="green", s=50, label="Control Points")
            
            # Plot the knots
            knot_positions = spline(knots[self.degree:-self.degree])  # Interior knots
            plt.scatter(knot_positions[:, 0], knot_positions[:, 1], color="blue", s=20, label="Knots", zorder=5)
            
            # Plot the reversed control points
            t = np.linspace(0, curve_length, 1000)
            plt.plot(r_spline(t)[:, 0], r_spline(t)[:, 1], label="Reversed B-spline", color="purple")
            plt.scatter(reversed_control_points[:, 0], reversed_control_points[:, 1], color="black", s=30, label="Reversed Control Points")
            
            # Plot the reversed knots
            reversed_knot_positions = r_spline(knots[self.degree:-self.degree])
            plt.scatter(reversed_knot_positions[:, 0], reversed_knot_positions[:, 1], color="yellow", s=20, label="Reversed Knots", zorder=5)
        
        plt.title('B-Spline Curves Fitted to Lidar Segments with Control Point Centroids')
        plt.legend()
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

            
    def visualize_continues(self):
        """Continuously visualize lidar segments, fitted B-spline curves, control points, and centroids."""
        if self.Collocation_Matrix_list is None:
            raise ValueError("You must fit the B-splines before getting results.")
        
        plt.clf()  # Clear the current figure
    
            
        for i, lidar_segment in enumerate(self.lidar_segments):
            curve_length = self.curve_length_list[i]
            spline = self.spline_list[i]
            control_points = self.control_points_list[i]
            knots = self.knots_list[i]
            r_spline = self.r_spline_list[i]
            reversed_control_points = self.reversed_control_points_list[i]
            
            # Skip if no valid spline is found
            if not isinstance(spline, BSpline):
                continue
            
            # Generate points on the B-spline
            t = np.linspace(0, curve_length, 1000)
            spline_points = spline(t)

            # Plot noisy data points
            plt.scatter(lidar_segment[:, 0], lidar_segment[:, 1], s=10, color="orange", label="Noisy Data")

            # Plot the fitted B-spline
            plt.plot(spline_points[:, 0], spline_points[:, 1], color="red", label="Fitted B-spline")

            # Plot control points
            plt.scatter(control_points[:, 0], control_points[:, 1], color="green", s=50, label="Control Points")
            
            # Plot knots
            knot_positions = spline(knots[self.degree:-self.degree])  # Interior knots
            plt.scatter(knot_positions[:, 0], knot_positions[:, 1], color="blue", s=20, label="Knots", zorder=5)
            
            # Plot reversed B-spline and control points
            plt.plot(r_spline(t)[:, 0], r_spline(t)[:, 1], color="purple", label="Reversed B-spline")
            plt.scatter(reversed_control_points[:, 0], reversed_control_points[:, 1], color="black", s=30, label="Reversed Control Points")
            
            # Plot reversed knots
            reversed_knot_positions = r_spline(knots[self.degree:-self.degree])
            plt.scatter(reversed_knot_positions[:, 0], reversed_knot_positions[:, 1], color="yellow", s=20, label="Reversed Knots", zorder=5)
        
        # Customize plot appearance
        plt.title('B-Spline Curves Fitted to Lidar Segments with Control Point Centroids')
        # plt.arrow(0, 0, 0.3, 0, head_width=0.15, head_length=0.3, fc='k', ec='k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        
        plt.legend()
        plt.draw()
        
        plt.pause(0.01)  # Pause to update the plot



# Example usage
def main():
    lidar_segments = [
        # np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),  # Example straight segment
        # np.array([[0, 0], [1, 0] ]),  # Example straight segment
        np.array([[0, 0], [0.8, 0] ]), # Gives an error
        np.array([[4, 0], [4.5, 0.2], [5.5, 0.9], [5.9, 1.5], [6, 2]])  # Example curved segment
    ]

    bspline_fitter = BSplineFitter()
    bspline_fitter.feed_lidar_segments(lidar_segments)  
    bspline_fitter.fit_all_segments(knot_spacing=1)
    # bspline_fitter.fit_all_segments(knot_distance=0.5) #Over fitting with reversed spline
    bspline_fitter.visualize()

if __name__ == "__main__":
    main()
