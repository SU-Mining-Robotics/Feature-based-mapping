import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
from scipy.special import comb

class SplineLaserPredictor:
    def __init__(self, control_points=None, laser_angle=0.0, robot_pose=None):
        self.control_points = control_points if control_points is not None else np.array([[0, 0], [1, 1], [2, 1], [3, 0]])
        self.laser_angle = laser_angle  # Angle in radians
        self.robot_pose = robot_pose if robot_pose is not None else [0, 0, 0]  # [x, y, theta]

    def get_control_points(self):
        return self.control_points

    def set_control_points(self, control_points):
        self.control_points = np.array(control_points)

    def get_laser_angle(self):
        return self.laser_angle

    def set_laser_angle(self, laser_angle):
        self.laser_angle = laser_angle

    def get_robot_pose(self):
        return self.robot_pose

    def set_robot_pose(self, robot_pose):
        self.robot_pose = robot_pose

    def rotate_and_translate(self):
        xr, yr, theta = self.robot_pose
        mu_p = theta + self.laser_angle

        rotation_matrix = np.array([[np.cos(mu_p), np.sin(mu_p)],
                                     [-np.sin(mu_p), np.cos(mu_p)]])
        transformed_points = []
        for x, y in self.control_points:
            translated = np.array([x - xr, y - yr])
            transformed = rotation_matrix @ translated
            transformed_points.append(transformed)

        return np.array(transformed_points)

    @staticmethod
    def bernstein_basis(n, i, t):
        """Compute the Bernstein basis polynomial of degree n."""
        return comb(n, i) * (1 - t)**(n - i) * t**i

    @staticmethod
    def bernstein_derivative(n, i, t):
        """Compute the derivative of the Bernstein basis polynomial."""
        if i > 0:
            return comb(n, i) * (i * t**(i - 1) * (1 - t)**(n - i) - (n - i) * t**i * (1 - t)**(n - i - 1))
        else:
            return -n * (1 - t)**(n - 1)

    @staticmethod
    def spline_function(control_points, t):
        """Evaluate a cubic Bézier curve for parameter t."""
        n = len(control_points) - 1  # Degree of the spline
        point = np.zeros(2)  # To store the evaluated point
        for i in range(n + 1):
            bernstein = SplineLaserPredictor.bernstein_basis(n, i, t)
            point += bernstein * control_points[i]
        return point

    @staticmethod
    def spline_derivative(control_points, t):
        """Evaluate the derivative of a cubic Bézier curve at parameter t."""
        n = len(control_points) - 1  # Degree of the spline
        derivative = np.zeros(2)
        for i in range(n + 1):
            bernstein_der = SplineLaserPredictor.bernstein_derivative(n, i, t)
            derivative += bernstein_der * control_points[i]
        return derivative

    def compute_tangent_angle(self, t_star):
        """Compute the tangent angle at t_star."""
        derivative = self.spline_derivative(self.control_points, t_star)
        tangent_angle = np.arctan2(derivative[1], derivative[0])  # atan2(dy, dx)
        return tangent_angle

    def predict_measurement(self):
        """Predict the laser measurement for a single laser beam."""
        transformed_points = self.rotate_and_translate()

        # Define a function for finding the root (sy = 0)
        def sy_root(t):
            return self.spline_function(transformed_points, t)[1]

        initial_guesses = np.linspace(0, 1, 10)  # Initial guesses for t
        for t_initial in initial_guesses:
            try:
                t_star = newton(sy_root, t_initial)  # Newton-Raphson to find the root
                if 0 <= t_star <= 1:
                    predicted_distance = self.spline_function(transformed_points, t_star)[0]
                    tangent_angle = self.compute_tangent_angle(t_star)  # Tangent angle at t_star
                    return predicted_distance, t_star, tangent_angle, transformed_points
            except RuntimeError:
                continue
        return 0.0, 0, 0.0, transformed_points
    
    def predict_distances(self, angles, robot_pose, control_points):
        """
        Predict laser measurements for multiple laser beams, given an array of angles.
        
        Parameters:
            angles (array-like): List or array of angles for which to predict distances.
            robot_pose (array-like): The pose of the robot [x, y, theta].
            control_points (array-like): Control points of the spline curve.
            
        Returns:
            np.ndarray: Array of predicted distances for each angle.
        """
        distances = []

        for angle in angles:
            # Update the robot pose and control points for this calculation
            self.robot_pose = robot_pose
            self.control_points = control_points
            self.laser_angle = angle

            # Perform rotation and translation
            transformed_points = self.rotate_and_translate()

            # Define the root-finding function
            def sy_root(t):
                return self.sy_function(t, transformed_points)

            # Try multiple initial guesses for t
            initial_guesses = np.linspace(0, 1, 10)  # Customize the range and number of guesses
            predicted_distance = 0.0  # Default value if no solution is found
            for t_initial in initial_guesses:
                try:
                    t_star = newton(sy_root, t_initial)  # Use Newton-Raphson to find the root
                    # Ensure t_star is within the valid range [0, 1]
                    if 0 <= t_star <= 1:
                        predicted_distance = self.spline_function(transformed_points, t_star)[0]
                        break
                except RuntimeError:
                    continue  # Try the next initial guess

            # Append the result for this angle
            distances.append(predicted_distance)

        return np.array(distances)




    def visualize_prediction(self):
        predicted_distance, t_star, tangent_angle, transformed_points = self.predict_measurement()
        print(f"Predicted Distance: {predicted_distance:.2f} m")
        print(f"t_star: {t_star:.2f}")
        print(f"Tangent Angle: {np.degrees(tangent_angle):.2f} degrees")

        t_values = np.linspace(0, 1, 100)
        original_spline_points = np.array([self.spline_function(self.control_points, t) for t in t_values])
        transformed_spline_points = np.array([self.spline_function(transformed_points, t) for t in t_values])

        laser_origin = np.array([self.robot_pose[0], self.robot_pose[1]])
        laser_direction = np.array([np.cos(self.laser_angle + self.robot_pose[2]), 
                                    np.sin(self.laser_angle + self.robot_pose[2])])
        laser_end = laser_origin + 5 * laser_direction

        laser_line_x = [0, predicted_distance]
        laser_line_y = [0, 0]

        # Calculate intersection point and tangent vector
        intersection_point = self.spline_function(self.control_points, t_star)
        tangent_vector = self.spline_derivative(self.control_points, t_star)  # Assuming spline_derivative is implemented
        tangent_vector_normalized = tangent_vector / np.linalg.norm(tangent_vector)
        
        test_intersection_point = self.spline_function(self.control_points, 0.9)

        # Generate tangent line
        tangent_start = intersection_point - tangent_vector_normalized
        tangent_end = intersection_point + tangent_vector_normalized

        plt.figure(figsize=(8, 6))
        
        plt.plot(original_spline_points[:, 0], original_spline_points[:, 1], label="Original Spline", color="blue")
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color="green", label="Control Points")
        plt.plot([laser_origin[0], laser_end[0]], [laser_origin[1], laser_end[1]], 
                label="Laser Beam", color="orange", linestyle="--")
        plt.scatter(intersection_point[0], intersection_point[1], color="red", label="Intersection Point", zorder=5)
        plt.plot([tangent_start[0], tangent_end[0]], [tangent_start[1], tangent_end[1]], 
                label="Tangent Line", color="purple", linestyle="-")
        plt.scatter(test_intersection_point[0], test_intersection_point[1], label="Test Intersection Point", color="yellow")  
        plt.title("Original Spline and Laser Beam in Global Frame")
        plt.xlabel("x (global frame)")
        plt.ylabel("y (global frame)")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        plt.figure(figsize=(8, 6))
        plt.plot(transformed_spline_points[:, 0], transformed_spline_points[:, 1], 
                label="Transformed Spline", color="blue")
        plt.plot(laser_line_x, laser_line_y, label="Laser Beam", color="orange", linestyle="--")
        plt.scatter(predicted_distance, 0, color="red", label="Intersection Point", zorder=5)
        plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color="green", label="Transformed Control Points")
        plt.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        plt.title("Transformed Spline and Laser Beam in Local Frame")
        plt.xlabel("x (local frame)")
        plt.ylabel("y (local frame)")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        plt.show()
        
    def visualize_lidar_beams(self, angles, robot_pose, control_points):
        """
        Visualize the spline and lidar beams.
        
        Parameters:
            angles (array-like): List or array of angles for the laser beams.
            robot_pose (array-like): The pose of the robot [x, y, theta].
            control_points (array-like): Control points of the spline curve.
        """
        # Predict distances for all angles
        distances = self.predict_distances(angles, robot_pose, control_points)

        # Generate the original spline points
        t_values = np.linspace(0, 1, 100)
        original_spline_points = np.array([self.spline_function(control_points, t) for t in t_values])

        # Plot the spline and lidar beams
        plt.figure(figsize=(10, 8))
        plt.plot(original_spline_points[:, 0], original_spline_points[:, 1], label="Original Spline", color="blue")
        plt.scatter(control_points[:, 0], control_points[:, 1], color="green", label="Control Points")

        # Plot each lidar beam
        for angle, distance in zip(angles, distances):
            laser_origin = np.array([robot_pose[0], robot_pose[1]])
            laser_direction = np.array([np.cos(angle + robot_pose[2]), np.sin(angle + robot_pose[2])])
            laser_end = laser_origin + distance * laser_direction
            plt.plot([laser_origin[0], laser_end[0]], [laser_origin[1], laser_end[1]], 
                    color="orange", linestyle="--", alpha=0.7)

        plt.title("Original Spline and Lidar Beams")
        plt.xlabel("x (global frame)")
        plt.ylabel("y (global frame)")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def main(args=None):
    # Example usage
    control_points = np.array([[-2, 2], [2, 2], [2.5, 1], [3, 2], [4, 2]])
    laser_angle = np.radians(60)
    robot_pose = [0, 0, np.radians(0)]
    
    robot_pose = np.array([0.0, 0.0, 0.0])  # Robot pose [x, y, theta]
    laser_angle = -2.7405292607843876  # Laser beam angle (in radians)
    laser_angle = -2.4405292607843876  # Laser beam angle (in radians)
    laser_angle = -3.1405292607843876  # behaves wierd
    laser_angle = -3.8405292607843876  #Still acceptable
    laser_angle = -2.7405292607843876  # Wierd
    laser_angle = -2.5405292607843876
    
    control_points = np.array([
        [-2.14050467,  2.01223198],
        [-2.4965321,   1.99794561],
        [-3.1908192,   1.82273862],
        [-3.94406919,  1.02167586],
        [-4.2070522,   0.03207072],
        [-3.9686886,  -0.94890253],
        [-3.23574283, -1.77646069],
        [-2.54665145, -1.96378615],
        [-2.1909611,  -1.9891872]
    ])

    predictor = SplineLaserPredictor(control_points, laser_angle, robot_pose)
    predictor.visualize_prediction()

if __name__ == "__main__":
    main()