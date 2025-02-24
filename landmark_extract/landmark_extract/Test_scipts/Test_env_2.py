import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

class ProbabilisticOccupancyGrid:
    def __init__(self, grid_size=(100, 100), resolution=0.5):
        self.grid_size = grid_size
        self.resolution = resolution
        self.log_odds_grid = np.zeros(grid_size)  # Log-odds initialized to 0 (unknown)

        # Log-odds values for occupancy updates
        self.occ_log_odds = np.log(0.7 / 0.3)  # Increase occupancy probability
        self.free_log_odds = np.log(0.3 / 0.7)  # Decrease occupancy probability

    def to_grid_coords(self, x, y):
        """Convert world coordinates to grid indices with clamping."""
        x_idx = min(max(int(x / self.resolution), 0), self.grid_size[1] - 1)
        y_idx = min(max(int(y / self.resolution), 0), self.grid_size[0] - 1)
        return x_idx, y_idx

    def bresenham_line(self, x0, y0, x1, y1, mark_free=True):
        """Bresenham's line algorithm with probabilistic updates."""
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy

        while True:
            if 0 <= y0 < self.grid_size[0] and 0 <= x0 < self.grid_size[1]:
                if mark_free:
                    self.log_odds_grid[y0, x0] += self.free_log_odds  # Mark as free
                else:
                    self.log_odds_grid[y0, x0] += self.occ_log_odds  # Mark as occupied

            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def update_probabilistic_grid(self, control_points, robot_position):
        """Updates the occupancy grid with spline-based scan data, marking free space."""
        control_points = np.array(control_points).T
        tck, _ = splprep(control_points, s=0)

        # Sample points along the spline
        u = np.linspace(0, 1, num=200)
        spline_points = np.array(splev(u, tck)).T

        # Convert robot position to grid
        robot_x, robot_y = self.to_grid_coords(*robot_position)

        # Apply Bresenham’s algorithm for free space and spline marking
        for i in range(len(spline_points) - 1):
            x0, y0 = self.to_grid_coords(*spline_points[i])
            x1, y1 = self.to_grid_coords(*spline_points[i + 1])

            # Mark free space from the robot to the spline
            self.bresenham_line(robot_x, robot_y, x0, y0, mark_free=True)

            # Mark the actual spline as occupied
            self.bresenham_line(x0, y0, x1, y1, mark_free=False)

    def get_probability_grid(self):
        """Convert log-odds back to probability values."""
        return 1 - 1 / (1 + np.exp(self.log_odds_grid))

    def plot_grid(self):
        """Visualizes the occupancy grid."""
        plt.imshow(self.get_probability_grid(), cmap="gray_r", origin="lower", vmin=0, vmax=1)
        plt.colorbar(label="Occupancy Probability")
        plt.title("Probabilistic Occupancy Grid with Free Space Marking")
        plt.show()

# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    grid = ProbabilisticOccupancyGrid()

    # Simulated robot position (grid coordinates)
    robot_position = (5, 5)

    # Simulated scans arriving over time
    scan_data = [
        [(5, 5), (10, 20), (20, 30), (30, 25), (40, 10), (50, 5)],
        [(6, 6), (12, 18), (22, 28), (32, 23), (42, 12), (52, 6)],
        [(7, 7), (14, 22), (24, 32), (34, 27), (44, 15), (54, 8)],
        [(8, 9), (16, 24), (26, 34), (36, 29), (46, 18), (56, 10)],
        [(10, 10), (18, 26), (28, 36), (38, 31), (48, 20), (58, 12)],
    ]

    for i, scan in enumerate(scan_data):
        print(f"Updating with scan {i+1}")
        grid.update_probabilistic_grid(scan, robot_position)
        grid.plot_grid()
