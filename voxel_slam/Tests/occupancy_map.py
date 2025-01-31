import numpy as np
import matplotlib.pyplot as plt

class ProbabilisticOccupancyGrid:
    def __init__(self, resolution, initial_size=10, p_occ=0.7, p_free=0.3, uniform_expand=False):
        self.resolution = resolution
        self.p_occ = np.log(p_occ / (1 - p_occ))
        self.p_free = np.log(p_free / (1 - p_free))
        
        # Initial grid size
        self.grid_size_x = int(initial_size / resolution)
        self.grid_size_y = int(initial_size / resolution)
        self.log_odds = np.zeros((self.grid_size_x, self.grid_size_y))
        
        # Fixed world origin
        self.origin = np.array([0, 0])
        self.extents = np.array([self.grid_size_x, self.grid_size_y])  
        self.uniform_expand = uniform_expand  # Option to expand uniformly

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        x_idx = int((x - self.origin[0]) / self.resolution + self.extents[0] / 2)
        y_idx = int((y - self.origin[1]) / self.resolution + self.extents[1] / 2)
        return x_idx, y_idx

    def expand_grid(self, new_x, new_y):
        """Expand the grid non-uniformly (or uniformly if enabled)."""
        x_idx, y_idx = self.world_to_grid(new_x, new_y)

        if 0 <= x_idx < self.extents[0] and 0 <= y_idx < self.extents[1]:
            return  # No expansion needed

        # Determine required new extents
        min_x = min(0, x_idx)
        max_x = max(self.extents[0], x_idx + 1)
        min_y = min(0, y_idx)
        max_y = max(self.extents[1], y_idx + 1)

        if self.uniform_expand:
            # Expand uniformly in all directions
            max_dim = max(max_x, max_y)
            new_extents = np.array([max_dim, max_dim])
        else:
            # Expand just enough to fit the new scan
            new_extents = np.array([max_x, max_y])

        # Create new grid and copy old data
        new_log_odds = np.zeros((new_extents[0], new_extents[1]))
        
        # Compute offsets to align old data within the new grid
        x_offset = (new_extents[0] - self.extents[0]) // 2
        y_offset = (new_extents[1] - self.extents[1]) // 2
        
        new_log_odds[x_offset:x_offset + self.extents[0], y_offset:y_offset + self.extents[1]] = self.log_odds
        
        # Update extents and log odds
        self.extents = new_extents
        self.log_odds = new_log_odds

    def update(self, robot_pose, lidar_ranges, lidar_angles, max_range):
        x_r, y_r, theta_r = robot_pose

        # Expand grid if needed
        for r, angle in zip(lidar_ranges, lidar_angles):
            if r >= max_range:
                continue
            
            end_x = x_r + r * np.cos(theta_r + angle)
            end_y = y_r + r * np.sin(theta_r + angle)
            
            self.expand_grid(end_x, end_y)

        # Perform occupancy updates
        for r, angle in zip(lidar_ranges, lidar_angles):
            if r >= max_range:
                continue
            
            end_x = x_r + r * np.cos(theta_r + angle)
            end_y = y_r + r * np.sin(theta_r + angle)
            
            x_idx, y_idx = self.world_to_grid(end_x, end_y)
            rx_idx, ry_idx = self.world_to_grid(x_r, y_r)
            
            # Get free cells along the ray
            free_cells = self.bresenham(rx_idx, ry_idx, x_idx, y_idx)
            
            for (fx, fy) in free_cells:
                if 0 <= fx < self.extents[0] and 0 <= fy < self.extents[1]:
                    self.log_odds[fx, fy] += self.p_free  # Mark free space
        
            if 0 <= x_idx < self.extents[0] and 0 <= y_idx < self.extents[1]:
                self.log_odds[x_idx, y_idx] += self.p_occ  # Mark occupied space
    
    def bresenham(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for ray tracing."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while (x0, y0) != (x1, y1):
            cells.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return cells

    def get_probability_map(self):
        """Convert log odds to probability values."""
        return 1 - 1 / (1 + np.exp(self.log_odds))
    
    def plot_map(self):
        """Plot the occupancy grid."""
        plt.imshow(self.get_probability_map(), cmap='gray_r', origin='lower', extent=[-self.extents[0]//2, self.extents[0]//2, -self.extents[1]//2, self.extents[1]//2])
        plt.colorbar(label='Occupancy Probability')
        plt.show()

# **Test Functions**
def test_moving_robot_map():
    # Initialize the grid with dimensions and resolution
    grid = ProbabilisticOccupancyGrid(resolution=0.1, initial_size=5, uniform_expand=True)

    # First test: Robot at the center (0, 0) and lidar scan
    robot_pose_1 = (0, 0, 0)  # Robot at the center of the map
    angles = np.linspace(0, 2 * np.pi, 360)  # Full lidar scan around the robot
    ranges = np.full_like(angles, 3.0)  # Circular lidar readings at 3m

    grid.update(robot_pose_1, ranges, angles, max_range=3.5)  # Update grid with first scan
    grid.plot_map()  # Plot the map after the first update
    
    # Second test: Robot moves to a new position (e.g., (2, 2)) and scans again
    robot_pose_2 = (2, 2, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_2 = np.full_like(angles, 3.5)  # Lidar readings at 3.5m
    grid.update(robot_pose_2, ranges_2, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update

def test_dynamic_expansion():
    grid = ProbabilisticOccupancyGrid(resolution=0.1, initial_size=5, uniform_expand=True)

    # First scan: robot in a corridor
    robot_pose_1 = (1, 1, 0)
    angles = np.linspace(-np.pi / 4, np.pi / 4, 90)
    ranges = np.full_like(angles, 3.0)
    
    grid.update(robot_pose_1, ranges, angles, max_range=3.5)
    grid.plot_map()

    # Second scan: robot moves into an open room
    robot_pose_2 = (5, 8, np.pi / 2)
    ranges_2 = np.full_like(angles, 6.0)
    
    grid.update(robot_pose_2, ranges_2, angles, max_range=6.5)
    grid.plot_map()
    
        # Second scan: robot moves into an open room
    robot_pose_3 = (-2, -5, -np.pi / 2)
    ranges_3 = np.full_like(angles, 6.0)
    
    grid.update(robot_pose_3, ranges_3, angles, max_range=6.5)
    grid.plot_map()

if __name__ == "__main__":
    test_moving_robot_map()
    # test_dynamic_expansion()
