import open3d as o3d
import os

class PointCloudMap:
    def __init__(self, map_name="global_map"):
        """
        Initialize the PointCloudMap class.
        :param map_name: The name of the map, which will be used for saving/loading.
        """
        self.map_name = map_name
        self.map = o3d.geometry.PointCloud()  # Initialize an empty global map

    def add_scan(self, scan_path, transform=None):
        """
        Add a new scan to the map.
        :param scan_path: The file path to the scan (e.g., PCD file).
        :param transform: A transformation matrix to apply to the scan before adding it to the map.
        """
        try:
            new_scan = o3d.io.read_point_cloud(scan_path)  # Load the new scan

            if transform is not None:
                new_scan.transform(transform)  # Apply the transformation

            self.map += new_scan  # Merge the new scan into the global map
            print(f"Added scan from {scan_path} to the map.")
        except Exception as e:
            print(f"Error adding scan {scan_path}: {e}")

    def update_map(self, new_scan, transform):
        """
        Update the global map with a new scan and its transform.
        :param new_scan: The new scan (PointCloud) to add.
        :param transform: The transformation matrix to align the scan to the map.
        """
        if transform is not None:
            new_scan.transform(transform)  # Apply the transformation
        self.map += new_scan  # Merge the transformed scan into the global map
        print("Map updated with new scan.")

    def save_map(self, file_path=None):
        """
        Save the global map to a file.
        :param file_path: The file path to save the map (default is <map_name>.pcd).
        """
        if file_path is None:
            file_path = f"{self.map_name}.pcd"
        try:
            o3d.io.write_point_cloud(file_path, self.map)
            print(f"Map saved to {file_path}.")
        except Exception as e:
            print(f"Error saving map: {e}")

    def load_map(self, file_path=None):
        """
        Load a previously saved point cloud map.
        :param file_path: The file path to load the map from (default is <map_name>.pcd).
        """
        if file_path is None:
            file_path = f"{self.map_name}.pcd"
        try:
            self.map = o3d.io.read_point_cloud(file_path)
            print(f"Map loaded from {file_path}.")
        except Exception as e:
            print(f"Error loading map: {e}")

    def visualize_map(self):
        """
        Visualize the current global map.
        """
        o3d.visualization.draw_geometries([self.map])

    def get_map(self):
        """
        Return the current point cloud map.
        :return: The current map (PointCloud object).
        """
        return self.map
