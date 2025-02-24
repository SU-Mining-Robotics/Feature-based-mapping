import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from scipy.spatial import KDTree
from scipy.linalg import svd
import json
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from landmark_extract import utils as Utils
import select
import matplotlib.pyplot as plt

import sys
import os

# Get the absolute path of the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from ICP_scan_to_scan import ICPScanToScan
from ICP_scan_to_scan import ICPScanToScan

class ScanToScanMatching(Node):
    def __init__(self):
        super().__init__('scan_to_scan_matching')
    
        # Declare parameters
        self.declare_parameter('measurement_topic', "/measurement_data")

        # Subscribers
        self.subscription = self.create_subscription(LaserScan, '/scan', self.laserscan_callback, 10)
        self.subscription = self.create_subscription(String, self.get_parameter('measurement_topic').value, self.measurement_CB, 10)
        self.gt_subscriber = self.create_subscription(ModelStates, '/model_states', self.model_callback, 10)
        
        self.prev_pcd = None
        self.prev_transform = np.eye(3)
        self.scan_scan_matching = ICPScanToScan(max_iterations=20, tolerance=1e-5, visualize=False)
        self.scan_map_matching = ICPScanToScan(max_iterations=200, tolerance=1e-5, visualize=False)
        self.intermediate_estimate = np.eye(3)
        self.estimate = np.eye(3)
        self.Map = None
        
        self.ground_truth_trajectory = []
        self.odom_trajectory = []
        self.estimated_trajectory = []
        
        # obect to save data
        self.flag = 0
        self.ds = saveToFile()
        self.iteration = 0
        self.process_interval = 10  # Only process every 10th message
        self.message_counter_gt = 0
        self.message_counter_odom = 0
        self.message_counter_odom = 0

    def laserscan_callback(self, msg):
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = []

        for r, theta in zip(msg.ranges, angles):
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y])  # Z = 0 for 2D lidar scans
        
        current_pcd = np.array(points)
        
        if self.prev_pcd is not None:
            # transformation = self.icp_manual(self.prev_pcd, current_pcd, max_icp_iterations=20, tolerance=1e-5)

            # print("Odometry scanmatching")
            odom_transformation, _, _ = self.scan_scan_matching.icp_scanmatching(self.prev_pcd, current_pcd, np.eye(3))
            # print(odom_transformation)
            self.prev_transform = self.prev_transform @ odom_transformation
            self.intermediate_estimate = self.estimate @ odom_transformation
            
            # Extract x, y, and yaw from the transformation matrix
            x, y = self.prev_transform[0, 2], self.prev_transform[1, 2]
            yaw = np.arctan2(self.prev_transform[1, 0], self.prev_transform[0, 0])  # Extract rotation in radians

            # Format values to two decimal places
            translation_str = f"x: {x:.2f}, y: {y:.2f}"
            yaw_str = f"yaw: {yaw:.2f} rad"
          
            
            formatted_matrix = np.array2string(self.prev_transform, formatter={'float_kind': lambda v: f"{v:.2f}"})
            # self.get_logger().info(f'Pose:\nTranslation: [{translation_str}]\nRotation: [{yaw_str}]')
            # self.get_logger().info(f'Transformation:\n{self.prev_transform}')
            
            self.ds.tempData[6] = x
            self.ds.tempData[7] = y
            self.ds.tempData[8] = 0
            self.ds.tempData[9] = yaw
            
            self.odom_trajectory.append(np.array([x, y, yaw]))
        
        self.prev_pcd = current_pcd
        self.message_counter_odom += 1
        
        # if np.linalg.norm(current_pcd - self.prev_pcd ) < 1e-2:  # Check for small scan differences
        #     print("Skipping map ICP update as the robot is stationary")
        #     map_flag = False
        # else:
        #     map_flag = True

        # Update map every 10th scan
        if self.message_counter_odom % 10 == 0:
            # if map_flag:
            if self.Map is None:
                self.Map = current_pcd
            else:
                
                #Localize the scan in the global map
                print("Global scanmatching")
                transformation, transformation_correction, _, final_error = self.scan_map_matching.icp_scanmatching_map(self.Map, current_pcd, self.intermediate_estimate)
                # transformation = np.eye(3)  # Disable map ICP for now
                print(f"Error: {final_error}")
                self.estimate = self.intermediate_estimate @ transformation_correction
                
                #Update the global map
                self.Map = self.update_map(self.Map, current_pcd, self.estimate)
                
        else:
            self.estimate = self.intermediate_estimate
            
         #Store the estimated trajectory
        x, y = self.estimate[0, 2], self.estimate[1, 2]
        yaw = np.arctan2(self.estimate[1, 0], self.estimate[0, 0])  # Extract rotation in radians
        self.estimated_trajectory.append(np.array([x, y, yaw]))
        
        x =self.ds.tempData[2] 
        y = self.ds.tempData[3] 
        yaw = self.ds.tempData[5]
        self.ground_truth_trajectory.append(np.array([x, y, yaw]))
            
    def plot_map(self, global_map, transformed_scan=None):
        """Plot the global map"""
        print("Map size:", global_map.shape)
        
        # Setup real-time visualization
        plt.ion()  # Interactive mode ON
        plt.clf()
        
        # Initial plot
        target_scatter = plt.scatter(global_map[:, 0], global_map[:, 1], color='black', label="Map (Reference)", s=0.4)
        source_scatter = plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], color='#1f77b4', label="New scan", alpha=0.5, s=5)
        
        #Plot trajectories
        odom_trajectory = np.array(self.odom_trajectory)
        estimated_trajectory = np.array(self.estimated_trajectory)
        ground_truth_trajectory = np.array(self.ground_truth_trajectory)
        plt.plot(odom_trajectory[:, 0], odom_trajectory[:, 1], color='red', label="Odometry Trajectory")
        plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], color='blue', label="Estimated Trajectory")
        plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], color='green', label="Ground Truth Trajectory")
        
        plt.legend()
        plt.draw()
        plt.axis('equal')   
        plt.pause(0.1)
            
    def update_map(self, global_map, scan, transformation):
        """Update the global map with the transformed scan"""
        transformed_scan = self.transform_scan(scan, transformation)
        global_map = np.vstack((global_map, transformed_scan))
        
        # # Apply 2D grid downsampling
        # global_map = grid_downsample(global_map, grid_size)
        
        # Merge close points
        merge_dist=0.05
        global_map = self.merge_close_points_2d(global_map, merge_dist)
        
        self.plot_map(global_map, transformed_scan)
        
        return global_map
    
    def update_map(self, global_map, scan, transformation, min_dist=0.05):
        transformed_scan = self.transform_scan(scan, transformation)
        global_map = self.filter_close_points_2d(global_map, transformed_scan, min_dist)

        self.plot_map(global_map, transformed_scan)
        return global_map
        
    def filter_close_points_2d(self, global_map, new_points, min_dist=0.05):
        """Removes new points that are too close to existing ones."""
        if len(global_map) == 0:
            return new_points  # No previous points, keep all new ones

        tree = KDTree(global_map)
        distances, _ = tree.query(new_points)
        filtered_points = new_points[distances > min_dist]
        return np.vstack((global_map, filtered_points))
        
    def merge_close_points_2d(self, points, merge_dist=0.05):
        """Merges nearby points by averaging their positions."""
        tree = KDTree(points)
        clusters = tree.query_ball_tree(tree, merge_dist)

        new_points = []
        visited = set()
        for i, cluster in enumerate(clusters):
            if i in visited:
                continue
            cluster_points = points[cluster]
            new_points.append(cluster_points.mean(axis=0))
            visited.update(cluster)

        return np.array(new_points)
    
    def transform_scan(self, scan, transformation):
        """Apply a 2D transformation to a set of points"""
        tx, ty = transformation[0, 2], transformation[1, 2]
        theta = np.arctan2(transformation[1, 0], transformation[0, 0])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return scan @ rotation_matrix.T + np.array([tx, ty])
    
    def grid_downsample(self, points, grid_size):
        """Reduces the number of points by keeping one per 2D grid cell."""
        grid_indices = np.floor(points / grid_size).astype(int)
        unique_indices, unique_idx = np.unique(grid_indices, axis=0, return_index=True)
        return points[unique_idx]
    
    def model_callback(self, model_data):
		
        self.message_counter_gt += 1
        if self.message_counter_gt % self.process_interval != 0:
            return

        try:
            robot_index = model_data.name.index("my_bot")
            robot_pose = PoseStamped()
            robot_pose.header.stamp = self.get_clock().now().to_msg()
            robot_pose.pose = model_data.pose[robot_index]
            
            time_s = robot_pose.header.stamp.sec
            time_ns = robot_pose.header.stamp.nanosec
            x = robot_pose.pose.position.x
            y = robot_pose.pose.position.y
            z = robot_pose.pose.position.z
            yaw = Utils.quaternion_to_angle(robot_pose.pose.orientation)


            self.ds.tempData[0] = time_s
            self.ds.tempData[1] = time_ns
            self.ds.tempData[2] = x
            self.ds.tempData[3] = y
            self.ds.tempData[4] = z
            self.ds.tempData[5] = yaw

            self.iteration += 1
            # self.get_logger().info('Iteration: '+str(self.iteration))
            # self.get_logger().info(str(self.ds.tempData))

            self.ds.saveData = np.vstack((self.ds.saveData, self.ds.tempData))
            self.ds.saveFlag = self.flag
            # self.get_logger().info('self.ds.saveFlag: '+str(self.ds.saveFlag))
            if self.ds.saveFlag:
                self.get_logger().info(str(self.ds.tempData))
                self.ds.saveToFile(self.ds.saveData)
                self.get_logger().info('Saving to file')
                # self.ds.saveFlag = False
                rclpy.shutdown()

            # self.get_logger().info(f"Robot position: {robot_pose.pose.position}")
            # self.get_logger().info(f"Robot orientation: {robot_pose.pose.orientation}")

        except ValueError:
            self.get_logger().warn("Robot 'my_bot' not found in model states.")
   
    def check_key_press(self):
        # Check if a key is pressed without blocking
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input_char = sys.stdin.readline().strip()
            if input_char == "":  # Enter key pressed
                self.flag = True
                self.message_counter_gt = 9
                self.message_counter_odom = 9
                self.get_logger().info("Flag is set. Shutting down...")
    
    def measurement_CB(self, msg):
        """
        Callback to process the received message.
        """
        try:
            # Deserialize the JSON data
            combined_data = json.loads(msg.data)
            self.measurement_data = []
            self.measurement_data = combined_data

            # Process each entry in the combined data
            for entry in combined_data:
                matrix_id = entry.get('id')
                rows = entry.get('rows')
                cols = entry.get('cols')
                matrix_data = entry.get('data')
                range_bearing_data = entry.get('range_bearing_data')

                self.get_logger().info(f"Received matrix ID: {matrix_id}")
                # self.get_logger().info(f"Matrix dimensions: {rows}x{cols}")
                # self.get_logger().info(f"Matrix data: {matrix_data}")
                # self.get_logger().info(f"Range-bearing data: {range_bearing_data}")

                # If needed, reconstruct the matrix from the flattened data
                if rows > 0 and cols > 0:
                    matrix = np.array(matrix_data).reshape(rows, cols)
                    # self.get_logger().info(f"Reconstructed matrix:\n{matrix}")

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to decode JSON: {e}")
            
class saveToFile:
	def __init__(self):
		self.lap = 0
		# 						  sec,nsec,x,y,z,w,x,y,z,w
		self.saveData = np.array([0,0,0,0,0,0,0,0,0,0],dtype='f')
		self.tempData = np.array([0,0,0,0,0,0,0,0,0,0],dtype='f')
		self.saveFlag = False

	def saveToFile(self,data):
		np.savetxt(fname='/home/ruan/dev_ws/src/landmark_extract/GT_vs_Odom.csv', 
			 X=data, 
			 delimiter=',',
			#  newline=' /n', 
			#  fmt='%.2f',
			 header='seconds, nanoseconds, trueX, trueY, trueZ, trueYaw, odomX, odomY, odomZ, odomYaw',
			 )
		# path = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/'+filename+'.csv'
		# # np.savetxt(fname=path, X=data, delimiter=',',newline='/n', fmt='%1.3f')
		# f = open(path, 'a')
		# f.write(str(data[0])+','+str(data[1])+','+str(data[2])+'\n')
		# f.close()
  
def main(args=None):
    # rclpy.init(args=args)
    # node = ScanToScanMatching()
    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()
    
    rclpy.init(args=args)
    node = ScanToScanMatching()
    # rclpy.spin(node)
    while rclpy.ok() and not node.flag:
            rclpy.spin_once(node) 
            node.check_key_press()  # Check if a key was pressed
    rclpy.spin_once(node)
    rclpy.spin_once(node)


    rclpy.shutdown()

if __name__ == '__main__':
    main()
