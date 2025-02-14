# PARAMETERS:
# PUBLISHES:
#	/scan_segments (visualization_msgs::msg::MarkerArray): segmented scan points
# SUBSCRIBES:
#	/scan (sensor_msgs::msg::LaserScan): laser scans
# SERVICE CLIENTS:

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from slam_interfaces.msg import BsplineArray, BsplineSegment
from std_msgs.msg import Float64MultiArray, String
import json

import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the directory containing the module to sys.path
# This is neccessary since the module is in the same directroy as the node, but we will probably be in the workspace directory when running it
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


# from Bezierfit import BezierCurveFitter
from Bezierfit_V2 import BezierCurveFitter
# from BSplinefit import BSplineFitter
# from BSpline_V2 import BSplineFitter
from BSplinefit_V3 import BSplineFitter
from scipy.interpolate import interpolate
  
class myNode(Node):
	def __init__(self):
		super().__init__("Lidar_proccesing_node")  
		# self.laserscan_sub = self.create_subscription(LaserScan, "/a200_1057/sensors/lidar2d_0/scan", self.scan_callback, 10) # For Husky robot
		self.laserscan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10) # For F1tenth car or simulation
  
  		# Publisher for visualization
		self.segment_publisher = self.create_publisher(MarkerArray, "/scan_segments", 10)
		self.publisher = self.create_publisher(BsplineArray, '/bspline_curves', 10)
		self.data_publisher = self.create_publisher(String, "/measurement_data", 10)
		# self.publisher = self.create_publisher(String, '/pseudo_matrices', 10)
		# self.range_bearing_publisher = self.create_publisher(String, "/range_bearing_segments", 10)
		
		self.alpha_max = np.pi / 4  # Relative angular threshold
		self.eta = 2 #Reletive lenght threshold	
		self.min_point_in_segment = 6  # Sets the minimum allowable points in a segment to avoid segments with only 1 point
		self.min_segmentation_threshold = 0.1 # Set minimum distance to avoid segmetation between sets of points that are too close to each other
		self.min_point_in_angle_check = 10 # Number of points to consider in angle check

		# Variables for storing data
		self.ranges = []
		self.angles = []
		self.lenghts = []
		self.scan_segments = []
		self.B_pseudoinverse_list = []

  		# Classes
		self.saveScanData = saveScanData()
		self.bspline_fitter = BSplineFitter()
  
		# self.create_timer(0.1, self.plot_segment_continous)
		# plt.ion()  
	
		self.get_logger().info("Lidar scan proccessing node has been started.")

	def scan_callback(self, msg: LaserScan):
		
		ranges = np.array(msg.ranges)
		angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
		self.lenghts = ranges
		self.angles = angles
  
		#Cut scan with 1
		ranges = ranges[:-1] # For F1tenh thcar
		# Check if scan and ranges are the same length
		if len(ranges) != len(angles):
			self.get_logger().error("Scan and ranges are not the same length.")
			return
		
		
		scan_segments, range_bearing_segments, excution_time = self.segment_scan(ranges, angles, self.alpha_max, self.eta, self.min_point_in_segment, self.min_segmentation_threshold, self.min_point_in_angle_check)
		self.scan_segments = scan_segments
		self.publish_segments(scan_segments)
		
  
		#Print range and bearing segments
		for i, segment in enumerate(range_bearing_segments):
			self.get_logger().info(f"Segment {i}: {len(segment)}")

  
  
		# bezier_fitter = BezierCurveFitter(scan_segments)
		# bezier_curves, control_points, centroids = bezier_fitter.fit_all_segments()
		# # bezier_fitter.visualize()
		# bezier_fitter.visualize_continues()

		# bezier_fitter = BezierCurveFitter(scan_segments, segment_length=1)
		# bezier_curves_list, control_points_list, centroids_list = bezier_fitter.fit_all_segments()
		# # bezier_fitter.visualize()
		# bezier_fitter.visualize_continues()

		# bspline_fitter = BSplineFitter(scan_segments)
		# bspline_curves_list, knot_points_list, control_points_list, centroids_list, tck_list = bspline_fitter.fit_all_segments(0.5)
		# ## bspline_fitter.visualize()
		# bspline_fitter.visualize_continues()

        #V2
		# self.bspline_fitter.feed_lidar_segments(scan_segments)
		# bspline_curves, knot_points_list, control_points_list, centroids_list, tck_list = self.bspline_fitter.fit_all_segments(knot_distance=0.5)
		# self.bspline_fitter.visualize_continues()
		# self.publish_spline_data(tck_list)
  
		# #V3
		self.bspline_fitter.feed_lidar_segments(scan_segments)
		self.bspline_fitter.fit_all_segments(knot_spacing=0.5)
		# self.bspline_fitter.calculate_knot_segment_lengths()
		# self.bspline_fitter.visualize()
		self.curve_length_list, self.knots_list, self.control_points_list, self.spline_list, self.Collocation_Matrix_list, self.B_pseudoinverse_list, self.reversed_control_points_list, self.r_spline_list = self.bspline_fitter.send_results()
		# self.bspline_fitter.visualize_continues()
		# self.bspline_fitter.fit_bspline_to_lidar(self.scan_segments[3], knot_distance=0.5)
		# self.bspline_fitter.plot_bspline()
  
		# # self.publish_range_bearing_segments(range_bearing_segments)
		# # self.publish_all_matrices(self.B_pseudoinverse_list)
		self.publish_data(self.B_pseudoinverse_list, range_bearing_segments)
		
  

		# # # Log the number of segments and their lengths
		# self.get_logger().info(f'Excution time: {excution_time}')
		# self.get_logger().info(f"Number of segments: {len(scan_segments)}")
		# self.get_logger().info(f'Range and bearing segments: {len(range_bearing_segments)}')
		# segment_lengths = [len(segment) for segment in scan_segments]
		# range_bearing_lengths = [len(segment) for segment in range_bearing_segments]
		# self.get_logger().info(f"Segment lengths: {segment_lengths}")
		# self.get_logger().info(f'Range and bearing lengths: {range_bearing_lengths}')
		# self.get_logger().info(f"Total number of points: {sum(segment_lengths)}")
  
		# # Visualisation
		# # self.visualise_scan_features(self.lenghts, self.angles)
		# self.plot_segments(scan_segments)	
		self.plot_segments_continous()
  
	def publish_all_matrices(self, matrix_list):
		matrices_data = []

		for matrix_id, matrix in enumerate(matrix_list):
			if matrix.size > 0:  # Matrix is not a placeholder
				matrices_data.append({
					'id': matrix_id,
					'rows': matrix.shape[0],
					'cols': matrix.shape[1],
					'data': matrix.flatten().tolist(),
				})
			else:  # Placeholder
				matrices_data.append({
					'id': matrix_id,
					'rows': 0,
					'cols': 0,
					'data': [],  # Empty data for placeholder
				})

		# #Print range and bearing segments
		# for matrix_id, matrix in enumerate(matrix_list):
		# 	self.get_logger().info(f"Matrix_number {matrix_id}: Colums { matrix.shape[1]}")
   
		# Serialize the data to JSON for publishing
		message = String()
		message.data = json.dumps(matrices_data)

		self.publisher.publish(message)
		self.get_logger().info(f'Published {len(matrices_data)} matrices.')
  
	def publish_range_bearing_segments(self, range_bearing_segments):
		# Convert numpy arrays to lists for JSON serialization
		range_bearing_serializable = [segment.tolist() for segment in range_bearing_segments]
		
		# Serialize to JSON
		range_bearing_json = json.dumps(range_bearing_serializable)
		
		# Publish as a String message
		msg = String()
		msg.data = range_bearing_json
		self.range_bearing_publisher.publish(msg)
  
	def publish_data(self, matrix_list, range_bearing_segments):
		"""
		Publishes matrices and their corresponding range-bearing segments together in JSON format.

		Args:
			matrix_list: List of numpy arrays representing matrices.
			range_bearing_segments: List of numpy arrays representing range-bearing segments.
		"""
		if len(matrix_list) != len(range_bearing_segments):
			self.get_logger().error("The number of matrices and range-bearing segments do not match.")
			return

		serialized_data = []

		for matrix_id, (matrix, range_bearing) in enumerate(zip(matrix_list, range_bearing_segments)):
			# Process matrix data
			if matrix.size > 0:  # Matrix is not a placeholder
				serialized_data.append({
					'id': matrix_id,
					'rows': matrix.shape[0],
					'cols': matrix.shape[1],
					'data': matrix.flatten().tolist(),
					'range_bearing_data': range_bearing.tolist(),  # Serialize range-bearing data
				})
			else:  # Placeholder for matrix
				serialized_data.append({
					'id': matrix_id,
					'rows': 0,
					'cols': 0,
					'data': [],  # Empty data for placeholder
					'range_bearing_data': range_bearing.tolist(),  # Still include range-bearing data
				})

		# Serialize the combined data to JSON
		message = String()
		message.data = json.dumps(serialized_data)

		# Publish the message
		self.data_publisher.publish(message)
		self.get_logger().info(f'Published {len(serialized_data)} combined matrix and range-bearing segments.')


	def publish_segments(self, scan_segments):
		marker_array = MarkerArray()
		for i, segment in enumerate(scan_segments):
			marker = Marker()
			marker.header.frame_id = "map"  # Replace with the appropriate frame
			marker.header.stamp = self.get_clock().now().to_msg()
			marker.ns = "scan_segments"
			marker.id = i
			marker.type = Marker.LINE_STRIP
			marker.action = Marker.ADD
			marker.pose.orientation.w = 1.0

			# Set the line width and color
			marker.scale.x = 0.05  # Line width
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 0.0
			marker.color.a = 1.0

			# Add points to the marker
			for point in segment:
				p = Point()
				p.x, p.y, p.z = point[0], point[1], 0.0
				marker.points.append(p)

			marker_array.markers.append(marker)

		self.segment_publisher.publish(marker_array)

	def publish_spline_data(self, bspline_curves):
		msg = BsplineArray()

		for idx, tck in enumerate(bspline_curves):
			segment_msg = BsplineSegment()
			segment_msg.id = idx

			# Populate knots
			segment_msg.knots = Float64MultiArray(data=tck[0])

			# Populate coefficients
			segment_msg.coefficients = Float64MultiArray(data=np.ravel(tck[1]))

			# Degree of the B-spline
			segment_msg.degree = tck[2]

			msg.segments.append(segment_msg)

		self.publisher.publish(msg)

	def segment_scan(self, ranges, angles, alpha_max=np.pi/4, eta=1.5, min_point_in_segment=2, min_segment_threshold = 0.1, min_point_in_angle_check = 1):
		start_time = time.time()  # Start timing

		x_coords = np.array(ranges) * np.cos(angles)
		y_coords = np.array(ranges) * np.sin(angles)
		points = np.vstack((x_coords, y_coords)).T  # Shape (N, 2) for N points

		segments = []
		range_bearing_segments = []
		current_segment = [points[0]]  # Start with the first point
		current_range_bearing = [(ranges[0], angles[0])]  # Corresponding range and bearing
		distance_variance_detected = False
		lengths = []
		cos_alphas = []

		for i in range(1, len(points)):
			# Vector from previous point to current point
			p_i = points[i] - points[i - 1]
			p_next = points[(i + 1) % len(points)] - points[i]  # Wrap-around for the last point

			# Distance between consecutive points
			d_i = np.linalg.norm(p_i)
			d_next = np.linalg.norm(p_next)

			# Calculate angle between consecutive vectors
			cos_alpha_i = np.dot(p_i, p_next) / (np.linalg.norm(p_i) * np.linalg.norm(p_next))

			lengths.append(d_next)
			cos_alphas.append(cos_alpha_i)
   
			# #Pure angle check
			# # Determine whether to continue or start a new segment
			# if cos_alpha_i >= np.cos(self.alpha_max):
			# 	current_segment.append(points[i])
			# else:
			# 	segments.append(np.array(current_segment))
			# 	current_segment = [points[i]]

			# # # Pure distance check 
			# if max(d_i, d_next) <= self.eta * min(d_i, d_next):
			# 	current_segment.append(points[i])
			# else:
			# 	segments.append(np.array(current_segment))
			# 	current_segment = [points[i]]

			# Pure distance check with flag to detect variance
			# Check relative distance between consecutive points but not if the distances are too small
			if (max(d_i, d_next) <= eta * min(d_i, d_next)) or (d_i <= min_segment_threshold and d_next <= min_segment_threshold):
				current_segment.append(points[i])
				current_range_bearing.append((ranges[i], angles[i]))
			else:
				if not distance_variance_detected:
					current_segment.append(points[i])
					current_range_bearing.append((ranges[i], angles[i]))
					distance_variance_detected = True
				else:
					segments.append(np.array(current_segment))
					range_bearing_segments.append(np.array(current_range_bearing))
					current_segment = [points[i]]
					current_range_bearing = [(ranges[i], angles[i])]
					distance_variance_detected = False
     
		# Check if the last segment should wrap around and join with the first segment
		if current_segment:
			# Calculate the angle and distance between the last and first points
			p_last = points[0] - points[-1]
			cos_alpha_last = np.dot(p_last, points[1] - points[0]) / (np.linalg.norm(p_last) * np.linalg.norm(points[1] - points[0]))

			if cos_alpha_last >= np.cos(alpha_max):
				segments[0] = np.vstack((current_segment, segments[0]))  # Merge with the first segment
				range_bearing_segments[0] = np.vstack((current_range_bearing, range_bearing_segments[0]))
			elif len(current_segment) >= min_point_in_segment:
				segments.append(np.array(current_segment))  # Add as a separate segment if it meets the minimum length
				range_bearing_segments.append(np.array(current_range_bearing))

		# Additional segmentation based on angle within each segment
		final_segments = []
		final_range_bearing_segments = []
		for segment, rb_segment in zip(segments, range_bearing_segments):
			sub_segment = [segment[0]]  # Start with the first point of the current segment
			sub_range_bearing = [rb_segment[0]]  # Start with the first range and bearing
		
  
			
   
			# for j in range(1, len(segment) - 1):
			# 	# Vector between consecutive points within the segment
			# 	p_j = segment[j] - segment[j - 1]
			# 	p_next_j = segment[j + 1] - segment[j]
			# 	# Lengt of vectors
			# 	# d_j = np.linalg.norm(segment[j + 1]-segment[j-1])

		
			# for j in range(n, len(segment) - n):
			# 	# Vector between consecutive points within the segment
			# 	p_j = segment[j] - segment[j - n]
			# 	p_next_j = segment[j + n] - segment[j]
   
			n = min_point_in_angle_check # Number of points to consider in angle check
			for j in range(1, len(segment) - 1):
				if j < n or j >= len(segment) - n:
					p_j = segment[j] - segment[j - 1]
					p_next_j = segment[j + 1] - segment[j]
				else:
					# Vector between consecutive points within the segment
					p_j = segment[j] - segment[j - n]
					p_next_j = segment[j + n] - segment[j]

				# Calculate angle between vectors within the segment
				cos_alpha_j = np.dot(p_j, p_next_j) / (np.linalg.norm(p_j) * np.linalg.norm(p_next_j))
    
				# Lengt of vectors
				d_j = np.linalg.norm(segment[j + 1]-segment[j-1])

				# Check angle condition
				# if (cos_alpha_j >= np.cos(alpha_max)) or (d_j <= min_segment_threshold):
				if (cos_alpha_j >= np.cos(alpha_max)):
					sub_segment.append(segment[j])
					sub_range_bearing.append(rb_segment[j])
				else:
					# End current sub-segment and start a new one
					if len(sub_segment) >= min_point_in_segment:
						final_segments.append(np.array(sub_segment))
						final_range_bearing_segments.append(np.array(sub_range_bearing))
					sub_segment = [segment[j]]
					sub_range_bearing = [rb_segment[j]]

			# Add the last point of the current segment to the sub-segment ()
			sub_segment.append(segment[-1])
			sub_range_bearing.append(rb_segment[-1])
   
			# Check if the sub-segment meets the minimum length
			if len(sub_segment) >= min_point_in_segment:
				final_segments.append(np.array(sub_segment))
				final_range_bearing_segments.append(np.array(sub_range_bearing))

		# End timing
		end_time = time.time()
		execution_time = end_time - start_time

		# return segments, range_bearing_segments, execution_time	# Use this ln for the pure distance segmentation
		return final_segments, final_range_bearing_segments, execution_time #Use the line for the angle segmentation


	def visualise_scan_features(self, lenghts, angles):
		figure, ax = plt.subplots(2)
		ax[0].plot(lenghts)
		ax[0].set_title("Lenghts")
  
		ax[1].plot(angles)
		ax[1].set_title("Cos alpha")
		plt.show()

	def plot_segments(self, segments):
		"""
		Visualizes the segmented points, marking the start and end of each segment.

		Parameters:
		segments (list of np.array): List of arrays where each array represents a segment of points
		"""

		plt.figure(figsize=(10, 10))

		# Iterate over each segment and plot its points
		for i, segment in enumerate(segments):
			if len(segment) > 0:
				x_points = segment[:, 0]
				y_points = segment[:, 1]

				# Plot the segment points
				plt.plot(x_points, y_points, label=f'Segment {i+1}', marker='o', linestyle='-', markersize=4)

				# Highlight the start and end points
				plt.plot(x_points[0], y_points[0], 'go', markersize=8, label=f'Start of Segment {i+1}')
				plt.plot(x_points[-1], y_points[-1], 'ro', markersize=8, label=f'End of Segment {i+1}')

		plt.xlabel("X Position")
		plt.ylabel("Y Position")
		plt.title("Segmented Points Visualization with Start and End Points")
		plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the graph
		plt.axis("equal")
		# plt.ylim(-5.5, 5.5)
		# plt.xlim(-5.5, 5.5)
		plt.tight_layout()
		plt.grid(True)
		plt.show()
  
	def plot_segments_continous(self):
		
		segments = self.scan_segments
		plt.clf()
  
		# Plot all points
		# plt.scatter(self.lenghts * np.cos(self.angles), self.lenghts * np.sin(self.angles),s=0.8,  label="All Points", alpha=0.4, color='grey') # Husky
		plt.scatter(self.lenghts[:-1] * np.cos(self.angles), self.lenghts[:-1] * np.sin(self.angles),s=0.8,  label="All Points", alpha=0.4, color='grey') #F1tenth

		if self.scan_segments:
			

			# Iterate over each segment and plot its points
			for i, segment in enumerate(segments):
				if len(segment) > 0:
					x_points = segment[:, 0]
     
					y_points = segment[:, 1]

					# Plot the segment points
					# plt.plot(x_points, y_points, label=f'Segment {i+1}', marker='o', linestyle='-', markersize=1)
					plt.scatter(x_points, y_points, s=1,  label=f'Segment {i+1}')

					# Highlight the start and end points
					# plt.plot(x_points[0], y_points[0], 'go', markersize=8, label=f'Start of Segment {i+1}')
					# plt.plot(x_points[-1], y_points[-1], 'ro', markersize=8, label=f'End of Segment {i+1}')

			# Plot arrow at centre
			plt.arrow(0, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
			plt.xlabel("X Position")
			plt.ylabel("Y Position")
			plt.title("Segmented Points Visualization with Start and End Points")
			# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the graph
			plt.axis("equal")
			# plt.ylim(-5.5, 5.5)
			# plt.xlim(-5.5, 5.5)
			plt.xlim(-15, 15)
			plt.ylim(-15, 15)
			plt.xlim(-30, 30)
			plt.ylim(-30, 30)
			plt.tight_layout()
			plt.grid(True)
			plt.draw()
			plt.pause(0.01)  # Pause to update the plot

	def plot_segments_with_angles(self, points, segments, alpha_max, eta):
		"""
		Visualizes the segmented points, marking the start and end of each segment,
		and includes angle and distance annotations for a circular scan.
		"""
		plt.figure(figsize=(10, 10))
		ax = plt.gca()

		# Plot all points
		plt.plot(points[:, 0], points[:, 1], 'bo', markersize=5, label="All Points")

		# Iterate over each segment and plot its points
		for i, segment in enumerate(segments):
			if len(segment) > 0:
				x_points = segment[:, 0]
				y_points = segment[:, 1]

				# Plot the segment points
				plt.plot(x_points, y_points, marker='o', linestyle='-', markersize=4, label=f'Segment {i+1}')

				# Highlight the start and end points
				plt.plot(x_points[0], y_points[0], 'go', markersize=8, label=f'Start of Segment {i+1}')
				plt.plot(x_points[-1], y_points[-1], 'ro', markersize=8, label=f'End of Segment {i+1}')

				# Annotate the angles and distances for visualization
				for j in range(len(segment)):
					prev_point = segment[j - 1] if j > 0 else segment[-1]
					next_point = segment[(j + 1) % len(segment)]
					
					# Calculate angle and distance for the current point
					p_i = segment[j] - prev_point
					p_next = next_point - segment[j]
					d_i = np.linalg.norm(p_i)
					alpha_i = np.arccos(np.clip(np.dot(p_i, p_next) / (np.linalg.norm(p_i) * np.linalg.norm(p_next)), -1.0, 1.0))
					
					# Draw lines between points to represent distances
					plt.plot([prev_point[0], segment[j][0]], [prev_point[1], segment[j][1]], 'k-', alpha=0.3)
					
					# Annotate angle α_i and distance d_i
					ax.annotate(f'{alpha_i:.2f}', (segment[j][0], segment[j][1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)
					ax.annotate(f'{d_i:.2f}', (prev_point[0] + (segment[j][0] - prev_point[0]) / 2, prev_point[1] + (segment[j][1] - prev_point[1]) / 2), 
								textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

		plt.xlabel("X Position")
		plt.ylabel("Y Position")
		plt.title("Segmented Points Visualization with Start and End Points, Angles, and Distances")
		plt.legend(loc='best')
		plt.axis("equal")
		plt.grid(True)
		plt.show()
  
class saveScanData:
	def __init__(self):
		self.scanData = np.zeros((360,5))
		self.scanParameters = np.zeros((4,1))
		self.testNumber = 0
		# self.path = f'/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanData_{self.testNumber}.csv'
		# self.pPath = f'/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Scan_noise/scanParameters_{self.testNumber}.csv'
		
		self.desktop_path = os.path.expanduser("src/landmark_extract/Simple_test_data") #path on car to Desktop
		self.path = os.path.join(self.desktop_path, f"scanData_{self.testNumber}.csv")
		
	def saveScan(self):
		while os.path.exists(self.path):
			self.testNumber += 1
			self.path = os.path.join(self.desktop_path, f"scanData_{self.testNumber}.csv")
		np.savetxt(self.path, self.scanData, delimiter=',')

	def saveParameters(self):
		self.pPath = os.path.join(self.desktop_path, f"scanParameters.csv")
		np.savetxt(self.pPath, self.scanParameters, delimiter=',')
		
  
def main(args=None):
	rclpy.init(args=args)
	node = myNode()
	# rclpy.spin_once(node)
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()