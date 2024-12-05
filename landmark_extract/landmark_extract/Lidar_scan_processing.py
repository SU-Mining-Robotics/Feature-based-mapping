#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import os
import matplotlib.pyplot as plt
import time
# from Bezierfit import BezierCurveFitter
from Bezierfit_V2 import BezierCurveFitter
# from BSplinefit import BSplineFitter
from BSpline_V2 import BSplineFitter
from scipy.interpolate import interpolate
  
class myNode(Node):
	def __init__(self):
		super().__init__("Bezier_curve_extract")  
		self.laserscan_sub = self.create_subscription(LaserScan, "/a200_1057/sensors/lidar2d_0/scan", self.scan_callback, 10)

		self.alpha_max = np.pi / 4  # Angular threshold
		self.eta = 2 # Length threshold	
		self.min_segment_length = 6  # Set minimum segment length to avoid segments with only 1 point
  
		self.saveScanData = saveScanData()
		self.lenghts = []
		self.angles = []
		self.scan_segments = []
  
		# self.create_timer(0.1, self.plot_segment_continous)
		# plt.ion()  


	def scan_callback(self, msg: LaserScan):
		
		angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
		x_coords = np.array(msg.ranges) * np.cos(angles)
		y_coords = np.array(msg.ranges) * np.sin(angles)
		points = np.vstack((x_coords, y_coords)).T  # Shape (N, 2) for N points

		scan_segments, excution_time = self.segment_scan(points, self.alpha_max, self.eta, self.min_segment_length)
		self.scan_segments = scan_segments
  
		# bezier_fitter = BezierCurveFitter(scan_segments)
		# bezier_curves, control_points, centroids = bezier_fitter.fit_all_segments()
		# # bezier_fitter.visualize()
		# bezier_fitter.visualize_continues()

		bezier_fitter = BezierCurveFitter(scan_segments, segment_length=1)
		bezier_curves, control_points, centroids = bezier_fitter.fit_all_segments()
		# bezier_fitter.visualize()
		bezier_fitter.visualize_continues()

		# bspline_fitter = BSplineFitter(scan_segments)
		# bspline_curves, knot_points, control_points,centroids = bspline_fitter.fit_all_segments(0.5)
		# ## bspline_fitter.visualize()
		# bspline_fitter.visualize_continues()
  
		# bspline_fitter = BSplineFitter(scan_segments)
		# bspline_curves, knot_points_list, control_points_list, centroids_list = bspline_fitter.fit_all_segments(knot_distance=0.5)
		# bspline_fitter.visualize_continues()

		# # Log the number of segments and their lengths
		# self.get_logger().info(f'Excution time: {excution_time}')
		# self.get_logger().info(f"Number of segments: {len(scan_segments)}")
		# segment_lengths = [len(segment) for segment in scan_segments]
		# self.get_logger().info(f"Segment lengths: {segment_lengths}")
		# # self.visualise_scan_features(self.lenghts, self.angles)
		# self.plot_segments(scan_segments)	

  
	def segment_scan(self, points, alpha_max=np.pi/4, eta=1.5, min_segment_length=2):
		
		start_time = time.time()  # Start timing
		
		segments = []
		current_segment = [points[0]]  # Start with the first point
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
			if max(d_i, d_next) <= eta * min(d_i, d_next):
				current_segment.append(points[i])
			else:
				if not distance_variance_detected:
					current_segment.append(points[i])
					distance_variance_detected = True
				else:
					segments.append(np.array(current_segment))
					current_segment = [points[i]]
					distance_variance_detected = False

		# Check if the last segment should wrap around and join with the first segment
		if current_segment:
			# Calculate the angle and distance between the last and first points
			p_last = points[0] - points[-1]
			cos_alpha_last = np.dot(p_last, points[1] - points[0]) / (np.linalg.norm(p_last) * np.linalg.norm(points[1] - points[0]))

			if cos_alpha_last >= np.cos(alpha_max):
				segments[0] = np.vstack((current_segment, segments[0]))  # Merge with the first segment
			elif len(current_segment) >= min_segment_length:
				segments.append(np.array(current_segment))  # Add as a separate segment if it meets the minimum length

		
  		# Additional segmentation based on angle within each segment
		final_segments = []
		for segment in segments:
			sub_segment = [segment[0]]  # Start with the first point of the current segment
			for j in range(1, len(segment) - 1):
				# Vector between consecutive points within the segment
				p_j = segment[j] - segment[j - 1]
				p_next_j = segment[j + 1] - segment[j]
				
				# Calculate angle between vectors within the segment
				cos_alpha_j = np.dot(p_j, p_next_j) / (np.linalg.norm(p_j) * np.linalg.norm(p_next_j))

				# Check angle condition
				if cos_alpha_j >= np.cos(alpha_max):
					sub_segment.append(segment[j])
				else:
					# End current sub-segment and start a new one
					if len(sub_segment) >= min_segment_length:
						final_segments.append(np.array(sub_segment))
					sub_segment = [segment[j]]

			# Add the last point of the current segment to the sub-segment
			sub_segment.append(segment[-1])
			if len(sub_segment) >= min_segment_length:
				final_segments.append(np.array(sub_segment))
		
		# End timing
		end_time = time.time()
		execution_time = end_time - start_time
  
		return final_segments, execution_time



	def scan_callback2(self, msg: LaserScan):
		angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
		x_coords = np.array(msg.ranges) * np.cos(angles)
		y_coords = np.array(msg.ranges) * np.sin(angles)
		points = np.vstack((x_coords, y_coords)).T  # Shape (N, 2) for N points

		segments = []
		current_segment = [points[0]]  # Start with the first point
		distance_variance_detected = False

		for i in range(1, len(points)):
			# Vector from previous point to current point
			p_i = points[i] - points[i - 1]
			p_next = points[(i + 1) % len(points)] - points[i]  # Wrap-around for the last point

			# Calculate distances
			d_i = np.linalg.norm(p_i)
			d_next = np.linalg.norm(p_next)

			# Calculate the angle between consecutive vectors
			cos_alpha_i = np.dot(p_i, p_next) / (np.linalg.norm(p_i) * np.linalg.norm(p_next))

			# Check the angle and distance conditions
			angle_condition = cos_alpha_i >= np.cos(self.alpha_max)
			distance_condition = max(d_i, d_next) <= self.eta * min(d_i, d_next)

			if angle_condition and distance_condition:
				# Conditions met, continue the current segment
				current_segment.append(points[i])
			else:
				# Start a new segment if conditions are not met
				segments.append(np.array(current_segment))
				current_segment = [points[i]]

		# Finalize the last segment by merging with the first if conditions allow
		if current_segment:
			p_last = points[0] - points[-1]
			cos_alpha_last = np.dot(p_last, points[1] - points[0]) / (np.linalg.norm(p_last) * np.linalg.norm(points[1] - points[0]))
			
			if (cos_alpha_last >= np.cos(self.alpha_max) and
				max(np.linalg.norm(p_last), np.linalg.norm(points[1] - points[0])) <= self.eta * min(np.linalg.norm(p_last), np.linalg.norm(points[1] - points[0]))):
				# Merge last segment with first if wrap-around conditions are met
				segments[0] = np.vstack((current_segment, segments[0]))  # Merge with the first segment
			else:
				segments.append(np.array(current_segment))  # Add as a separate segment
				# Start a new segment
				current_segment = [points[i]]
    
		# Log the number of segments and their lengths
		self.get_logger().info(f"Number of segments: {len(segments)}")
		segment_lengths = [len(segment) for segment in segments]
		self.get_logger().info(f"Segment lengths: {segment_lengths}")
		self.visualise_scan_features(self.lenghts, self.angles)
		self.plot_segments(segments)	
		
		# min_segment_length = 2  # Set minimum segment length to avoid segments with only 1 point
		# # Check if the last segment should wrap around and join with the first segment
		# if current_segment:
		# 	# Calculate the angle and distance between the last and first points
		# 	p_last = points[0] - points[-1]
		# 	cos_alpha_last = np.dot(p_last, points[1] - points[0]) / (np.linalg.norm(p_last) * np.linalg.norm(points[1] - points[0]))

		# 	if cos_alpha_last >= np.cos(self.alpha_max):
		# 		segments[0] = np.vstack((current_segment, segments[0]))  # Merge with the first segment
		# 	elif len(current_segment) >= min_segment_length:
		# 		segments.append(np.array(current_segment))  # Add as a separate segment if it meets the minimum length

		# # Log the number of segments and their lengths
		# self.get_logger().info(f"Number of segments: {len(segments)}")
		# segment_lengths = [len(segment) for segment in segments]
		# self.get_logger().info(f"Segment lengths: {segment_lengths}")
		# self.visualise_scan_features(self.lenghts, self.angles)
		self.plot_segments(segments)	
  
		# Handle any single-point segments by merging with neighbors
		final_segments = []
		for segment in segments:
			if len(segment) == 1:
				# Try to merge with the last segment if angle and distance conditions allow
				if final_segments and len(final_segments[-1]) > 1:
					last_point = final_segments[-1][-1]
					p_last_to_single = segment[0] - last_point
					d_last_to_single = np.linalg.norm(p_last_to_single)
					cos_alpha_merge = np.dot(p_last_to_single, final_segments[-1][-1] - final_segments[-1][-2]) / (np.linalg.norm(p_last_to_single) * np.linalg.norm(final_segments[-1][-1] - final_segments[-1][-2]))

					# Check conditions to merge with the last segment
					if cos_alpha_merge >= np.cos(self.alpha_max) and d_last_to_single <= self.eta * np.linalg.norm(final_segments[-1][-1] - final_segments[-1][-2]):
						final_segments[-1] = np.vstack((final_segments[-1], segment))
					else:
						final_segments.append(segment)
				else:
					final_segments.append(segment)
			else:
				final_segments.append(segment)

		# Log the number of segments and their lengths
		self.get_logger().info(f"Number of segments: {len(final_segments)}")
		segment_lengths = [len(segment) for segment in final_segments]
		self.get_logger().info(f"Segment lengths: {segment_lengths}")
		self.visualise_scan_features(self.lenghts, self.angles)
		self.plot_segments(final_segments)


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
  
	def plot_segment_continous(self):
		
		segments = self.scan_segments
		plt.clf()
		if self.scan_segments:
			

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
  
def interpolate_track_new(points, n_points=None, s=0):
    if len(points) <= 1:
        return points
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
    return track

def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
    if points[0, 0] > points[-1, 0]:
        points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / seperation_distance), 2)
    smooth_line = interpolate_track_new(points, None, smoothing)
    resampled_points = interpolate_track_new(smooth_line, n_pts, 0)

    return resampled_points, smooth_line
    
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