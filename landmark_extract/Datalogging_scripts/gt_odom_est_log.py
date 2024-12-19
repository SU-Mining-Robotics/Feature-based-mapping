#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  
#from f110_interfaces.msg import CrashStatus
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from example_interfaces.msg import Bool
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Path
import numpy as np
import sys
import select
from landmark_extract import utils as Utils

class localisation_test_accuracy(Node):
	def __init__(self):
		super().__init__("localisation_test_accuracy")
		#Subscribers 
		self.gt_subscriber = self.create_subscription(ModelStates, '/model_states', self.model_callback, 10)
		self.Odom_subscriber_ = self.create_subscription(Odometry, "/diff_cont/odom", self.OdomCallback, 10)
		self.expected_pose_sub = self.create_subscription(PoseStamped, 'expected_pose', self.expected_pose_callback, 10)
		# self.pfOdom_subscriber_ = self.create_subscription(Odometry, "/pf/pose/odom", self.pfOdomCallback, 10)
		# self.collison_subscriber_ = self.create_subscription(CrashStatus, "ego_crash", self.collisionCallback, 10)
		# self.drive_subscriber_ = self.create_subscription(AckermannDriveStamped, "/drive", self.driveCallback, 10)
		#self.done_subscriber_ = self.create_subscription(Bool, "/ego_done", self.doneCallback, 10)

		# timers
		# self.save_timer = self.create_timer(0.05, self.savedata)
		
		# obect to save data
		self.flag = 0
		self.ds = saveToFile()
		self.iteration = 0
		self.process_interval = 10  # Only process every 10th message
		self.message_counter_gt = 0
		self.message_counter_odom = 0
		self.message_counter_filter = 0

	def doneCallback(self, msg: Bool):
		self.get_logger().info(str(msg.data))
		if msg.data:
			self.ds.saveFlag = msg.data
			self.get_logger().info("Done")
   
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
			self.get_logger().info('Iteration: '+str(self.iteration))
			# self.get_logger().info(str(self.ds.tempData))
   
			self.ds.saveData = np.vstack((self.ds.saveData, self.ds.tempData))
			self.ds.saveFlag = self.flag
			self.get_logger().info('self.ds.saveFlag: '+str(self.ds.saveFlag))
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

	def OdomCallback(self, msg: Odometry):
     
		self.message_counter_odom += 1
		if self.message_counter_odom % self.process_interval != 0:
			return
		# time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		z = msg.pose.pose.position.z
		yaw = Utils.quaternion_to_angle(msg.pose.pose.orientation)

		self.ds.tempData[6] = x
		self.ds.tempData[7] = y
		self.ds.tempData[8] = z
		self.ds.tempData[9] = yaw
  
	def check_key_press(self):
			# Check if a key is pressed without blocking
			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				input_char = sys.stdin.readline().strip()
				if input_char == "":  # Enter key pressed
					self.flag = True
					self.message_counter_gt = 9
					self.message_counter_odom = 9
					self.get_logger().info("Flag is set. Shutting down...")

	
	def pfOdomCallback(self, msg: Odometry):
		
		time_s = msg.header.stamp.sec 
		time_ns = msg.header.stamp.nanosec
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		z = msg.pose.pose.orientation.z
		w = msg.pose.pose.orientation.w

		self.ds.tempData[0] = time_s
		self.ds.tempData[1] = time_ns
		self.ds.tempData[6] = x
		self.ds.tempData[7] = y
		self.ds.tempData[8] = z
		self.ds.tempData[9] = w

		self.ds.saveData = np.vstack((self.ds.saveData, self.ds.tempData))
		if self.ds.saveFlag:
			self.ds.saveToFile(self.ds.saveData, 'cornerHall')
			self.get_logger().info('saving to file')
			# self.ds.saveFlag = False
			rclpy.shutdown()
			# self.ds.saveData = np.array([0,0,0,0,0,0,0,0,0])

		# dataArray = np.array([time, x, y])
		# self.get_logger().info("True Odom: "+str(dataArray))
		# self.saveToFile(dataArray, 'pfOdom')
  
	def expected_pose_callback(self, msg: PoseStamped):
		'''
		Callback function to process received PoseStamped messages.
		'''
		self.message_counter_filter += 1
		if self.message_counter_filter % self.process_interval != 0:
			return
		# time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		x = msg.pose.position.x
		y = msg.pose.position.y
		z = msg.pose.position.z
		yaw = Utils.quaternion_to_angle(msg.pose.orientation)

		self.ds.tempData[10] = x
		self.ds.tempData[11] = y
		self.ds.tempData[12] = z
		self.ds.tempData[13] = yaw
		self.get_logger().info(f"Received expected pose: position=({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}), "
								f"orientation=({msg.pose.orientation.x}, {msg.pose.orientation.y}, {msg.pose.orientation.z}, {msg.pose.orientation.w})")
		

class saveToFile:
	def __init__(self):
		self.lap = 0
		# 						  sec,nsec,x,y,z,w,x,y,z,w
		self.saveData = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype='f')
		self.tempData = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype='f')
		self.saveFlag = False

	def saveToFile(self,data):
		np.savetxt(fname='/home/ruan/dev_ws/src/landmark_extract/GT_vs_Odom_vs_Est.csv', 
			 X=data, 
			 delimiter=',',
			#  newline=' /n', 
			#  fmt='%.2f',
			 header='seconds, nanoseconds, trueX, trueY, trueZ, trueYaw, odomX, odomY, odomZ, odomYaw, pfX, pfY, pfZ, pfYaw',
			 )
		# path = '/home/chris/sim_ws/src/benchmark_tests/benchmark_tests/Results/Localisation/Accuracy/'+filename+'.csv'
		# # np.savetxt(fname=path, X=data, delimiter=',',newline='/n', fmt='%1.3f')
		# f = open(path, 'a')
		# f.write(str(data[0])+','+str(data[1])+','+str(data[2])+'\n')
		# f.close()
  
	

def main(args=None):
	rclpy.init(args=args)
	node = localisation_test_accuracy()
	# rclpy.spin(node)
	while rclpy.ok() and not node.flag:
			rclpy.spin_once(node) 
			node.check_key_press()  # Check if a key was pressed
	rclpy.spin_once(node)
	rclpy.spin_once(node)
	
	rclpy.shutdown()

if __name__ == '__main__':
	main()