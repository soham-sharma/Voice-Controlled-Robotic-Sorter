#!/usr/bin/env python3
import time
import json
import os
import re
import math
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from rcl_interfaces.msg import Log

try:
	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('Agg')
except ImportError:
	import subprocess
	import sys
	print("Installing matplotlib...")
	subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('Agg')

RESULTS_DIR = os.path.expanduser("~/ws/src/panda_gz_moveit2/benchmark_results")
ALL_RESULTS_FILE = os.path.join(RESULTS_DIR, "all_results.json")

class BenchmarkComprehensiveNode(Node):
	def __init__(self):
		super().__init__('benchmark_comprehensive_node')

		# self.publisher = self.create_publisher(String, '/sort_command', 10)
		self.publisher = self.create_publisher(String, '/speech_text', 10)

		self.objects = ['red_step_block', 'blue_cuboid', 'green_cross_block']
		self.poses = {}
		self.subs = []
		for obj in self.objects:
			sub = self.create_subscription(
				Pose,
				f'/gz/{obj}/pose',
				lambda msg, o=obj: self.pose_callback(o, msg),
				10
			)
			self.subs.append(sub)

		self.rosout_sub = self.create_subscription(Log, '/rosout', self.rosout_callback, 100)

		self.bins = {
			'bin_a': {'center_xy': (0.34, 0.30), 'width': 0.18, 'depth': 0.18},
			'bin_b': {'center_xy': (0.34, -0.30), 'width': 0.18, 'depth': 0.18}
		}

		self.current_task_logs = []

	def pose_callback(self, obj, msg):
		self.poses[obj] = msg

	def in_bin(self, obj, bin_name):
		if obj not in self.poses:
			return False
		pose = self.poses[obj]
		b = self.bins[bin_name]
		cx, cy = b['center_xy']
		hx, hy = b['width'] / 2.0, b['depth'] / 2.0
		return (abs(pose.position.x - cx) < hx and
				abs(pose.position.y - cy) < hy)

	def rosout_callback(self, msg):
		if 'bt_pick_place' in msg.name or 'robot_interface' in msg.name:
			self.current_task_logs.append({
				'time': msg.stamp.sec + msg.stamp.nanosec * 1e-9,
				'level': msg.level,
				'msg': msg.msg
			})
