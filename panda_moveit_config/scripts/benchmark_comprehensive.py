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

	def analyze_logs(self, start_time):
		metrics = {
			"grasp_time": None,
			"place_time": None,
			"retries": 0,
			"total_time": None,
			"grasp_failures": 0
		}

		t_start = start_time
		t_grasp = None
		t_place = None

		for log in self.current_task_logs:
			if "Goal received" in log['msg'] and t_start == start_time:
				t_start = log['time']
			if "CheckObjectIsAttached: gap" in log['msg'] and "[OK]" in log['msg']:
				t_grasp = log['time']
			if "DetachObject:" in log['msg']:
				t_place = log['time']
			if "requeueing command to retry" in log['msg']:
				metrics["retries"] += 1
			if "no valid grasp" in log['msg'] or "no valid grasps" in log['msg']:
				metrics["grasp_failures"] += 1

		if t_grasp and t_start:
			metrics["grasp_time"] = t_grasp - t_start
		if t_place and t_grasp:
			metrics["place_time"] = t_place - t_grasp
		if t_place and t_start:
			metrics["total_time"] = t_place - t_start

		return metrics

	def run_benchmark(self):
		tasks = [
			("red_step_block", "bin_a", "move the red block to bin alpha"),
			("blue_cuboid", "bin_b", "move the blue cuboid to bin bravo"),
			("green_cross_block", "bin_b", "move the green cross block to bin bravo"),
		]

		run_results = []

		self.get_logger().info("Waiting for initial object poses...")
		time.sleep(2.0)
		rclpy.spin_once(self, timeout_sec=2.0)

		for obj, bin_name, command in tasks:
			self.get_logger().info(f"--- Starting task: {command} ---")
			self.current_task_logs = []

			msg = String()
			msg.data = command
			self.publisher.publish(msg)

			start_time = time.time()
			success = False
			timeout = 120.0

			while time.time() - start_time < timeout:
				rclpy.spin_once(self, timeout_sec=0.1)
				if self.in_bin(obj, bin_name):
					stable = True
					for _ in range(20):
						rclpy.spin_once(self, timeout_sec=0.1)
						if not self.in_bin(obj, bin_name):
							stable = False
							break

					if stable:
						success = True
						break

			elapsed = time.time() - start_time

			# Spin a bit more to catch trailing logs
			t_end = time.time()
			while time.time() - t_end < 2.0:
				rclpy.spin_once(self, timeout_sec=0.1)

			metrics = self.analyze_logs(start_time)

			if success:
				self.get_logger().info(f"[SUCCESS] {obj} sorted to {bin_name} in {elapsed:.2f}s")
			else:
				self.get_logger().info(f"[FAILURE] {obj} not sorted to {bin_name} within timeout")

			task_result = {
				"object": obj,
				"bin": bin_name,
				"command": command,
				"success": success,
				"overall_time": elapsed,
				"retries": metrics["retries"]
			}
			run_results.append(task_result)

			time.sleep(3.0)

		self.save_and_generate_reports(run_results)
