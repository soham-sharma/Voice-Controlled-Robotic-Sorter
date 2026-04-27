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

	def save_and_generate_reports(self, run_results):
		os.makedirs(RESULTS_DIR, exist_ok=True)

		# Save individual run
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		run_file = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")
		with open(run_file, 'w') as f:
			json.dump(run_results, f, indent=4)

		# Append to all results
		all_results = []
		if os.path.exists(ALL_RESULTS_FILE):
			with open(ALL_RESULTS_FILE, 'r') as f:
				try:
					all_results = json.load(f)
				except json.JSONDecodeError:
					pass

		# Add a run identifier
		run_id = len(set(r.get('run_id', 0) for r in all_results)) + 1
		for r in run_results:
			r['run_id'] = run_id
			all_results.append(r)

		with open(ALL_RESULTS_FILE, 'w') as f:
			json.dump(all_results, f, indent=4)

		self.generate_latex_table(all_results)
		self.generate_plots(all_results)

		self.get_logger().info(f"=== Saved run results to {run_file} ===")
		self.get_logger().info(f"=== Regenerated reports in {RESULTS_DIR} ===")

	def generate_latex_table(self, all_results):
		# Group by object
		stats = {}
		for r in all_results:
			obj = r['object']
			if obj not in stats:
				stats[obj] = {'trials': 0, 'successes': 0, 'times': [], 'retries': 0}
			stats[obj]['trials'] += 1
			if r['success']:
				stats[obj]['successes'] += 1
				if r['overall_time'] is not None:
					stats[obj]['times'].append(r['overall_time'])
			stats[obj]['retries'] += r['retries']

		latex = [
			"\\begin{table}[h]",
			"\\centering",
			"\\begin{tabular}{|l|c|c|c|c|}",
			"\\hline",
			"\\textbf{Object} & \\textbf{Success Rate} & \\textbf{Avg Time (s)} & \\textbf{Total Retries} & \\textbf{Trials} \\\\",
			"\\hline"
		]

		for obj, s in stats.items():
			rate = (s['successes'] / s['trials']) * 100 if s['trials'] > 0 else 0
			avg_time = sum(s['times']) / len(s['times']) if s['times'] else 0.0
			latex.append(f"{obj.replace('_', '\\_')} & {rate:.1f}\\% & {avg_time:.2f} & {s['retries']} & {s['trials']} \\\\")

		latex.extend([
			"\\hline",
			"\\end{tabular}",
			"\\caption{Robotic Sorting Benchmark Results over Multiple Runs}",
			"\\label{tab:benchmark_results}",
			"\\end{table}"
		])

		with open(os.path.join(RESULTS_DIR, "benchmark_table.tex"), 'w') as f:
			f.write("\n".join(latex))

	def generate_plots(self, all_results):
		objects = list(set(r['object'] for r in all_results))

		# 2. Task Consistency (Box and Whisker Plot)
		fig, ax = plt.subplots(figsize=(8, 6))
		box_data = []
		for obj in objects:
			times = [r['overall_time'] for r in all_results if r['object'] == obj and r['success'] and r['overall_time'] is not None]
			box_data.append(times if times else [0])

		ax.boxplot(box_data, labels=[o.replace('_', ' ').title() for o in objects])
		ax.set_ylabel('Total Execution Time (s)')
		ax.set_title('Task Execution Time Consistency')
		plt.tight_layout()
		plt.savefig(os.path.join(RESULTS_DIR, "benchmark_consistency.png"), dpi=300)
		plt.close()

		# 3. Robustness Breakdown (Stacked Success/Retry/Fail Bar)
		fig, ax = plt.subplots(figsize=(8, 6))
		first_try = []
		retry_success = []
		fail = []

		for obj in objects:
			ft = len([r for r in all_results if r['object'] == obj and r['success'] and r['retries'] == 0])
			rs = len([r for r in all_results if r['object'] == obj and r['success'] and r['retries'] > 0])
			f = len([r for r in all_results if r['object'] == obj and not r['success']])
			first_try.append(ft)
			retry_success.append(rs)
			fail.append(f)

		bar_width = 0.5
		ax.bar(objects, first_try, bar_width, label='First-Try Success', color='mediumseagreen')
		ax.bar(objects, retry_success, bar_width, bottom=first_try, label='Success After Retry', color='goldenrod')
		bottom_fail = [first_try[i] + retry_success[i] for i in range(len(objects))]
		ax.bar(objects, fail, bar_width, bottom=bottom_fail, label='Failed', color='tomato')

		ax.set_ylabel('Count')
		ax.set_title('Robustness Breakdown by Object')
		ax.legend()
		ax.set_xticklabels([o.replace('_', ' ').title() for o in objects], rotation=15)

		plt.tight_layout()
		plt.savefig(os.path.join(RESULTS_DIR, "benchmark_robustness.png"), dpi=300)
		plt.close()

def main(args=None):
	rclpy.init(args=args)
	node = BenchmarkComprehensiveNode()
	try:
		node.run_benchmark()
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()
