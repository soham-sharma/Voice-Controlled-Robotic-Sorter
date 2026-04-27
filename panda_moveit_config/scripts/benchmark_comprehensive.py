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
