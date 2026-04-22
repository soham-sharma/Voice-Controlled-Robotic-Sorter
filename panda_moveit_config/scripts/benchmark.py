#!/usr/bin/env python3
import time
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose

class BenchmarkNode(Node):
    def __init__(self):
        super().__init__('benchmark_node')
        
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
            
        self.bins = {
            'bin_a': {'center_xy': (0.34, 0.30), 'width': 0.18, 'depth': 0.18},
            'bin_b': {'center_xy': (0.34, -0.30), 'width': 0.18, 'depth': 0.18}
        }
        
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

    def run_benchmark(self):
        tasks = [
            ("red_step_block", "bin_a", "move the red block to bin alpha"),
            ("blue_cuboid", "bin_b", "move the blue cuboid to bin bravo"),
            ("green_cross_block", "bin_b", "move the green cross block to bin bravo"),
        ]
        
        results = []
        
        # Wait for initial poses
        self.get_logger().info("Waiting for initial object poses...")
        time.sleep(2.0)
        rclpy.spin_once(self, timeout_sec=2.0)
        
        for obj, bin_name, command in tasks:
            self.get_logger().info(f"--- Starting task: {command} ---")
            msg = String()
            msg.data = command
            self.publisher.publish(msg)
            
            start_time = time.time()
            success = False
            timeout = 120.0 # 2 minutes max per object
            
            while time.time() - start_time < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.in_bin(obj, bin_name):
                    # Ensure it stays stable for 2 seconds
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
            if success:
                self.get_logger().info(f"[SUCCESS] {obj} sorted to {bin_name} in {elapsed:.2f}s")
            else:
                self.get_logger().info(f"[FAILURE] {obj} not sorted to {bin_name} within timeout")
                
            results.append({
                "object": obj,
                "bin": bin_name,
                "command": command,
                "success": success,
                "time": elapsed
            })
            
            # Wait a bit before next command
            time.sleep(3.0)
            
        self.get_logger().info("=== Benchmark Results ===")
        for r in results:
            self.get_logger().info(json.dumps(r))

def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    try:
        node.run_benchmark()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
