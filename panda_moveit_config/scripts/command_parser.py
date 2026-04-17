#!/usr/bin/env python3
import json
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CommandParserNode(Node):
    def __init__(self):
        super().__init__('command_parser')
        
        self.subscription = self.create_subscription(
            String,
            '/speech_text',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(String, '/sort_command', 10)
        
        self.get_logger().info('Command Parser listening for speech on /speech_text...')

    def listener_callback(self, msg):
        text = msg.data.lower()
        self.get_logger().info(f"Parsing: {text}")
        
        # Simple keyword matching
        obj_id = None
        if "red" in text:
            obj_id = "red_step_block"
        elif "blue" in text:
            obj_id = "blue_cuboid"
        elif "green" in text:
            obj_id = "green_cross_block"
            
        bin_id = None
        if "bin a" in text or "alpha" in text:
            bin_id = "bin_a"
        elif "bin b" in text or "bravo" in text:
            bin_id = "bin_b"
            
        if obj_id and bin_id:
            cmd = {"object": obj_id, "bin": bin_id}
            self.get_logger().info(f"Valid command parsed: {cmd}")
            
            out_msg = String()
            out_msg.data = json.dumps(cmd)
            self.publisher_.publish(out_msg)
        else:
            self.get_logger().warn("Command incomplete or not recognized. Required: [color] and [bin_name]")

def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
