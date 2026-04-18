#!/usr/bin/env python3
import json
import subprocess
import threading
from vosk import Model, KaldiRecognizer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SpeechNode(Node):
    def __init__(self):
        super().__init__('speech_node')
        self.publisher_ = self.create_publisher(String, '/speech_text', 10)
        
        self.get_logger().info('Initializing Vosk Model (will use cached en-us model)...')
        self.model = Model(lang="en-us")
        
        # Vosk prefers 16kHz
        self.samplerate = 16000 
        grammar = json.dumps([
            "move", "put", "place", "take",
            "the", "to", "in", "into",
            "red", "blue", "green",
            "block", "cuboid", "cross",
            "bin", "alpha", "bravo", "a", "b",
            "[unk]"
        ])
        self.rec = KaldiRecognizer(self.model, self.samplerate, grammar)
        
        self.get_logger().info('Starting parecord capture thread...')
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.audio_thread.start()

    def _capture_audio(self):
        cmd = [
            "parecord", 
            "--rate=16000", 
            "--channels=1", 
            "--format=s16le", 
            "--raw",
            "-d", "@DEFAULT_SOURCE@"
        ]
        
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.get_logger().info('Vosk audio capture active via parecord.')
            
            while self.is_running and rclpy.ok():
                data = self.process.stdout.read(4000)
                if len(data) == 0:
                    break
                    
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    text = res.get("text", "")
                    if text:
                        self.get_logger().info(f"Recognized: {text}")
                        msg = String()
                        msg.data = text
                        self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Audio capture failed: {e}")

    def destroy_node(self):
        self.is_running = False
        if hasattr(self, 'process'):
            self.process.kill()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SpeechNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
