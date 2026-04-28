# Voice-Controlled Robotic Sorter

A ROS 2 (Jazzy) simulation environment for the Franka Emika FR3 robot arm, integrating Gazebo for physics simulation, MoveIt 2 for motion planning, and Vosk for offline voice control. The robot autonomously sorts complex geometries into bins using a Behavior Tree and GPD (Grasp Pose Detection) based on spoken natural language commands.

## Building and Running

Build the Docker image:

```bash
.docker/build.bash
```

Start the container:

```bash
.docker/run.bash
```

Build the workspace (inside container):
```bash
build
source install/setup.bash
```

### Execution Terminals

Launch the simulation (Gazebo + MoveIt 2 + RViz):
```bash
launch_ctrl
```

Run the Behavior Tree Pick & Place Node:
```bash
launch_bt
```

Run the Command Parser Node:
```bash
ros2 run panda_moveit_config command_parser.py
```

Run the Speech Node:
```bash
ros2 run panda_moveit_config speech_node.py
```

*(Optional)* Run the Comprehensive Benchmark Utility:
```bash
ros2 run panda_moveit_config benchmark_comprehensive.py
```