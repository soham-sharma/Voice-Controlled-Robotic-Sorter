cmake_minimum_required(VERSION 3.5)
project(panda_moveit_config)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(panda_description REQUIRED)

# Install directories
install(DIRECTORY config launch rviz srdf DESTINATION share/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/action_node.py
  scripts/robot_interface.py
  scripts/gpd.py
  scripts/bt_pick_place.py
  scripts/command_parser.py
  DESTINATION lib/${PROJECT_NAME}
)

# Setup the project
ament_package()
