#!/usr/bin/env python3
"""
action_node.py — ActionNode base class for single-ROS-action behaviour tree nodes.

DO NOT MODIFY THIS FILE.
"""

import py_trees
from moveit_msgs.msg import MoveItErrorCodes
from robot_interface import RobotInterface


class ActionNode(py_trees.behaviour.Behaviour):
    """
    Drives a single ROS action to completion.

    Subclasses implement _send_goal() which fires the async request.
    update() polls goal acceptance then the result, returning RUNNING until
    one of them resolves, then SUCCESS or FAILURE.
    """

    _TIMEOUT_SEC = 30.0

    def __init__(self, name: str, robot: RobotInterface):
        super().__init__(name)
        self.robot = robot
        self._goal_future = None
        self._result_future = None
        self._deadline = None

    def _send_goal(self):
        raise NotImplementedError

    def _result_ok(self, result) -> bool:
        return result.result.error_code.val == MoveItErrorCodes.SUCCESS

    def initialise(self):
        self._goal_future = self._send_goal()
        self._result_future = None
        self._deadline = self.robot.get_clock().now().nanoseconds + int(self._TIMEOUT_SEC * 1e9)

    def update(self):
        if self.robot.get_clock().now().nanoseconds > self._deadline:
            self.robot.log(f'[FAIL] {self.name}: timeout')
            return py_trees.common.Status.FAILURE

        if self._result_future is None:
            if not self._goal_future.done():
                self.robot.log(f'[DBG]  {self.name}: waiting for goal acceptance')
                return py_trees.common.Status.RUNNING
            handle = self._goal_future.result()
            if not handle.accepted:
                self.robot.log(f'[FAIL] {self.name}: goal rejected')
                return py_trees.common.Status.FAILURE
            self.robot.log(f'[DBG]  {self.name}: goal accepted, waiting for result')
            self._result_future = handle.get_result_async()
            return py_trees.common.Status.RUNNING

        if not self._result_future.done():
            self.robot.log(f'[DBG]  {self.name}: waiting for result')
            return py_trees.common.Status.RUNNING

        result = self._result_future.result()
        if self._result_ok(result):
            self.robot.log(f'[OK]   {self.name}')
            return py_trees.common.Status.SUCCESS
        self.robot.log(
            f'[FAIL] {self.name}: error_code={result.result.error_code.val}'
        )
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        if self._result_future is None and self._goal_future is not None:
            if self._goal_future.done():
                handle = self._goal_future.result()
                if handle.accepted:
                    handle.cancel_goal_async()
        self._goal_future = None
        self._result_future = None


class GripperNode(ActionNode):
    """ActionNode variant for gripper actions (different success code)."""
    _TIMEOUT_SEC = 5.0

    def _result_ok(self, result) -> bool:
        code = result.result.error_code
        self.robot.get_logger().info(f'{self.name}: gripper error_code={code}')
        return code in (0, -5)  # 0=OK, -5=GOAL_TOLERANCE_VIOLATED (gripper blocked by object)
