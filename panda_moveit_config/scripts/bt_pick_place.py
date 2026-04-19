#!/usr/bin/env python3
"""
LAB03 — Behavior Tree pick-and-place.

Run after launch_ctrl:
    ros2 run panda_moveit_config bt_pick_place.py
"""

import copy
import rclpy
import py_trees
import py_trees_ros
import numpy as np  # noqa: F401
import tf_transformations

from geometry_msgs.msg import Pose  # noqa: F401
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
import json

from robot_interface import RobotInterface, BINS, GZ_OBJECTS, DetectedObject
from action_node import ActionNode, GripperNode
from gpd import sample_cuboid_surface, detect_grasps, gpd_to_panda_pose, GraspCandidate  # noqa: F401


# ─── Provided nodes ───────────────────────────────────────────────────────────

class ReadScene(py_trees.behaviour.Behaviour):
    """
    Entry point for every run: clean up stale planning-scene state, then
    wait for fresh collision-object data for ALL known objects.

    initialise(): detach all known objects from the previous run,
                  clear the object cache so stale latched messages are ignored.
    update():     poll detected_objects; once all objects in GZ_OBJECTS are
                  detected and stable, add them to the planning scene and
                  return SUCCESS.

    Blackboard writes: /detected_objects (dict[str, DetectedObject])
    """

    ALL_OBJECTS = ['red_step_block', 'blue_cuboid', 'green_cross_block']

    def __init__(self, robot: RobotInterface):
        super().__init__('ReadScene')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='ReadScene')
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.WRITE)

    _STABLE_TICKS = 10  # ~2 s at 200 ms/tick

    def initialise(self) :
        for obj_id in self.ALL_OBJECTS:
            self.robot.detach_object(obj_id)
        self.robot.log('[INFO] ReadScene: detached stale objects')
        self._stable = {}   # obj_id -> consecutive stable ticks
        self._last_z = {}   # obj_id -> last seen z

    def update(self):
        all_stable = True
        for obj_id in self.ALL_OBJECTS:
            obj = self.robot._detected_objects.get(obj_id)
            if obj is None:
                all_stable = False
                continue
            z = obj.pose.position.z
            last = self._last_z.get(obj_id)
            if last is None or abs(z - last) > 0.001:
                self._stable[obj_id] = 0
            else:
                self._stable[obj_id] = self._stable.get(obj_id, 0) + 1
            self._last_z[obj_id] = z
            if self._stable.get(obj_id, 0) < self._STABLE_TICKS:
                all_stable = False

        if not all_stable:
            return py_trees.common.Status.RUNNING

        # All objects stable — add to planning scene
        scene = PlanningScene(is_diff=True)
        for obj_id in self.ALL_OBJECTS:
            obj = self.robot._detected_objects[obj_id]
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.header.stamp = self.robot.get_clock().now().to_msg()
            co.id = obj_id
            co.operation = CollisionObject.ADD
            # Build rotation matrix from object orientation
            q = (obj.pose.orientation.x, obj.pose.orientation.y,
                 obj.pose.orientation.z, obj.pose.orientation.w)
            R = tf_transformations.quaternion_matrix(q)[:3, :3]
            for part in GZ_OBJECTS[obj_id]:
                co.primitives.append(SolidPrimitive(
                    type=SolidPrimitive.BOX, dimensions=part['dims']))
                # Compute world-frame pose for this sub-box
                off = np.array(part['offset'])
                world_off = R @ off
                part_pose = copy.deepcopy(obj.pose)
                part_pose.position.x += float(world_off[0])
                part_pose.position.y += float(world_off[1])
                part_pose.position.z += float(world_off[2])
                co.primitive_poses.append(part_pose)
            scene.world.collision_objects.append(co)
            self.robot.log(f'[INFO] ReadScene: {obj_id} at '
                            f'({obj.pose.position.x:.3f}, {obj.pose.position.y:.3f}, {obj.pose.position.z:.3f})')
        self.robot._scene_pub.publish(scene)
        # Re-publish table + bin walls that MoveToHome cleared for escape planning.
        self.robot._publish_table()
        self.bb.detected_objects = dict(self.robot._detected_objects)
        return py_trees.common.Status.SUCCESS


class MoveToHome(ActionNode):
    """Reset to a known-good state, then move home.

    On every entry (including restarts after failure):
      1. Open gripper (fire-and-forget)
      2. Gz-detach + planning-scene remove all objects
      3. Move arm to INITIAL_JOINTS

    Always succeeds at resetting state regardless of prior situation.
    Retries the motion up to _MAX_RETRIES times on transient planner failures.
    """

    _MAX_RETRIES = 3

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToHome', robot)
        self._retries = 0

    def _reset_state(self):
        # Open gripper (best-effort, don't wait for result)
        self.robot.send_gripper_goal(open=True)
        # Detach and remove all objects
        for obj_id in ReadScene.ALL_OBJECTS:
            self.robot.detach_object(obj_id)
        # Clear planning scene: remove grasped objects AND bin walls so the
        # arm can plan freely out of the bin.  Bins are static and will be
        # re-published the next time _publish_table() is called (on ReadScene).
        scene = PlanningScene(is_diff=True)
        ids_to_remove = list(ReadScene.ALL_OBJECTS) + list(BINS.keys())
        for obj_id in ids_to_remove:
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.header.stamp = self.robot.get_clock().now().to_msg()
            co.id = obj_id
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
        self.robot._scene_pub.publish(scene)

    def initialise(self):
        self._retries = 0
        self._reset_state()
        super().initialise()

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE and self._retries < self._MAX_RETRIES:
            self._retries += 1
            self.robot.log(f'[WARN] MoveToHome: retrying ({self._retries}/{self._MAX_RETRIES})')
            self._reset_state()
            super().initialise()
            return py_trees.common.Status.RUNNING
        return status

    def _send_goal(self):
        return self.robot.send_joints_goal(RobotInterface.INITIAL_JOINTS)



# ─── Pick sub-nodes ───────────────────────────────────────────────────────────


class OpenGripper(GripperNode):
    def __init__(self, robot: RobotInterface):
        super().__init__('OpenGripper', robot)

    def _send_goal(self):
        return self.robot.send_gripper_goal(open=True)


class CloseGripper(GripperNode):
    def __init__(self, robot: RobotInterface):
        super().__init__('CloseGripper', robot)

    def _send_goal(self):
        return self.robot.send_gripper_goal(open=False)


class MoveToPreGrasp(ActionNode):
    """Move to 15 cm above the top-ranked grasp proposal.

    Blackboard reads: /grasp_proposals (list[Pose])
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToPreGrasp', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToPreGrasp')
        self.bb.register_key('/grasp_proposals', access=py_trees.common.Access.READ)

    def _send_goal(self):
        pre = copy.deepcopy(self.bb.grasp_proposals[0])
        pre.position.z += 0.08
        self.robot.publish_pose_axes(pre, 'pre_grasp', scale=0.1)
        return self.robot.send_pose_goal(pre)


class MoveToGrasp(ActionNode):
    """Descend to the top-ranked grasp proposal.

    Blackboard reads: /grasp_proposals (list[Pose]), /target_object_id (str)
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToGrasp', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToGrasp')
        self.bb.register_key('/grasp_proposals',  access=py_trees.common.Access.READ)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def _send_goal(self):
        grasp = self.bb.grasp_proposals[0]
        self.robot.publish_pose_axes(grasp, 'grasp')
        return self.robot.send_pose_goal(
            grasp,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
        )


class CheckObjectIsAttached(py_trees.behaviour.Behaviour):
    """
    Verify the gripper is not fully closed — if fingers stopped before
    GRIPPER_CLOSED, something is between them.

    Returns SUCCESS if gap > threshold, FAILURE if fingers closed fully
    (missed the object).
    """

    # If total finger gap is above this the gripper caught something
    _CONTACT_THRESHOLD = sum(RobotInterface.GRIPPER_CLOSED) + 0.005

    def __init__(self, robot: RobotInterface):
        super().__init__('CheckObjectIsAttached')
        self.robot = robot

    def update(self):
        gap = sum(self.robot.gripper_pos)
        if gap > self._CONTACT_THRESHOLD:
            self.robot.log(f'[OK]   CheckObjectIsAttached: gap={gap:.4f}')
            return py_trees.common.Status.SUCCESS
        self.robot.log(f'[FAIL] CheckObjectIsAttached: gap={gap:.4f} (fully closed)')
        return py_trees.common.Status.FAILURE


class DetachObject(py_trees.behaviour.Behaviour):
    """Detach the target object from the planning scene. Always returns SUCCESS.

    Blackboard reads: /target_object_id (str)
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('DetachObject')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='DetachObject')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def update(self):
        self.robot.detach_object(self.bb.target_object_id)
        self.robot.log(f'[INFO] DetachObject: {self.bb.target_object_id}')
        return py_trees.common.Status.SUCCESS


class AttachObject(py_trees.behaviour.Behaviour):
    """Attach the target object in the planning scene.

    Blackboard reads: /target_object_id (str)
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('AttachObject')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='AttachObject')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def update(self):
        self.robot.attach_object(self.bb.target_object_id)
        return py_trees.common.Status.SUCCESS


class Retreat(ActionNode):
    """Move back to 15 cm above the grasp pose after closing.

    Blackboard reads: /grasp_proposals (list[Pose])
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('Retreat', robot)
        self.bb = py_trees.blackboard.Client(name='Retreat')
        self.bb.register_key('/grasp_proposals', access=py_trees.common.Access.READ)

    def initialise(self):
        co = CollisionObject()
        co.header.frame_id = 'world'
        co.header.stamp = self.robot.get_clock().now().to_msg()
        co.id = 'table'
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        self.robot._scene_pub.publish(scene)
        super().initialise()

    def terminate(self, new_status):
        self.robot._publish_table()
        super().terminate(new_status)

    def _send_goal(self):
        retreat = copy.deepcopy(self.bb.grasp_proposals[0])
        retreat.position.z += 0.15
        return self.robot.send_pose_goal(
            retreat,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
        )


# ─── Drop Nodes ──────────────────────────────────────────────────────────

class MoveToDrop(ActionNode):
    """Move to the drop pose written by ProposeDropPose.

    Blackboard reads: /drop_pose        (Pose)
                      /target_object_id (str)  — for ACM during approach
    """

    _TIMEOUT_SEC = 60.0

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToDrop', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToDrop')
        self.bb.register_key('/drop_pose',        access=py_trees.common.Access.READ)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def _send_goal(self):
        drop = self.bb.drop_pose
        self.robot.publish_pose_axes(drop, 'drop_pose', scale=0.10)
        return self.robot.send_pose_goal(
            drop,
            acm_object_id=self.bb.target_object_id,
            position_tolerance=0.03,
            orientation_tolerance=0.15,
        )


class CheckObjectSorted(py_trees.behaviour.Behaviour):
    """Poll until the target object lands in the target bin.

    RUNNING : current object not yet stable in bin
    FAILURE : timeout
    SUCCESS : object confirmed in bin

    Blackboard reads: /target_object_id (str)
                      /target_bin_id (str)
    """

    _TIMEOUT_SEC = 5.0
    _STABLE_TICKS = 5

    def __init__(self, robot: RobotInterface):
        super().__init__('CheckObjectSorted')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='CheckObjectSorted')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self.bb.register_key('/target_bin_id',    access=py_trees.common.Access.READ)

    def initialise(self):
        self._deadline = (self.robot.get_clock().now().nanoseconds
                          + int(self._TIMEOUT_SEC * 1e9))
        self._stable = 0

    def _in_container_xy(self, pose) -> bool:
        if pose is None:
            return False
        container = BINS[self.bb.target_bin_id]
        cx, cy = container['center_xy']
        hx, hy = container['width'] / 2.0, container['depth'] / 2.0
        return (abs(pose.position.x - cx) < hx and
                abs(pose.position.y - cy) < hy)

    def update(self):
        if self.robot.get_clock().now().nanoseconds > self._deadline:
            self.robot.log('[FAIL] CheckObjectSorted: timeout', speak=True)
            return py_trees.common.Status.FAILURE

        obj = self.robot._detected_objects.get(self.bb.target_object_id)
        if obj is None:
            return py_trees.common.Status.RUNNING

        if self._in_container_xy(obj.pose):
            self._stable += 1
        else:
            self._stable = 0

        if self._stable < self._STABLE_TICKS:
            return py_trees.common.Status.RUNNING

        self.robot.log(f'[OK]   Goal Status: SUCCESS — {self.bb.target_object_id} sorted', speak=True)
        # Return FAILURE to reset the Sequence(memory=True) so it can handle the next command
        return py_trees.common.Status.FAILURE


class RepeatAlways(py_trees.decorators.Decorator):
    """Repeat forever: restart the child on both SUCCESS and FAILURE."""
    def update(self):
        if self.decorated.status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.RUNNING
        return self.decorated.status


# ─── Lab03 TODOs ─────────────────────────────────────────────────────────────

def build_tree(robot: RobotInterface) -> py_trees.behaviour.Behaviour:  # noqa: ARG001
    reset = py_trees.composites.Sequence(name='Reset', memory=True)
    reset.add_children([
        MoveToHome(robot),
        ReadScene(robot),
        WaitForCommand(robot),
    ])

    grasp = py_trees.composites.Sequence(name='Grasp', memory=True)
    grasp.add_children([
        ProposeGrasps(robot),
        OpenGripper(robot),
        MoveToPreGrasp(robot),
        MoveToGrasp(robot),
        CloseGripper(robot),
        CheckObjectIsAttached(robot),
        AttachObject(robot),
        Retreat(robot),
    ])  

    place = py_trees.composites.Sequence(name='Place', memory=True)
    place.add_children([
        ProposeDropPose(robot),
        MoveToDrop(robot),
        OpenGripper(robot),
        DetachObject(robot),
        CheckObjectSorted(robot),
    ])

    # Wrap grasp in RequeueOnFailure so a missed/toppled pick automatically
    # re-queues the command and the robot retries from the Reset phase.
    resilient_grasp = RequeueOnFailure(grasp, robot)

    cycle = py_trees.composites.Sequence(name='PickPlaceCycle', memory=True)
    cycle.add_children([reset, resilient_grasp, place])

    return RepeatAlways(name='RepeatAlways', child=cycle)


class RequeueOnFailure(py_trees.decorators.Decorator):
    """If the decorated subtree fails, push the current command back onto
    the queue so the Reset→WaitForCommand phase will pick it up again
    and the robot retries the same goal (e.g. after an object topples)."""

    def __init__(self, child, robot: RobotInterface):
        super().__init__(name='RequeueOnFailure', child=child)
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='RequeueOnFailure')
        self.bb.register_key('/command_queue',    access=py_trees.common.Access.WRITE)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self.bb.register_key('/target_bin_id',    access=py_trees.common.Access.READ)

    def update(self):
        new_status = self.decorated.status
        if new_status == py_trees.common.Status.FAILURE:
            obj_id  = self.bb.target_object_id
            bin_id  = self.bb.target_bin_id
            if obj_id and bin_id:
                self.robot.log(
                    f'[WARN] Grasp failed for {obj_id} — requeueing command to retry'
                )
                self.bb.command_queue = [{'object': obj_id, 'bin': bin_id}]
            return py_trees.common.Status.FAILURE
        return new_status


def init_blackboard():
    bb = py_trees.blackboard.Client(name='init')
    bb.register_key('/detected_objects',   access=py_trees.common.Access.WRITE)
    bb.register_key('/target_object_id',   access=py_trees.common.Access.WRITE)
    bb.register_key('/target_bin_id',      access=py_trees.common.Access.WRITE)
    bb.register_key('/grasp_proposals',    access=py_trees.common.Access.WRITE)
    bb.register_key('/drop_pose',          access=py_trees.common.Access.WRITE)
    bb.register_key('/command_queue',      access=py_trees.common.Access.WRITE)
    bb.detected_objects = dict()
    bb.target_object_id = ''
    bb.target_bin_id = ''
    bb.grasp_proposals = list()
    bb.drop_pose = None
    bb.command_queue = []

class WaitForCommand(py_trees.behaviour.Behaviour):
    """
    Wait for a valid voice command from /command_queue.
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('WaitForCommand')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='WaitForCommand')
        self.bb.register_key('/command_queue', access=py_trees.common.Access.WRITE)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.WRITE)
        self.bb.register_key('/target_bin_id', access=py_trees.common.Access.WRITE)
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)

    def update(self):
        if len(self.bb.command_queue) > 0:
            cmd = self.bb.command_queue.pop(0)
            target_id = cmd['object']
            target_bin = cmd['bin']

            # Check if object is currently on the table
            if target_id not in self.bb.detected_objects:
                self.robot.log(f'[WARN] Object {target_id} not visible', speak=True)
                return py_trees.common.Status.RUNNING

            self.bb.target_object_id = target_id
            self.bb.target_bin_id = target_bin
            self.robot.log(f'[INFO] Goal received: {target_id} to {target_bin}', speak=True)
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class ProposeGrasps(py_trees.behaviour.Behaviour):
    """
    Sample the target object surface, run GPD, filter candidates, and
    write valid grasp poses to /grasp_proposals.

    Retries up to _MAX_RETRIES times if no valid candidates are found,
    then returns FAILURE.
    """

    _MAX_RETRIES = 3

    def __init__(self, robot: RobotInterface):
        super().__init__('ProposeGrasps')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='ProposeGrasps')
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)
        self.bb.register_key('/container', access=py_trees.common.Access.READ)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self.bb.register_key('/grasp_proposals', access=py_trees.common.Access.WRITE)

    _N_SURFACE_POINTS = 2500
    _MAX_OUTPUT_PROPOSALS = 8
    _MIN_APPROACH_DOWN_Z = 0.20
    _MIN_GRASP_Z_CLEARANCE = 0.01
    _MIN_TCP_Z_CLEARANCE = 0.015
    _MIN_REACH_M = 0.20
    _MAX_REACH_M = 0.90
    _MIN_X_M = 0.15
    _MAX_ABS_Y_M = 0.75

    def initialise(self):
        self._retries = 0
        self.bb.grasp_proposals = []

    def _pose_is_reachable(self, pose: Pose) -> bool:
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        r = float(np.hypot(x, y))
        table_z = self.bb.container['table_z']
        return (
            self._MIN_REACH_M <= r <= self._MAX_REACH_M and
            x >= self._MIN_X_M and
            abs(y) <= self._MAX_ABS_Y_M and
            table_z + self._MIN_TCP_Z_CLEARANCE <= z <= table_z + 0.45
        )

    def update(self):
        obj_id = self.bb.target_object_id
        obj = self.bb.detected_objects.get(obj_id)
        if obj is None:
            self.robot.log(f'[FAIL] ProposeGrasps: unknown target "{obj_id}"')
            return py_trees.common.Status.FAILURE

        center = (
            obj.pose.position.x,
            obj.pose.position.y,
            obj.pose.position.z,
        )
        orientation = (
            obj.pose.orientation.x,
            obj.pose.orientation.y,
            obj.pose.orientation.z,
            obj.pose.orientation.w,
        )
        cloud = sample_cuboid_surface(
            center=center,
            dims=obj.dims,
            n_points=self._N_SURFACE_POINTS,
            orientation=orientation,
        )
        self.robot.publish_cloud(cloud)

        candidates = detect_grasps(cloud)
        table_z = self.bb.container['table_z']
        valid: list[Pose] = []
        for cand in candidates:
            approach = cand.R[:, 0]
            # Table-mounted Panda cannot physically approach from underneath.
            if approach[2] > -self._MIN_APPROACH_DOWN_Z:
                continue
            if cand.pos[2] < table_z + self._MIN_GRASP_Z_CLEARANCE:
                continue

            pose = gpd_to_panda_pose(cand.pos, cand.R)
            if not self._pose_is_reachable(pose):
                continue

            valid.append(pose)
            if len(valid) >= self._MAX_OUTPUT_PROPOSALS:
                break

        if valid:
            self.bb.grasp_proposals = valid
            for i, pose in enumerate(valid[:3]):
                self.robot.publish_pose_axes(pose, f'grasp_{i}', scale=0.08)
            self.robot.log(
                f'[OK]   ProposeGrasps: {len(valid)} valid of {len(candidates)} candidates'
            )
            return py_trees.common.Status.SUCCESS

        self._retries += 1
        if self._retries < self._MAX_RETRIES:
            self.robot.log(
                f'[WARN] ProposeGrasps: no valid grasp ({self._retries}/{self._MAX_RETRIES}), retrying'
            )
            return py_trees.common.Status.RUNNING

        self.bb.grasp_proposals = []
        self.robot.log('[FAIL] ProposeGrasps: no valid grasps after retries')
        return py_trees.common.Status.FAILURE

class ProposeDropPose(py_trees.behaviour.Behaviour):
    """
    Compute a drop pose above the container and write it to /drop_pose.

    The pose must be within the container footprint with sufficient
    clearance above the walls.
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('ProposeDropPose')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='ProposeDropPose')
        self.bb.register_key('/container', access=py_trees.common.Access.READ)
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self.bb.register_key('/drop_pose', access=py_trees.common.Access.WRITE)

    _WALL_MARGIN_M = 0.02
    _DROP_CLEARANCE_ABOVE_WALL_M = 0.10
    _MIN_USABLE_HALF_SPAN_M = 0.03
    _DROP_PATTERN = [
        (-0.55, -0.55),
        (0.55, -0.55),
        (0.00, 0.55),
    ]

    def update(self):
        container = self.bb.container
        target_id = self.bb.target_object_id
        obj = self.bb.detected_objects.get(target_id)

        if obj is None:
            self.robot.log(f'[FAIL] ProposeDropPose: unknown target "{target_id}"')
            return py_trees.common.Status.FAILURE

        cx, cy = container['center_xy']
        half_w = container['width'] / 2.0
        half_d = container['depth'] / 2.0

        # Keep released object footprint away from the walls.
        margin_x = self._WALL_MARGIN_M + 0.5 * obj.dims[0]
        margin_y = self._WALL_MARGIN_M + 0.5 * obj.dims[1]
        usable_half_w = half_w - margin_x
        usable_half_d = half_d - margin_y
        if (usable_half_w < self._MIN_USABLE_HALF_SPAN_M or
                usable_half_d < self._MIN_USABLE_HALF_SPAN_M):
            self.robot.log('[FAIL] ProposeDropPose: container usable area too small')
            return py_trees.common.Status.FAILURE

        # Deterministic spread pattern based on target id to reduce stacking.
        ordered_ids = sorted(self.bb.detected_objects.keys())
        try:
            idx = ordered_ids.index(target_id)
        except ValueError:
            idx = 0
        fx, fy = self._DROP_PATTERN[idx % len(self._DROP_PATTERN)]
        x = cx + fx * usable_half_w
        y = cy + fy * usable_half_d

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(
            container['table_z'] +
            container['height'] +
            self._DROP_CLEARANCE_ABOVE_WALL_M
        )
        pose.orientation = copy.deepcopy(RobotInterface.TOP_DOWN_ORIENTATION)

        self.bb.drop_pose = pose
        self.robot.publish_pose_axes(pose, 'drop_proposal', scale=0.10)
        self.robot.log(
            f'[OK]   ProposeDropPose: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})'
        )
        return py_trees.common.Status.SUCCESS


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    robot = RobotInterface()

    init_blackboard()
    root = build_tree(robot)

    tree = py_trees_ros.trees.BehaviourTree(root=root, unicode_tree_debug=False)
    tree.setup(node=robot, timeout=15.0)

    def on_tick(t):
        log_panel = '\n'.join(robot.log_lines) if robot.log_lines else '(no log)'
        print('\033[2J\033[H' +
              py_trees.display.unicode_tree(t.root, show_status=True) +
              '\n' +
              py_trees.display.unicode_blackboard() +
              '\n─── log ───────────────────────────────\n' +
              log_panel)
        if t.root.status in (py_trees.common.Status.SUCCESS,
                             py_trees.common.Status.FAILURE):
            rclpy.shutdown()

    tree.post_tick_handlers.append(on_tick)
    robot.create_timer(0.2, tree.tick)

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        tree.shutdown()
        robot.destroy_node()


if __name__ == '__main__':
    main()
