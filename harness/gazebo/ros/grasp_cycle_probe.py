from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
SAFE_APPROACH_JOINTS = (0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0)
JOINT_ARRIVAL_TOLERANCE_RAD = 0.005


class GraspCycleProbe(Node):
    def __init__(self) -> None:
        super().__init__("pick_cell_grasp_cycle")
        self.latest_joints: dict[str, float] = {}
        self.joint_trace: list[dict[str, Any]] = []
        self._record_trace = False
        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.compute_ik = self.create_client(GetPositionIK, "/compute_ik")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "/joint_states", self._joint_state, 30)

    def _joint_state(self, message: JointState) -> None:
        self.latest_joints = dict(zip(message.name, message.position, strict=True))
        if self._record_trace and all(name in self.latest_joints for name in JOINT_NAMES):
            stamp_s = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            if self.joint_trace and stamp_s - float(self.joint_trace[-1]["stamp_s"]) < 0.1:
                return
            self.joint_trace.append(
                {
                    "stamp_s": stamp_s,
                    "joint_position_rad": [self.latest_joints[name] for name in JOINT_NAMES],
                }
            )

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        if not self.move_group.wait_for_server(timeout_sec=timeout_s):
            raise RuntimeError("MoveIt action server was not ready")
        if not self.compute_ik.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("MoveIt IK service was not ready")
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(name in self.latest_joints for name in JOINT_NAMES):
                return
        raise RuntimeError("UR5e joint state was not received")

    def execute_pose(self, name: str, target: dict[str, list[float]]) -> dict[str, Any]:
        joint_target = self._joint_target_from_ik(target)
        return self.execute_joint_target(name, joint_target, target)

    def _joint_target_from_ik(self, target: dict[str, list[float]]) -> list[float]:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target["position"]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = target[
            "quaternion_xyzw"
        ]
        request = GetPositionIK.Request()
        request.ik_request.group_name = "ur_manipulator"
        request.ik_request.ik_link_name = "tool0"
        request.ik_request.pose_stamped.header.frame_id = "base_link"
        request.ik_request.pose_stamped.pose = pose
        # Supply a complete six-axis UR seed rather than merging against the
        # momentary simulated state.  The latter lets MoveIt choose a different
        # equivalent branch between runs, which is not suitable for a repeatable
        # A/B demonstration.
        request.ik_request.robot_state.is_diff = False
        seed_value = os.environ.get("PICK_CELL_IK_SEED")
        if seed_value:
            seed = json.loads(seed_value)
            if not isinstance(seed, list) or len(seed) != len(JOINT_NAMES):
                raise ValueError("PICK_CELL_IK_SEED must contain six joint values")
            request.ik_request.robot_state.joint_state.name = list(JOINT_NAMES)
            request.ik_request.robot_state.joint_state.position = [float(value) for value in seed]
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout = Duration(seconds=2.0).to_msg()
        future = self.compute_ik.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if response is None else response.error_code.val
            raise RuntimeError(f"MoveIt IK failed: {code}")
        solved = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
                strict=True,
            )
        )
        if not all(name in solved for name in JOINT_NAMES):
            raise RuntimeError("MoveIt IK response lacks UR5e joint values")
        # The IK service may return an equivalent continuous-joint value on a
        # different 2π turn.  Keep the command on the turn nearest the live
        # state: otherwise a nearby TCP target can look like a full-revolution
        # controller command and abort before the intended pick begins.
        return [
            self._nearest_equivalent_angle(float(solved[name]), self.latest_joints[name])
            for name in JOINT_NAMES
        ]

    @staticmethod
    def _nearest_equivalent_angle(target: float, current: float) -> float:
        return current + (target - current + math.pi) % math.tau - math.pi

    def _joint_target_reached(self, target: list[float] | tuple[float, ...]) -> bool:
        """Confirm the simulated arm has physically settled at the planned target."""

        if not all(name in self.latest_joints for name in JOINT_NAMES):
            return False
        for name, goal in zip(JOINT_NAMES, target, strict=True):
            current = self.latest_joints[name]
            nearest_goal = self._nearest_equivalent_angle(float(goal), current)
            if abs(nearest_goal - current) > JOINT_ARRIVAL_TOLERANCE_RAD:
                return False
        return True

    def execute_joint_target(
        self,
        name: str,
        joint_target: list[float] | tuple[float, ...],
        target_flange: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = "RRTConnectkConfigDefault"
        goal.request.num_planning_attempts = 4
        goal.request.allowed_planning_time = 10.0
        # The mounted visual gripper makes the last wrist movement less
        # forgiving than the bare stock arm.  A moderate limit keeps the
        # controller inside its final-position tolerance while remaining
        # short enough for the live demonstration.
        goal.request.max_velocity_scaling_factor = 0.25
        goal.request.max_acceleration_scaling_factor = 0.25
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [
            Constraints(
                name=name,
                joint_constraints=[
                    JointConstraint(
                        joint_name=joint_name,
                        position=float(joint_value),
                        tolerance_above=0.003,
                        tolerance_below=0.003,
                        weight=1.0,
                    )
                    for joint_name, joint_value in zip(JOINT_NAMES, joint_target, strict=True)
                ],
            )
        ]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self._record_trace = True
        started = time.monotonic()
        send = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=20.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            self._record_trace = False
            raise RuntimeError(f"MoveIt rejected {name}")
        result_future = handle.get_result_async()
        # Gazebo's scaled controller can leave MoveIt waiting for an action
        # result even after the physical joint feedback has settled.  For this
        # arrival-based demonstration the observed joint target is definitive:
        # record it, cancel the latched action, then allow the next step.
        deadline = time.monotonic() + 150.0
        settled_at: float | None = None
        while time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._joint_target_reached(joint_target):
                if settled_at is None:
                    settled_at = time.monotonic()
                elif time.monotonic() - settled_at >= 0.6:
                    cancel = handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel, timeout_sec=10.0)
                    self._record_trace = False
                    return {
                        "name": name,
                        "moveit_error_code": int(MoveItErrorCodes.SUCCESS),
                        "completion": "joint_target_observed",
                        "duration_s": time.monotonic() - started,
                        "target_flange_base": target_flange,
                        "target_joints_rad": list(joint_target),
                        "actual_joints_rad": [self.latest_joints[joint] for joint in JOINT_NAMES],
                    }
            else:
                settled_at = None
        self._record_trace = False
        wrapped = result_future.result()
        if wrapped is None:
            # Leave nothing running behind us.  A trajectory still latched in the
            # execution manager rejects every later goal in the same cycle.
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel, timeout_sec=10.0)
            raise RuntimeError(f"MoveIt timed out at {name}")
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
        return {
            "name": name,
            "moveit_error_code": int(wrapped.result.error_code.val),
            "duration_s": time.monotonic() - started,
            "target_flange_base": target_flange,
            "target_joints_rad": list(joint_target),
            "actual_joints_rad": [self.latest_joints[name] for name in JOINT_NAMES],
        }

    def actual_flange(self) -> dict[str, list[float]]:
        deadline = time.monotonic() + 5.0
        last_error = ""
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_link", "tool0", rclpy.time.Time()
                ).transform
                return {
                    "position": [
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ],
                    "quaternion_xyzw": [
                        transform.rotation.x,
                        transform.rotation.y,
                        transform.rotation.z,
                        transform.rotation.w,
                    ],
                }
            except Exception as exc:
                last_error = str(exc)
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("tool0 transform was unavailable: " + last_error)


def _pose_from_environment(name: str, *, required: bool = True) -> dict[str, list[float]] | None:
    encoded = os.environ.get(name)
    if encoded is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    value = json.loads(encoded)
    position = value.get("position")
    orientation = value.get("quaternion_xyzw")
    if not isinstance(position, list) or not isinstance(orientation, list):
        raise ValueError("flange target is not a pose")
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError("flange target dimensions are invalid")
    return {
        "position": [float(item) for item in position],
        "quaternion_xyzw": [float(item) for item in orientation],
    }


def _target_from_environment() -> dict[str, list[float]]:
    target = _pose_from_environment("PICK_CELL_TARGET_FLANGE")
    assert target is not None
    return target


def main() -> int:
    rclpy.init()
    node = GraspCycleProbe()
    steps: list[dict[str, Any]] = []
    payload: dict[str, Any]
    try:
        target = _target_from_environment()
        pregrasp = _pose_from_environment("PICK_CELL_PREGRASP_FLANGE", required=False)
        node.wait_ready(60.0)
        approach = node.execute_joint_target("APPROACH", SAFE_APPROACH_JOINTS)
        steps.append(approach)
        if approach["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt failed APPROACH: {approach['moveit_error_code']}")
        if pregrasp is not None:
            above = node.execute_pose("PREGRASP", pregrasp)
            steps.append(above)
            if above["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"MoveIt failed PREGRASP: {above['moveit_error_code']}")
        pick = node.execute_pose("PICK", target)
        steps.append(pick)
        if pick["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt failed PICK: {pick['moveit_error_code']}")
        actual = node.actual_flange()
        if pregrasp is not None:
            retract = node.execute_pose("RETRACT_TO_PREGRASP", pregrasp)
            steps.append(retract)
            if retract["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(
                    f"MoveIt failed RETRACT_TO_PREGRASP: {retract['moveit_error_code']}"
                )
        retreat = node.execute_joint_target("RETREAT", SAFE_APPROACH_JOINTS)
        steps.append(retreat)
        succeeded = retreat["moveit_error_code"] == MoveItErrorCodes.SUCCESS
        payload = {
            "success": succeeded,
            "robot": "UR5e",
            "planner": "MoveIt 2 / OMPL RRTConnect",
            "steps": steps,
            "actual_flange_base": actual,
            "joint_trajectory": node.joint_trace,
        }
    except Exception as exc:
        payload = {
            "success": False,
            "error": str(exc),
            "steps": steps,
            "joint_trajectory": node.joint_trace,
        }
    finally:
        output = os.environ.get("PICK_CELL_MOTION_OUTPUT")
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print("PICK_CELL_MOTION=" + json.dumps(payload, sort_keys=True), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
