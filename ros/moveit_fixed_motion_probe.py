from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetPositionFK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HOME = (0.0, -1.57, 0.0, -1.57, 0.0, 0.0)
OBSERVATION = (-0.15, -1.45, -0.20, -1.55, 0.10, 0.0)
VALIDATION = (0.20, -1.35, -0.35, -1.45, 0.20, 0.10)


@dataclass(frozen=True)
class MotionStep:
    name: str
    target: tuple[float, ...]


class FixedMotionProbe(Node):
    def __init__(self) -> None:
        super().__init__("visiondoctor_fixed_motion_probe")
        self.latest_joints: dict[str, float] = {}
        self.create_subscription(JointState, "/joint_states", self._joint_state, 10)
        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.compute_fk = self.create_client(GetPositionFK, "/compute_fk")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _joint_state(self, message: JointState) -> None:
        self.latest_joints = dict(zip(message.name, message.position, strict=True))

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        if not self.move_group.wait_for_server(timeout_sec=timeout_s):
            raise RuntimeError("MoveIt /move_action server was not ready")
        remaining = max(0.1, deadline - time.monotonic())
        if not self.compute_fk.wait_for_service(timeout_sec=remaining):
            raise RuntimeError("MoveIt /compute_fk service was not ready")
        while time.monotonic() < deadline and not all(
            name in self.latest_joints for name in JOINT_NAMES
        ):
            rclpy.spin_once(self, timeout_sec=0.1)
        if not all(name in self.latest_joints for name in JOINT_NAMES):
            raise RuntimeError("Gazebo joint states were not received")

    def expected_tcp(self, joints: tuple[float, ...]) -> dict[str, list[float]]:
        request = GetPositionFK.Request()
        request.header.frame_id = "base_link"
        request.fk_link_names = ["tool0"]
        request.robot_state.joint_state.name = list(JOINT_NAMES)
        request.robot_state.joint_state.position = list(joints)
        future = self.compute_fk.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if response is None else response.error_code.val
            raise RuntimeError(f"MoveIt FK failed with code {code}")
        pose = response.pose_stamped[0].pose
        return {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "quaternion_xyzw": [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        }

    def execute(self, step: MotionStep) -> dict[str, object]:
        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = "RRTConnectkConfigDefault"
        goal.request.num_planning_attempts = 3
        goal.request.allowed_planning_time = 8.0
        goal.request.max_velocity_scaling_factor = 0.15
        goal.request.max_acceleration_scaling_factor = 0.15
        goal.request.start_state.is_diff = True
        constraints = Constraints(name=step.name)
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=position,
                tolerance_above=0.003,
                tolerance_below=0.003,
                weight=1.0,
            )
            for name, position in zip(JOINT_NAMES, step.target, strict=True)
        ]
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        started = time.monotonic()
        send_future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"MoveIt rejected the {step.name} goal")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=45.0)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError(f"MoveIt timed out executing {step.name}")
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)
        actual = [self.latest_joints[name] for name in JOINT_NAMES]
        return {
            "name": step.name,
            "moveit_error_code": wrapped.result.error_code.val,
            "duration_s": time.monotonic() - started,
            "target_joints": list(step.target),
            "actual_joints": actual,
            "max_joint_error_rad": max(
                abs(target - measured)
                for target, measured in zip(step.target, actual, strict=True)
            ),
        }

    def execute_pose(self, name: str, target: dict[str, list[float]]) -> dict[str, object]:
        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = "RRTConnectkConfigDefault"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 12.0
        goal.request.max_velocity_scaling_factor = 0.15
        goal.request.max_acceleration_scaling_factor = 0.15
        goal.request.start_state.is_diff = True

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target["position"]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = target["quaternion_xyzw"]
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.003])
        position = PositionConstraint(
            link_name="tool0",
            constraint_region=BoundingVolume(primitives=[sphere], primitive_poses=[pose]),
            weight=1.0,
        )
        position.header.frame_id = "base_link"
        orientation = OrientationConstraint(
            link_name="tool0",
            orientation=pose.orientation,
            absolute_x_axis_tolerance=0.01,
            absolute_y_axis_tolerance=0.01,
            absolute_z_axis_tolerance=0.01,
            weight=1.0,
        )
        orientation.header.frame_id = "base_link"
        goal.request.goal_constraints = [
            Constraints(
                name=name,
                position_constraints=[position],
                orientation_constraints=[orientation],
            )
        ]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        started = time.monotonic()
        send_future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=20.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"MoveIt rejected the {name} pose goal")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError(f"MoveIt timed out executing {name}")
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)
        return {
            "name": name,
            "moveit_error_code": wrapped.result.error_code.val,
            "duration_s": time.monotonic() - started,
            "target_tcp": target,
            "actual_joints": [self.latest_joints[joint] for joint in JOINT_NAMES],
        }

    def actual_tcp(self) -> dict[str, list[float]]:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_link",
                    "tool0",
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
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
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("TF base_link -> tool0 was not available")


def _pose_errors(
    expected: dict[str, list[float]], actual: dict[str, list[float]]
) -> tuple[float, float]:
    translation = math.sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(expected["position"], actual["position"], strict=True)
        )
    )
    dot = abs(
        sum(
            left * right
            for left, right in zip(
                expected["quaternion_xyzw"], actual["quaternion_xyzw"], strict=True
            )
        )
    )
    rotation = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
    return translation, rotation


def main() -> int:
    rclpy.init()
    node = FixedMotionProbe()
    results: list[dict[str, object]] = []
    failure_stage = "startup"
    moveit_error_code: int | None = None
    try:
        node.wait_ready(60.0)
        target_json = os.getenv("VISIONDOCTOR_TARGET_TCP")
        expected_validation_tcp = (
            json.loads(target_json) if target_json else node.expected_tcp(VALIDATION)
        )
        for step in (MotionStep("HOME", HOME), MotionStep("OBSERVATION_POSE", OBSERVATION)):
            failure_stage = step.name
            result = node.execute(step)
            results.append(result)
            if result["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
                moveit_error_code = int(result["moveit_error_code"])
                raise RuntimeError(
                    f"MoveIt failed {step.name}: error {result['moveit_error_code']}"
                )
        failure_stage = "VALIDATION_POSE"
        validation_result = (
            node.execute_pose("VALIDATION_POSE", expected_validation_tcp)
            if target_json
            else node.execute(MotionStep("VALIDATION_POSE", VALIDATION))
        )
        results.append(validation_result)
        if validation_result["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
            moveit_error_code = int(validation_result["moveit_error_code"])
            raise RuntimeError(
                "MoveIt failed VALIDATION_POSE: "
                f"error {validation_result['moveit_error_code']}"
            )
        actual_validation_tcp = node.actual_tcp()
        translation_error, rotation_error = _pose_errors(
            expected_validation_tcp, actual_validation_tcp
        )
        failure_stage = "FINAL_HOME"
        final_home = node.execute(MotionStep("HOME", HOME))
        results.append(final_home)
        if final_home["moveit_error_code"] != MoveItErrorCodes.SUCCESS:
            moveit_error_code = int(final_home["moveit_error_code"])
        succeeded = (
            final_home["moveit_error_code"] == MoveItErrorCodes.SUCCESS
            and max(float(item.get("max_joint_error_rad", 0.0)) for item in results) <= 0.01
            and translation_error <= 0.005
            and rotation_error <= 0.01745
        )
        payload = {
            "success": succeeded,
            "backend": "gazebo",
            "planner": "MoveIt 2 / OMPL RRTConnect",
            "robot": "UR5e",
            "runtime_uid": os.getuid(),
            "motion_sequence": [item["name"] for item in results],
            "steps": results,
            "expected_validation_tcp": expected_validation_tcp,
            "actual_validation_tcp": actual_validation_tcp,
            "tcp_translation_error_m": translation_error,
            "tcp_rotation_error_rad": rotation_error,
        }
        print("VISIONDOCTOR_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if succeeded else 2
    except Exception as exc:
        failure_payload = {
            "success": False,
            "error": str(exc),
            "failure_stage": failure_stage,
            "steps": results,
        }
        if moveit_error_code is not None:
            failure_payload["moveit_error_code"] = moveit_error_code
        print(
            "VISIONDOCTOR_RESULT="
            + json.dumps(failure_payload, sort_keys=True),
            flush=True,
        )
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
