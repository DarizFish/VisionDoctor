"""Diagnostic templates: small computable pieces of the grasp graph.

A template records structure only -- which equation holds which variables,
which records can measure a variable, and which graph node a fault would break.
Nothing here says what went wrong in any cell.  Which faults can be detected or
told apart is derived from this structure at run time; the residual of each
test is computed by a hand-written evaluator after the structure has admitted it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Equation:
    id: str
    relation: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class Measurement:
    """A kind of record; it makes a variable known only if it carries that variable's fields."""

    id: str
    record: str
    provides: dict[str, tuple[str, ...]]
    #: Taken from a different run, such as a normal reference; an array needs no field.
    reference: bool = False


@dataclass(frozen=True)
class Fault:
    """One way a graph node can fail, and the equations that failure would break.

    The id is ``node`` or ``node.mode``: a node that can fail in more than one way
    (a stale frame, a depth handoff that loses data) has one fault per way, so a
    test that clears one way does not clear the others.  Modes stay coarse --
    classes of failure, never a mechanism.
    """

    id: str
    name: str
    equations: tuple[str, ...]

    @property
    def node(self) -> str:
        return self.id.split(".", maxsplit=1)[0]


@dataclass(frozen=True)
class Test:
    """A residual evaluator written in advance over a declared set of equations."""

    id: str
    name: str
    equations: tuple[str, ...]
    measurements: tuple[str, ...]
    residual: str
    threshold: dict[str, float]
    threshold_source: str


@dataclass(frozen=True)
class Template:
    id: str
    name: str
    variables: dict[str, str]
    equations: tuple[Equation, ...]
    measurements: tuple[Measurement, ...]
    faults: tuple[Fault, ...]
    tests: tuple[Test, ...]


GEOMETRY = Template(
    id="geometry_chain",
    name="几何指令链",
    variables={
        "d": "感知输出的工件相机系位姿",
        "p_used": "感知所用帧中工件的真实相机系位姿",
        "p": "本次采集时工件的真实相机系位姿",
        "K": "相机到基座外参（配置）",
        "K_true": "相机到基座外参（物理）",
        "O": "抓取偏移（配置）",
        "O_true": "抓取偏移（实际需要）",
        "G": "法兰到 TCP 工具变换（配置）",
        "G_true": "法兰到 TCP 工具变换（物理）",
        "w": "抓取时工件在基座系的真实位姿",
        "x": "程序内部的期望 TCP（未记录）",
        "f_cmd": "法兰指令",
        "f_act": "实际法兰",
        "t": "实际 TCP",
        "err": "独立测得的到位误差",
    },
    equations=(
        Equation("e1", "检测位姿 d = 所用帧中的工件位姿 p_used", ("d", "p_used")),
        Equation("e2", "所用帧中的位姿 p_used = 本次采集时的位姿 p", ("p_used", "p")),
        Equation("e3", "物理外参 K_true = 外参配置 K", ("K_true", "K")),
        Equation("e4", "抓取时工件位姿 w = K_true · p", ("w", "K_true", "p")),
        Equation("e5", "实际需要的偏移 O_true = 偏移配置 O", ("O_true", "O")),
        Equation("e6", "物理工具 G_true = 工具配置 G", ("G_true", "G")),
        Equation("e7", "期望 TCP x = K · d · O", ("x", "K", "d", "O")),
        Equation("e8", "法兰指令 f_cmd = x · G⁻¹", ("f_cmd", "x", "G")),
        Equation("e9", "实际法兰 f_act = 法兰指令 f_cmd", ("f_act", "f_cmd")),
        Equation("e10", "实际 TCP t = f_act · G_true", ("t", "f_act", "G_true")),
        Equation("e11", "到位误差 err = ‖t − w · O_true‖", ("err", "t", "w", "O_true")),
    ),
    measurements=(
        Measurement("detection", "感知记录（algorithm/input.json）：detected_part_in_camera",
                    {"d": ("detected_part_in_camera",)}),
        Measurement("calibration", "标定配置（cell_calibration.yaml）：camera_to_base、"
                    "pick_offset_from_part",
                    {"K": ("camera_to_base",), "O": ("pick_offset_from_part",)}),
        Measurement("tool", "工具配置（tool_profile.yaml）：tool0_to_tcp",
                    {"G": ("tool0_to_tcp",)}),
        Measurement("command", "指令记录（algorithm/output.json）：commanded_flange_base",
                    {"f_cmd": ("commanded_flange_base",)}),
        Measurement("motion", "运动记录（motion.json）：actual_flange_base",
                    {"f_act": ("actual_flange_base",)}),
        Measurement("result", "独立到位测量（result.json 或任务结果）：position_error_m、"
                    "rotation_error_rad",
                    {"err": ("position_error_m", "rotation_error_rad")}),
    ),
    faults=(
        Fault("perception", "目标识别与位姿：位姿估计偏差", ("e1",)),
        Fault("acquisition.frame", "采集与帧对应：用错帧", ("e2",)),
        Fault("calibration", "相机与手眼标定：外参与物理不符", ("e3",)),
        Fault("part.moved", "工件与夹具：采集后移动", ("e4",)),
        Fault("planning", "抓取策略：偏移与实际需要不符", ("e5",)),
        Fault("tool", "工具几何与配置：工具与物理不符", ("e6",)),
        Fault("interface", "指令计算与接口：指令算错", ("e7", "e8")),
        Fault("robot", "机器人运动：未跟踪指令", ("e9",)),
    ),
    tests=(
        Test("A", "指令一致", ("e7", "e8"), ("detection", "calibration", "tool", "command"),
             "command_consistency", {"position_m": 0.001, "rotation_rad": 0.005},
             "模板默认：同一算式重算，只容许数值舍入"),
        Test("B", "运动跟踪", ("e9",), ("command", "motion"),
             "motion_tracking", {"position_m": 0.005, "rotation_rad": 0.02},
             "模板默认：项目未声明跟踪容差"),
        Test("C", "到位一致", ("e1", "e2", "e3", "e4", "e5", "e6", "e10", "e11"),
             ("detection", "calibration", "tool", "motion", "result"),
             "outcome_consistency", {"position_m": 0.005, "rotation_rad": 0.005},
             "模板默认：项目未声明到位测量不确定度"),
    ),
)

FRAME = Template(
    id="frame_correspondence",
    name="帧对应",
    variables={
        "capture_used": "感知所用采集的标识与源帧时刻",
        "capture_now": "本次动作前采集的标识与时刻",
    },
    equations=(
        Equation("e12", "感知所用采集 = 本次动作前的采集", ("capture_used", "capture_now")),
    ),
    measurements=(
        Measurement("detection", "感知记录（algorithm/input.json）：capture_id、source_stamp_s、"
                    "clock_domain", {"capture_used": ("capture_id",)}),
        Measurement("capture", "本次采集记录（capture/capture.json）：capture_id、"
                    "sensor_stamps_s、clock_domain", {"capture_now": ("capture_id",)}),
    ),
    faults=(Fault("acquisition.frame", "采集与帧对应：用错帧", ("e12",)),),
    tests=(
        Test("D", "帧对应", ("e12",), ("detection", "capture"),
             "frame_correspondence", {"age_s": 0.05},
             "模板默认：同一次采集的源帧时刻应一致"),
    ),
)

DEPTH = Template(
    id="depth_handoff",
    name="RGB-D 交接",
    variables={
        "r": "传感器原始深度在目标区域的有效比例",
        "q": "目标表面在本工况下的真实可测程度",
        "r_ref": "正常参考运行同一区域的有效比例",
        "c": "程序消费的深度在目标区域的有效比例",
        "s": "感知是否输出位姿",
        "theta": "感知配置的最低目标深度比例",
    },
    equations=(
        Equation("e13", "原始比例 r = 真实可测程度 q", ("r", "q")),
        Equation("e14", "真实可测程度 q = 参考运行比例 r_ref（工件与表面条件相同）",
                 ("q", "r_ref")),
        Equation("e15", "消费比例 c = 原始比例 r", ("c", "r")),
        Equation("e16", "感知输出 s ⇔ c ≥ theta", ("s", "c", "theta")),
    ),
    measurements=(
        Measurement("detection", "感知记录（algorithm/input.json）：roi_xyxy、status",
                    {"s": ("roi_xyxy", "status")}),
        Measurement("raw_depth", "传感器原始深度（capture/raw-depth.npy）", {"r": ()}),
        Measurement("consumed_depth", "程序实际消费的深度（capture/depth.npy）", {"c": ()}),
        Measurement("perception_config", "感知配置（perception.yaml）：minimum_target_depth_ratio",
                    {"theta": ("minimum_target_depth_ratio",)}),
        Measurement("reference_depth", "正常参考运行同一工件的深度（另一运行的 capture/depth.npy）",
                    {"r_ref": ()}, reference=True),
    ),
    faults=(
        Fault("imaging.depth", "光学成像：目标区域深度不可测", ("e13",)),
        Fault("part.surface", "工件与夹具：表面条件与参考不同", ("e14",)),
        Fault("acquisition.depth", "采集与帧对应：深度交接丢数据", ("e15",)),
        Fault("perception.rule", "目标识别与位姿：不按声明规则拒绝", ("e16",)),
    ),
    tests=(
        Test("F", "交接保真", ("e15",), ("detection", "raw_depth", "consumed_depth"),
             "depth_handoff", {"ratio_gap": 0.05}, "模板默认：同一帧的原始与消费数据应一致"),
        Test("G", "感知守约", ("e16",), ("detection", "consumed_depth", "perception_config"),
             "perception_rule", {"rule_mismatch": 0.5}, "阈值来自项目感知配置"),
        Test("H", "原始可测", ("e13", "e14"), ("detection", "raw_depth", "reference_depth"),
             "raw_measurable", {"ratio_gap": 0.1}, "模板默认：与参考运行同区域比较"),
    ),
)

TEMPLATES: dict[str, Template] = {item.id: item for item in (GEOMETRY, FRAME, DEPTH)}


def template_catalogue() -> list[dict]:
    """What the model is told a template needs, without anything it would conclude."""

    return [
        {
            "template_id": item.id,
            "name": item.name,
            "faults": [{"id": fault.id, "name": fault.name} for fault in item.faults],
            "measurements": [
                {"id": measurement.id, "record": measurement.record}
                for measurement in item.measurements
            ],
            "tests": [
                {"id": test.id, "name": test.name, "measurements": list(test.measurements)}
                for test in item.tests
            ],
        }
        for item in TEMPLATES.values()
    ]
