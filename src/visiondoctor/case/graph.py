"""A small grasp dependency model, shared by the model and the workbench.

It describes the task, not a discovered installation. A node or edge acquires a
verdict only through an explicit, observed finding; neighbours do not turn green.

Nodes marked ``software`` are carried by a running program.  Their behaviour is
judged at the software layer from what the program read, wrote and logged; the
source layer can only be entered beneath one of them once that judgment exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .chain import Segment, SegmentFinding, SegmentStatus


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    segment: Segment
    knowledge_ids: tuple[str, ...]
    software: bool = False


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    name: str
    segment: Segment
    contract: str
    check: str
    knowledge_ids: tuple[str, ...]


NODES = (
    Node("task", "任务与验收", Segment.TASK_RESULT, ("task", "measurement")),
    Node("part", "工件与夹具", Segment.PART_SUPPLY, ("task",)),
    Node("imaging", "光学成像", Segment.IMAGING, ("optics",)),
    Node("acquisition", "采集与帧对应", Segment.ACQUISITION, ("timing", "software"), True),
    Node("perception", "目标识别与位姿", Segment.ALGORITHM, ("perception", "software"), True),
    Node("calibration", "相机与手眼标定", Segment.CALIBRATION, ("geometry",)),
    Node("planning", "抓取策略", Segment.PLANNING, ("planning", "software"), True),
    Node("tool", "工具几何与配置", Segment.CALIBRATION, ("geometry", "contact")),
    Node("interface", "指令计算与接口", Segment.INTERFACE, ("software", "geometry"), True),
    Node("robot", "机器人运动", Segment.ROBOT, ("execution",)),
    Node("contact", "夹持与保持", Segment.GRASPING, ("contact",)),
    Node("result", "独立结果观测", Segment.TASK_RESULT, ("measurement",)),
)

EDGES = (
    Edge("part_image", "part", "imaging", "工件可见性", Segment.IMAGING,
         "目标表面、光照与视角应提供完成本任务需要的特征。",
         "核对目标 ROI 的反光、遮挡与成像尺度；背景清晰不代表目标可测。", ("task", "optics")),
    Edge("image_frame", "imaging", "acquisition", "成像到帧", Segment.ACQUISITION,
         "帧应对应实际曝光事件，保留帧号与采集时间语义。",
         "区分曝光、接收、文件拷贝和导出时刻，检查丢帧及重复帧。", ("timing",)),
    Edge("frame_target", "acquisition", "perception", "帧与检测对应", Segment.ACQUISITION,
         "检测输入必须对应本次工件及正确图像/深度帧。",
         "核对帧号、对象 ID 和时间；考虑速度乘以延迟造成的位置差。", ("timing", "perception")),
    Edge("intrinsics_pose", "calibration", "perception", "投影与深度尺度", Segment.CALIBRATION,
         "内参应匹配图像分辨率、裁剪及深度尺度。",
         "将目标区域的三维点重投影；使用独立尺寸检查尺度，不能只看有效像素比例。",
         ("geometry", "measurement")),
    Edge("target_grasp", "perception", "planning", "目标到抓取点", Segment.PLANNING,
         "所选抓取点和方向应对应正确实例、几何和接触面。",
         "在同一对象上对照检测、抓取点和接近方向；视觉中心未必可抓。", ("perception", "planning")),
    Edge("task_grasp", "task", "planning", "工艺约束", Segment.PLANNING,
         "策略应满足本配方的姿态、夹持区域和动作限制。",
         "核对工件型号、配方、允许姿态及禁抓区域。", ("task", "planning")),
    Edge("tool_grasp", "tool", "planning", "工具与可抓性", Segment.PLANNING,
         "工具开口、接触方式和负载能力应适合目标。",
         "比较开口与工件尺寸、接触面及障碍间隙；不只检查点位可达。", ("planning", "contact")),
    Edge("handeye_command", "calibration", "interface", "手眼坐标交接", Segment.INTERFACE,
         "位姿交接应声明源/目标坐标系、单位和采样时刻。",
         "核对完整变换关系、配置和代码用法；内部一致性不证明物理标定正确。",
         ("geometry", "timing")),
    Edge("grasp_command", "planning", "interface", "抓取位姿到指令", Segment.INTERFACE,
         "期望 TCP 与控制器消费的位姿应按接口契约转换。",
         "逐项比较目标对象、坐标系、单位、TCP/法兰定义及约定来源。", ("geometry", "software")),
    Edge("tool_command", "tool", "interface", "TCP 与法兰交接", Segment.INTERFACE,
         "工具参数值和使用方向是两个不同问题，需分别检查。",
         "按 T_A_B 将 B 系变到 A 系的约定，T_base_flange = T_base_tcp @ inverse(T_flange_tcp)。",
         ("geometry",)),
    Edge("command_motion", "interface", "robot", "指令与实际运动", Segment.ROBOT,
         "实际轨迹应在规定时间和容差内跟踪有效指令。",
         "关联命令 ID、反馈时刻、实际关节/法兰与错误码；到位不代表抓住。", ("execution", "timing")),
    Edge("motion_contact", "robot", "contact", "到位与接触", Segment.GRASPING,
         "接近和闭合应在正确对象与接触位置发生。",
         "对照到位、闭合事件及工件运动；无接触证据时保留未查。", ("execution", "contact")),
    Edge("part_contact", "part", "contact", "工件与夹持", Segment.GRASPING,
         "摩擦、形变、重心和表面条件应允许稳定保持。",
         "查看抬起后的相对运动；区分目标变化、夹持设置与机械失效。", ("task", "contact")),
    Edge("tool_contact", "tool", "contact", "工具与保持", Segment.GRASPING,
         "闭合/真空命令需要适用的接触与保持证据。",
         "检查开口、力或压力和工件保持；工具类型不适用的信号不作要求。", ("contact",)),
    Edge("contact_result", "contact", "result", "保持效果", Segment.TASK_RESULT,
         "完整抓取成功需有工件被保持或抬起的观测。",
         "使用视频或适用传感器确认保持；位姿成功只支持到位。", ("contact", "measurement")),
    Edge("motion_result", "robot", "result", "到位测量", Segment.TASK_RESULT,
         "到位测量应说明参考来源、误差与观测阶段。",
         "检查测量独立性；同一错误配置生成的两个量可能同时吻合。", ("measurement",)),
    Edge("task_result", "task", "result", "结果与验收条件", Segment.TASK_RESULT,
         "成功含义、容差和覆盖工况应先于结果声明。",
         "核对结果是到位、接触还是保持；单次成功不能排除间歇故障。", ("task", "measurement")),
)

TARGETS = {item.id: item for item in (*NODES, *EDGES)}


def is_software_target(target_id: str) -> bool:
    """A software node, or a handoff consumed by one."""

    node = consuming_node(target_id)
    return bool(node) and TARGETS[node].software


def consuming_node(target_id: str | None) -> str | None:
    """A node itself, or the node that consumes a handoff."""

    target = TARGETS.get(target_id or "")
    if isinstance(target, Node):
        return target.id
    if isinstance(target, Edge):
        return target.target
    return None


def target_segment(target_id: str) -> Segment:
    if target_id not in TARGETS:
        raise ValueError(f"unknown grasp graph target: {target_id}")
    return TARGETS[target_id].segment


def graph_view(
    findings: list[SegmentFinding] | tuple[SegmentFinding, ...] = (),
    layers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Graph state; ``layers`` maps evidence ids so each finding shows what it rests on."""

    standing = {item.target_id: item for item in findings if item.target_id}
    layers = layers or {}

    def present(item: Node | Edge) -> dict[str, Any]:
        finding = standing.get(item.id)
        shown = finding.model_dump(mode="json") if finding else None
        if shown is not None:
            cited = [layers.get(name, "observation") for name in finding.evidence_ids]
            shown["layers"] = sorted(set(cited), key=("observation", "software", "source").index)
        return {
            **asdict(item),
            "software": is_software_target(item.id),
            "segment": item.segment.value,
            "status": finding.status.value if finding else SegmentStatus.UNTESTED.value,
            "finding": shown,
        }

    return {
        "id": "grasp-reference-v1",
        "basis": "抓取任务参考结构，需用本工位证据确认；未观测的能力不视为已实现。",
        "nodes": [present(node) for node in NODES],
        "edges": [present(edge) for edge in EDGES],
    }


def inspect_target(target_id: str) -> dict[str, Any]:
    target_segment(target_id)
    target = TARGETS[target_id]
    edges = [edge for edge in EDGES if target_id in (edge.id, edge.source, edge.target)]
    node_ids = {name for edge in edges for name in (edge.source, edge.target)} | {target_id}
    return {
        "kind": "domain_guidance",
        "target": asdict(target),
        "nodes": [asdict(node) for node in NODES if node.id in node_ids],
        "edges": [asdict(edge) for edge in edges],
        "limits": "依赖关系帮助提出假设，不证明某个组件或交接已出错。",
    }
