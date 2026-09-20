"""The shared map a demarcation is written on.

The chain is not a fixed troubleshooting order and not the project's directory
tree.  It exists so that "where did this first go wrong" has addressable answers
that two people -- or a person and an agent -- can disagree about precisely.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Segment(StrEnum):
    PART_SUPPLY = "part_supply"
    IMAGING = "imaging"
    ACQUISITION = "acquisition"
    ALGORITHM = "algorithm"
    TASK_RESULT = "task_result"
    CALIBRATION = "calibration"
    INTERFACE = "interface"
    ROBOT = "robot"
    PLANNING = "planning"
    GRASPING = "grasping"


#: Every vision cell has these.
CORE_SEGMENTS: tuple[Segment, ...] = (
    Segment.PART_SUPPLY,
    Segment.IMAGING,
    Segment.ACQUISITION,
    Segment.ALGORITHM,
    Segment.TASK_RESULT,
)

#: Grasp-specific groups. The dependency graph carries the actual relationships.
GUIDED_SEGMENTS: tuple[Segment, ...] = (
    Segment.CALIBRATION,
    Segment.INTERFACE,
    Segment.ROBOT,
    Segment.PLANNING,
    Segment.GRASPING,
)


SEGMENT_NAME: dict[Segment, str] = {
    Segment.PART_SUPPLY: "工件与来料",
    Segment.IMAGING: "光学成像",
    Segment.ACQUISITION: "采集与时间链路",
    Segment.ALGORITHM: "视觉算法",
    Segment.TASK_RESULT: "任务结果",
    Segment.CALIBRATION: "标定配置",
    Segment.INTERFACE: "变换接口",
    Segment.ROBOT: "机器人侧",
    Segment.PLANNING: "抓取策略",
    Segment.GRASPING: "夹持与保持",
}

#: What each segment covers.  A map is only shared if both sides read it the
#: same way; these say what a segment is responsible for, never what is wrong
#: with any particular cell.
SEGMENT_SCOPE: dict[Segment, str] = {
    Segment.PART_SUPPLY: "工件与来料状态：种类、摆放姿态、表面状况、上下料的一致性。",
    Segment.IMAGING: "成像本身：照明、曝光、对焦、镜头、内参，以及图像与深度的画面质量。",
    Segment.ACQUISITION: "取图与传输链路：触发时机、帧序、时间戳、丢帧、多路数据的同步。",
    Segment.ALGORITHM: "从图像算到视觉输出：预处理、检测识别、位姿估计及其置信度。",
    Segment.TASK_RESULT: "本次任务的成败与指标本身，以及它是否如实反映了现场。",
    Segment.CALIBRATION: (
        "已标定并版本化的参数值：相机内外参、手眼关系、工具几何。"
        "这一段问的是这些数值本身对不对。"
    ),
    Segment.INTERFACE: (
        "视觉输出被下游消费时所依赖的约定：坐标系、变换、单位、字段契约。"
        "这一段问的是这些数值被怎么使用。"
    ),
    Segment.ROBOT: "机器人执行：轨迹、伺服跟踪、实际到位与机械状态。",
    Segment.PLANNING: "从目标选择抓取点、接近方向与路径，满足工艺、工具和可达性约束。",
    Segment.GRASPING: "到位之后的接触、闭合、夹持与抬起保持；到位成功不证明抓住。",
}


class SegmentStatus(StrEnum):
    UNTESTED = "untested"
    CLEARED = "cleared"
    SUSPECT = "suspect"


class SegmentFinding(BaseModel):
    """A verdict on one segment.

    Anything but UNTESTED has to name the evidence it rests on.  This is the
    single rule that keeps a demarcation from being an opinion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment: Segment
    status: SegmentStatus
    note: str
    evidence_ids: tuple[str, ...] = ()
    target_id: str | None = None
    checked_scope: str = ""
    limitations: str = ""

    @model_validator(mode="after")
    def _conclusions_cite_evidence(self) -> SegmentFinding:
        if self.status is not SegmentStatus.UNTESTED and not self.evidence_ids:
            raise ValueError(f"{self.status} on {self.segment} cites no evidence")
        return self


class Hypothesis(BaseModel):
    """An explanation still competing with others, and what it rests on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    target_segment: Segment
    statement: str
    evidence_ids: tuple[str, ...] = ()
    target_id: str | None = None
    prediction: str = ""
    counter_evidence_ids: tuple[str, ...] = ()
    next_check: str = ""
    remedy: Literal[
        "configuration", "handoff", "source_patch", "physical", "more_evidence"
    ] = "more_evidence"


def chain_for(*, guided_motion: bool) -> tuple[Segment, ...]:
    """The segments drawn on the map for this case."""

    return CORE_SEGMENTS + (GUIDED_SEGMENTS if guided_motion else ())
