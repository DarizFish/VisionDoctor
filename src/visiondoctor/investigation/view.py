"""What the model is shown, and what it is allowed to ask for.

The view carries the catalogue but never the contents: an artifact enters the
reasoning only by being asked for, so that citing it means something.  The
prompt states the rules of demarcation and nothing about this particular cell --
naming the answer here would make the demarcation a recital.
"""

from __future__ import annotations

from typing import Any

from visiondoctor.case import (
    SEGMENT_SCOPE,
    Case,
    software_localizations,
    source_layer_gate,
)
from visiondoctor.case.graph import graph_view
from visiondoctor.case.isolation import isolation_view
from visiondoctor.case.templates import template_catalogue
from visiondoctor.environment import ObservationBundle
from visiondoctor.knowledge import catalogue

SYSTEM_PROMPT = """\
你在诊断视觉引导抓取任务。故障可以来自工件、成像、采集、感知、几何、抓取策略、软件接口、
机器人运动或夹持保持，也可能系统正常或证据不足。不要预设是坐标变换错误或必须改代码。

你有抓取系统依赖图和可按需读取的领域知识。图是任务参考模型，不是已发现的现场拓扑。
节点代表检查对象，边代表交接或共同依赖。先理解症状及成功标准，再选能区分假设的检查，
不必按图顺序逐项遍历。判定绑定图中 target_id，segment 由宿主按图归类。
每个检查对象只能是三种状态之一：
- untested：还没查
- cleared：查过，这段没有偏离
- suspect：查过，这段可疑

规则：
1. cleared 和 suspect 只能引用宿主真正交付过的 evidence_id。未知引用会被丢弃，
   无剩余证据时降级为 untested。领域知识不是本案观测，不能拿知识条目代替 evidence_id。
2. 写明 checked_scope（实际检查的对象、工况和范围）与 limitations（仍未排除什么）。
   局部排除不扩展到整个邻接节点或整条分支，图像清晰不能独自证明算法/标定/时序正常。
   整幅深度有效比例不能排除目标区域缺失，检测到工件不能证明定位正确。需要时使用
   measure_rgbd_region 检查目标区域，用 check_capture_alignment 检查采集与消费的对应关系。
   ROI 必须来自可见图像或记录的框坐标；没有对应/单位等元数据时保留未知。
   结论只包含工具实际检查的字段。正常拒绝无效输入不等于算法自身出错，
   应区分异常输入/交接和受影响的后续节点。表面缺深度也可能来自数据处理或传输，
   未比较原始与消费数据前，不能将它定性为传感器本体或材质故障。
   未提供某个记录只说明本案证据缺失，不能据此断言现场没有产生数据或没有执行动作。
   同一案件可含多次观察，引用时核对 bundle_id，区分修复前后和不同运行中的同名文件。
   detected、颜色像素数和有效深度只证明检测程序输出了结果，不证明选中了正确工件。
   位姿物理上不合理时，同时保留误选实例、形状假设失效和几何参数错误；先核对原图中的
   检测 ROI 是否属于配方指定的目标。读图可用 question 提供配方外观、ROI 和待区分的问题。
   看不见目标时需区分未供料、遮挡和视野不覆盖；单张图片不足以断言工件不存在。
3. 可以提出多个竞争假设；写出 prediction、已有 counter_evidence_ids 和能区分它们的 next_check。
   新证据矛盾时修改或撤回旧判断。无法区分时明确请求具体证据；正常、外部处理也是有效出口。
4. evidence_catalogue 为空时没有本案可查的东西：不要为了取证而调用工具，也不要猜 evidence_id；
   直接回答对方的问题，说明需要哪些材料，findings 全部保持 untested。
   要讲清系统结构或领域规律时，仍可读 inspect_grasp_graph 与 read_domain_knowledge。
   有证据时，先按相关领域读取 read_domain_knowledge，必要时用 inspect_grasp_graph
   核对交接与共同依赖。
   知识中的公式有适用前提；具体单位、坐标、时刻、参数与阈值必须从项目资料确认。
5. 几何问题的定位与排除以 structural_diagnose 的核算为准。
   check_transform_chain 只解释已定位异常的机理：declared_matches 里残差为零（仅差数值舍入）
   的那一行，说明误差等于哪个声明变换及其用法（一次、两次或其逆）；残差不为零的行不能作为机理。
   error_by_frame 说明误差锚定在哪个坐标系。零残差只说明一种关系吻合，不证明约定正确。
   引用具体输出字段和数值，不心算；只检查 A 时不能声称 B。
6. 诊断分两层。软件层看运行起来的程序：运行版本、实际加载的配置、模块输入输出、日志，
   以及记录版本在记录输入上的复跑。定位在软件层完成：suspect 和 cleared 至少引用一条
   非源码证据，只引用源码的判断会被宿主降为 untested。对承载软件的节点（采集、感知、
   抓取策略、指令计算，以及流入它们的交接），用运行记录说明哪个输入、输出、配置或版本
   偏离了约定，或输入超出了模块声明的适用条件。有参考运行时用 compare_runs 按内容对比；
   程序可复跑时用 replay_running_version 确认记录输入能否复现记录输出。
   源码层由宿主开放：本案有源码、源码版本与运行版本一致，且已提交基于软件层证据的 suspect。
   开放后源码只用来解释已定位对象的机制并生成补丁，读取时要给出对应的 hypothesis_id。
   源码里的注释、命名和字符串不是事实。不要为了打开源码层把证据不足的对象判为 suspect。
7. 每个假设写明 remedy：configuration（改运行配置）、handoff（把定位的模块、违反的约定和
   复现输入交开发方或厂商）、source_patch（源码补丁，只有这类假设能提交 propose_repair）、
   physical（现场物理处理）、more_evidence（补充取证）。没有源码时，软件层定位加 handoff
   就是完整结论，不要猜测实现细节，也不要因此降低定位的确定程度。
   复跑结果只描述代码在记录输入上的表现；到位不等于夹持/抬起成功。
   外部调整交人工处理，并说明期望回证。现场恢复需要改动应用之后的新观察。
8. 按证据 phase 与时间语义检查覆盖的任务阶段。before_command 画面只能说明命令前情况，
   不可用于判断随后 PICK/RETRACT 是否夹住。未知阶段不补猜。每次读取图像都应针对
   尚未解决的鉴别问题；不要重复读同阶段相似帧来替代缺失的动作阶段记录。
   固定 pick_offset、工装姿态和工具配置也是实际抓取策略的一部分，应先解释其作用。
9. 有适用的诊断模板时，先用 structural_diagnose 核算再下判定。你负责把本案记录绑定到模板的测量
   （测量编号 → evidence_id，按工件分别调用）；宿主做结构分析（哪些故障可检测、可隔离、
   哪些检验可用）、运行预先实现的检验，并给出剩余候选、核算排除和未检验。核算结果是派生证据，
   可以引用。引用核算时不能说反话：不能把核算隔离出的唯一候选判为 cleared，
   也不能把核算排除的节点判为 suspect，宿主会退回。核算排除的节点也不能作为打开源码层的定位。
   未检验不等于正常。
   剩余候选不止一个时，优先按返回的 suggestions 绑定所需测量再核算；gaps.missing_measurement
   表示用现有记录在结构上分不开，应请求新的测量来源，不要凭推测排除。数值结论引用检验输出，不心算。

只返回一个 JSON 对象：
{
  "findings": [{"target_id": "图中节点或边的 id", "status": "cleared|suspect|untested",
                "note": "判断依据", "evidence_ids": ["EV-001"],
                "checked_scope": "实际检查范围", "limitations": "仍未排除什么"}],
  "hypotheses": [{"hypothesis_id": "H1", "target_id": "图中节点或边的 id",
                  "statement": "候选解释", "evidence_ids": ["EV-001"],
                  "prediction": "成立时应看到什么", "counter_evidence_ids": [],
                  "next_check": "能区分它与其他解释的检查",
                  "remedy": "configuration|handoff|source_patch|physical|more_evidence"}],
  "next_step": "简体中文，说明下一步该做什么"
}
"""

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "inspect_grasp_graph",
            "description": "查看抓取图中一个节点或交接的依赖、正常约定与检查方向，不产生观测证据。",
            "parameters": {
                "type": "object",
                "properties": {"target_id": {"type": "string"}},
                "required": ["target_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_domain_knowledge",
            "description": (
                "读取 1 至 3 个相关领域的正常规律、竞争解释、鉴别检查与局限。知识不是现场证据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_ids": {
                        "type": "array", "items": {"type": "string"},
                        "minItems": 1, "maxItems": 3,
                    },
                },
                "required": ["knowledge_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_evidence",
            "description": (
                "读取证据，一次最多 8 件。文本与 JSON 直接返回内容；"
                "图片返回逐图观察，可用 question 指定配方、ROI 和需核对的视觉问题；"
                "深度图返回统计摘要。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                    "question": {"type": "string", "description": "待核对的视觉问题与上下文"},
                },
                "required": ["evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_rgbd_region",
            "description": (
                "检查指定 RGB 目标矩形的亮度/锐度和配准深度有效比例，并与全图对照。"
                "roi_xyxy 为左上/右下像素坐标，右下不包含。相机元数据明确时才做米制表面测量"
                "和检测中心投影。矩形不是分割，表面不等于物体中心，不自动判定故障。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rgb_evidence_id": {"type": "string"},
                    "depth_evidence_id": {"type": "string"},
                    "camera_evidence_id": {"type": "string"},
                    "roi_xyxy": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                    },
                    "detection_evidence_id": {"type": "string"},
                },
                "required": ["rgb_evidence_id", "depth_evidence_id", "camera_evidence_id",
                             "roi_xyxy"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_capture_alignment",
            "description": (
                "比较采集记录、感知记录与可选指令记录的采集/检测/工件标识，检查 RGB-D 时差、"
                "感知消费的源帧时间。缺失字段返回未知，不对不同时钟域做减法。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capture_evidence_id": {"type": "string"},
                    "detection_evidence_id": {"type": "string"},
                    "command_evidence_id": {"type": "string"},
                },
                "required": ["capture_evidence_id", "detection_evidence_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_transform_chain",
            "description": (
                "机理测量：用你指定的证据重算声明的变换链，给出误差与每个声明变换（一次、两次、逆）"
                "的差距和误差在各坐标系下的表示。用于解释已定位的几何异常，不用于定位或排除。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "detection_evidence_id": {"type": "string"},
                    "calibration_evidence_id": {"type": "string"},
                    "tool_evidence_id": {"type": "string"},
                    "command_evidence_id": {"type": "string"},
                    "motion_evidence_id": {"type": "string"},
                },
                "required": [
                    "detection_evidence_id",
                    "calibration_evidence_id",
                    "tool_evidence_id",
                    "command_evidence_id",
                ],
                "additionalProperties": False,
            },
        },
    },
)


STRUCTURAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "structural_diagnose",
        "description": (
            "结构诊断：把一个工件的记录绑定到诊断模板的测量上。宿主据模板的方程结构判断哪些故障"
            "可检测、可隔离、哪些预先实现的检验可用，运行这些检验，按单故障假设给出剩余候选、"
            "核算排除、未检验，以及能区分剩余候选的下一项测量。同一运行和工件的绑定会累积。"
            "模板与测量见案件状态里的 diagnostic_templates。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "template_ids": {
                    "type": "array", "items": {"type": "string"}, "minItems": 1,
                    "description": "如 geometry_chain、frame_correspondence",
                },
                "part_id": {"type": "string", "description": "工件，如 A"},
                "bindings": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "测量编号 → evidence_id，如 {\"command\": \"EV-003\"}",
                },
            },
            "required": ["template_ids", "part_id", "bindings"],
            "additionalProperties": False,
        },
    },
}

COMPARE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compare_runs",
        "description": (
            "软件层：按内容对比本案两次观察的运行版本和软件层记录（程序输入输出、运行配置）。"
            "标识和时间字段不参与比较；逐字节不同但内容相同的配置会如实标出。"
            "差异只说明哪里变了，不说明哪一边正确或为什么变。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "baseline_run_id": {"type": "string", "description": "参考运行，如正常运行"},
                "run_id": {"type": "string", "description": "待诊断的运行"},
            },
            "required": ["baseline_run_id", "run_id"],
            "additionalProperties": False,
        },
    },
}

REPLAY_RUNNING_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "replay_running_version",
        "description": (
            "软件层：在隔离环境里用记录的程序输入复跑运行版本，逐项对比记录的输出。"
            "不读取源码。复现说明结果由该版本和这些输入决定，不说明原因在实现的哪里。"
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}

SOURCE_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "list_source",
            "description": "源码层：列出运行版本的源码文件名。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "源码层：为一个指向已定位对象的已提交假设，读取运行版本的一个源码文件。"
                "源码只解释机制，注释和命名不是事实。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "hypothesis_id": {"type": "string"},
                },
                "required": ["path", "hypothesis_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_repair",
            "description": (
                "源码层：为一个 remedy 为 source_patch 的已提交假设提交补丁，给出文件路径和"
                "修改后的完整内容。它在隔离工作树里用本次记录输入复跑并与记录输出对比，"
                "绑定的仓库不会被改动。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "path": {"type": "string"},
                    "new_text": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["hypothesis_id", "path", "new_text", "rationale"],
                "additionalProperties": False,
            },
        },
    },
)


def tools_for(case: Case) -> tuple[dict[str, Any], ...]:
    """The software layer is always offered; the source layer only once the host opens it."""

    tools = TOOLS + (STRUCTURAL_TOOL,)
    if len(case.observations) >= 2:
        tools += (COMPARE_TOOL,)
    if case.project is not None and case.project.runnable:
        tools += (REPLAY_RUNNING_TOOL,)
    if source_layer_gate(case).passed:
        tools += SOURCE_TOOLS
    return tools


def access_note(case: Case) -> str:
    """What this case can reach, said plainly so the model does not assume more."""

    verdict = source_layer_gate(case)
    project = case.project
    if verdict.passed:
        located = "、".join(str(item.target_id) for item in software_localizations(case))
        return (
            f"源码层已开放，依据软件层定位：{located}。源码只解释这些对象的机制并生成补丁，"
            "读取时给出对应 hypothesis_id；新的可疑或排除仍须引用运行记录或观测。"
        )
    if project is None:
        return (
            "本案只有运行记录与现场观测，没有可复跑的程序，也没有源码。"
            "软件层定位就是终点，出口为 handoff、configuration、physical 或 more_evidence。"
        )
    if not project.source_readable:
        return (
            "本案可在记录版本上复跑程序（replay_running_version），但没有源码，相当于只有发布产物。"
            "软件层定位就是终点，出口为 handoff、configuration、physical 或 more_evidence。"
        )
    return (
        "本案有源码接入，但源码层尚未开放：" + "；".join(verdict.reasons) + "。"
        "先用运行记录在软件层定位并提交结论；满足条件后，下一轮才提供源码工具。"
    )


def build_view(case: Case, bundle: ObservationBundle | None) -> dict[str, Any]:
    """Case state, the observation's own account of itself, and the catalogue."""

    verdict = source_layer_gate(case)
    standing = {item.segment: item for item in case.findings}
    return {
        "case": {
            "case_id": case.case_id,
            "title": case.title,
            "access": {
                "runnable": bool(case.project and case.project.runnable),
                "source_readable": bool(case.project and case.project.source_readable),
            },
            "source_layer": {"open": verdict.passed, "missing": list(verdict.reasons)},
            "access_note": access_note(case),
        },
        "graph": graph_view(
            case.findings, {item.evidence_id: item.layer for item in case.evidence}
        ),
        "knowledge_catalogue": catalogue(),
        "diagnostic_templates": template_catalogue(),
        "isolation": [
            {key: row[key] for key in (
                "evidence_id", "run_id", "part_id", "diagnosed_run", "templates", "bindings",
                "isolation", "gaps", "suggestions",
            )}
            for row in isolation_view(case)
        ],
        "observations": [
            {"run_id": observed.run_id, "created_at": observed.created_at.isoformat(),
             "project_revision": observed.project_revision}
            for observed in case.observations
        ],
        "observation": (
            {
                "run_id": bundle.run_id,
                "collected_at": bundle.created_at.isoformat(),
                "project_revision": bundle.project_revision,
                "task_results": [result.model_dump(mode="json") for result in bundle.results],
                "timeline": [event.model_dump(mode="json") for event in bundle.timeline],
                "cross_source_comparable": bundle.cross_source_comparable,
            }
            if bundle is not None
            else None
        ),
        "chain": [
            {
                "segment": segment.value,
                "status": status.value,
                "scope": SEGMENT_SCOPE[segment],
                **({"note": standing[segment].note} if segment in standing else {}),
                **(
                    {"evidence_ids": list(standing[segment].evidence_ids)}
                    if segment in standing
                    else {}
                ),
            }
            for segment, status in case.demarcation().items()
        ],
        "hypotheses": [
            item.model_dump(mode="json")
            for item in case.hypotheses
        ],
        "repair_plans": [
            {
                "plan_id": plan.plan_id,
                "target_segment": plan.target_segment.value,
                "frozen_hash": plan.frozen_hash,
            }
            for plan in case.repair_plans
        ],
        "evidence_catalogue": [
            {
                "evidence_id": item.evidence_id,
                "bundle_id": item.bundle_id,
                "reference": item.reference,
                "media_type": item.media_type,
                "captured_at": item.captured_at.isoformat(),
                "clock_domain": item.clock_domain,
                "phase": item.phase,
                "layer": item.layer,
                #: Already delivered by the host, so it can be cited without
                #: being asked for again.
                "examined": item.evidence_id in case.examined,
            }
            for item in case.evidence
        ],
    }
