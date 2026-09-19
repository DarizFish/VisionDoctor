# VisionDoctor —— 证据驱动的视觉引导抓取诊断

VisionDoctor 面向视觉引导抓取工位。当前产品主线以一次 Case 为生命周期根：接入观察证据和项目版本，围绕系统依赖图提出竞争假设，用宿主确定性核算与实际运行记录定位异常；只有软件层已经定位且源码版本与运行版本一致时，才开放源码解释和候选补丁。

## 当前范围

当前场景固定为视觉引导抓取，覆盖工件/来料、成像、RGB-D 采集与时序、感知与识别定位、几何关系、目标计算、抓取规划、机器人实际到位，以及版本化源码、隔离复跑、候选修复、人工审批和改动后的重新验证。

当前 Demo 的成功判定止于机器人到达声明的抓取位姿。接触、夹持、吸附和抬起保持等物理抓取结果不属于当前已验证产品能力。Gazebo、故障注入和判定真值属于 `harness/`，不属于诊断产品。

## 当前架构

```text
Case
├─ ObservationBundle / Evidence
├─ system graph
│  ├─ nodes / edges
│  ├─ domain groups (Segment)
│  └─ competing hypotheses
├─ investigation
│  ├─ evidence / graph / domain knowledge
│  ├─ measurements and run comparison
│  ├─ structural_diagnose
│  └─ residual / mechanism checks
├─ software-layer localization
├─ source-layer gate
├─ isolated replay / repair proposal
├─ human approval
└─ post-change recheck
```

系统图是共享参考模型，不是固定排查顺序。Agent 根据症状、证据覆盖范围和竞争假设选择下一项检查；宿主把结论绑定到图节点或边，并验证证据引用、结构核算和门控条件。

`src/visiondoctor/case/chain.py` 定义 `Segment`、`SegmentFinding` 和 `Hypothesis` 等领域分组模型。guided-motion Case 包含工件、成像、采集、算法、任务结果、标定、接口、机器人、规划、抓取共 10 个诊断域，它们是并列的检查范围，不是必须依次走完的顺序。

结构诊断由 `case/templates.py`、`investigation/structure.py` 和 `investigation/residuals.py` 提供。几何类问题由 `structural_diagnose` 负责定位；`check_transform_chain` 等工具只在已经定位之后解释残差机理，不替代定位本身。

## 两层诊断

软件层只依据实际运行产生或消费的材料：运行版本、加载配置、模块输入输出、日志、轨迹、记录输入上的复跑结果，以及宿主计算出的派生证据。一个 `suspect` 或 `cleared` 结论必须引用真正交付给 Agent 的非源码证据。

源码层只有在以下条件同时满足时才开放：

- 本案连接了可读源码；
- 源码版本与观察包记录的运行版本一致；
- 软件层已经用运行证据定位到承载软件的图目标。

源码只用于解释已定位问题的实现机制和形成候选修复。源码注释、变量名或字符串本身不能定位或排除运行故障。

## 快速开始

需要 Python 3.11+。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,api,web]"
```

模型配置参考 `.env.example`。启动当前 Case API：

```powershell
visiondoctor serve --host 127.0.0.1 --port 8000
```

另一个终端启动工作台：

```powershell
visiondoctor-web
```

默认工作台访问 `http://127.0.0.1:8000` 的 Case API，并在 `http://127.0.0.1:8501` 提供 Streamlit 界面。

## Case API

当前服务只暴露 Case 主线：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 服务状态 |
| GET / POST | `/api/v1/cases` | 列出 / 创建诊断案件 |
| GET | `/api/v1/cases/{case_id}` | 读取案件完整视图 |
| POST | `/api/v1/cases/{case_id}/observations` | 接入观察证据包 |
| POST | `/api/v1/cases/{case_id}/attachments` | 提交图片、日志等附件 |
| GET | `/api/v1/cases/{case_id}/evidence/{evidence_id}` | 读取案件证据 |
| POST | `/api/v1/cases/{case_id}/project` | 绑定项目及运行版本 |
| POST | `/api/v1/cases/{case_id}/turns` | 推进一轮诊断 |
| POST | `/api/v1/cases/{case_id}/approvals` | 批准或退回候选修复 |
| POST | `/api/v1/cases/{case_id}/applied` | 记录批准方案已应用 |
| POST | `/api/v1/cases/{case_id}/recheck` | 用应用后的新证据复核 |

## 一次诊断怎样推进

1. 新建 Case，描述现象和成功标准。
2. 接入观察证据。证据进入账本不等于 Agent 已经读过；只有工具实际交付的 `evidence_id` 才能支撑结论。
3. 连接项目仓库和对应运行版本。
4. Agent 在系统图上提出竞争假设，选择有区分度的测量、结构检验、时序检查或运行对比。
5. 宿主记录图目标的 `untested / cleared / suspect`，并保存检查范围、局限和证据引用。
6. 软件目标被运行证据定位后，源码层门才可能打开。
7. 对 `source_patch` 假设生成候选修改并在隔离环境复跑；候选不会自动应用。
8. 人工批准后才允许把冻结候选落到项目仓库。
9. 现场恢复必须由改动之后新采集的观察证据重新核算。

## 可信边界

- `cleared / suspect` 必须绑定本案已交付证据；
- 源码不能独自定位或排除运行故障；
- 一次局部检查不能自动扩展为整条分支正常；
- 缺少材料表示证据不足，不表示现场没有发生该事件；
- 修复前后、不同运行中的同名文件必须按 bundle / 时间语义区分；
- 候选补丁不会自动合并或部署；
- 现场恢复只能由应用后新证据授予。

## Gazebo 演示环境

Gazebo 工位位于 `harness/gazebo`，与产品 API 隔离。它负责真实 RGB-D 采集、MoveIt 运动、故障注入、观察包导出，以及不向产品公开的判定真值。

```powershell
py -3 harness/gazebo/build_demo_project.py
streamlit run harness/gazebo/console.py --server.address 127.0.0.1 --server.port 8502
```

也可使用独立 harness CLI：

```powershell
py -3 -m harness.gazebo.cli bootstrap-project
py -3 -m harness.gazebo.cli start
py -3 -m harness.gazebo.cli capture
py -3 -m harness.gazebo.cli run --workspace .runtime/gazebo-pick-cell/projects/rgbd-grasp-faulty
py -3 -m harness.gazebo.cli stop
```

完整环境说明见 `harness/gazebo/README.md`。

## 仓库结构

```text
src/visiondoctor/
  api/case_api.py       HTTP 产品入口
  case/                 Case 状态、领域分组、硬门、持久化与服务
  environment/          观察证据包边界
  investigation/        调查、测量、结构诊断、残差与工具
  knowledge/            抓取领域知识
  repair/               项目绑定、隔离复跑、落地与复核
  sandbox/              Git worktree 与执行隔离
  adapters/             Gazebo 契约与可视化适配
  geometry/             刚体变换与位姿误差
  vision/               RGB-D 标记位姿测量
  llm/ multimodal.py    模型网关、工具协议与图像观察
  projects/ schemas/    项目与证据的数据模型
  web/                  Streamlit 工作台
  cli.py                命令行入口
harness/gazebo/         独立 Gazebo / ROS 2 演示与故障注入
example/                可还原的故障示例工程（Git bundle）
```

## 验证

常规代码改动至少运行：

```powershell
pytest
ruff check .
```

真实模型诊断和真实 Gazebo 抓取必须实际运行后才能计为验证；方案设计、逐轮实验记录与演示留痕作为文档单独交付，不在本仓库分发。
