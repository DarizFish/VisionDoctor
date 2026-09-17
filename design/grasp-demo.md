# 抓取诊断比赛演示指引

本指引配合 [比赛方案](grasp-competition.md) 与 [实验记录](results/grasp-round3-20260912.md)。
当前工位的成功标准是机器人到位，未实现工件被夹住、抬起与保持的物理验证。

## 讲述顺序

1. 展示完整抓取依赖图：同一个“没抓到”可能来自不同环节，知识覆盖完整任务。
   接着用结构诊断对比几何故障与旧帧消费：两者都是 A、B 没到位，核算面板里检验特征不同，隔离到不同节点。
   展开"结构分析"，用可隔离性矩阵说明哪些原因凭现有记录分不开；
   旧帧只核算几何链时出现的区分建议来自确定性核对记录，不是模型行为（见下表）。
2. 主案例展示几何软件故障的两层诊断：先看不读源码的案件，软件层按内容对比正常运行，定位到指令计算并给出交接说明；再看含源码的案件，同一定位打开源码层，模型解释机制、生成最小补丁、隔离复跑，批准落地后用新的 Gazebo 运行复核。
3. 用旧帧消费展示非坐标原因：当前图像存在，但实际检测使用较早的图像；单纯重算几何不能解决。
4. 用原始/消费深度对照展示多模态证据：RGB 能看到工件，原始深度有值，程序消费区域却缺失。
5. 展示遮挡误诊、实际侧视补证后的局部改善和证据不足对照。侧视后识别了目标存在但主视角不可见，
   具体白板遮挡仍未准确辨别。不要隐藏失败或把补证要求包装成根因确认。

模型诊断在当前实验中通常需要数分钟，现场时间有限时展示保存的真实工具记录，并明确这是历史运行。
可现场执行一次工位节拍；不要把历史模型输出说成刚刚实时生成。

## 可核实材料

路径相对于仓库根目录，完整诊断目录含 `summary.json`、`case-view.json`、`cases/CASE-001.json`。
最后一项保留原始模型对话、工具参数和返回值，适合核对图上的结论。

| 内容 | 材料 |
|---|---|
| 软件故障输入 | `.runtime/gazebo-pick-cell/exports/run-20260912T082922Z-f603527a` |
| 模型诊断、候选与审批记录 | `.runtime/grasp-evaluation/round3-repair-20260912` |
| 修复后的新观察 | `.runtime/gazebo-pick-cell/exports/run-20260912T090708Z-293764aa` |
| 修复复核记录 | 上述诊断目录的 `candidate-review.json`、`site-recheck.json` |
| 时序故障与诊断 | `exports/run-20260912T085057Z-f4e53f0e`、`.runtime/grasp-evaluation/round3-timing-retry-20260912` |
| 遮挡原始误诊 | `exports/run-20260912T085331Z-850d5b5d`、`.runtime/grasp-evaluation/round3-physical-20260912` |
| 原始/消费深度对照 | `exports/run-20260912T090840Z-d6384ee9` |
| 证据不足输入 | `exports/run-20260912T085751Z-00942e46` |
| 侧视补证与部分正确诊断 | `exports/run-20260912T094303Z-e93e20ee`、`.runtime/grasp-evaluation/round5-alternate-view-20260912` |
| 两层诊断 · 不读源码 | `.runtime/grasp-evaluation/round6-runnable-geometry-20260916` |
| 两层诊断 · 含源码、补丁与复核 | `.runtime/grasp-evaluation/round6-source-geometry-20260916`、修复副本 `projects/repair-evaluation-20260916`、复核运行 `exports/run-20260916T095525Z-7fd260e0` |
| 两层诊断 · 深度缺失 | `.runtime/grasp-evaluation/round6-source-depth-20260916` |
| 结构诊断 · 确定性核对（宿主工具直接调用） | `.runtime/grasp-evaluation/round7-structural-check-20260916.json`、`...-depth.json` |
| 结构诊断 · 几何故障（真实模型，不读源码） | `.runtime/grasp-evaluation/round7-runnable-geometry-20260916` |
| 结构诊断 · 旧帧消费（真实模型，不读源码） | `.runtime/grasp-evaluation/round7-runnable-stale-20260916` |
| 结构诊断 · 深度缺失（真实模型，含源码） | `.runtime/grasp-evaluation/round7-source-depth-20260916` |
| 结构诊断 · 同输入重复（旧帧、几何） | `.runtime/grasp-evaluation/round7-repeat-stale-20260916`、`round7-repeat-geometry-20260916` |
| 结构诊断 · 复核前后检验对照 | `.runtime/grasp-evaluation/round7-recheck-structure-20260916`（第六轮已批准案件的副本，核算由脚本调用） |

表中的 `exports/` 均位于 `.runtime/gazebo-pick-cell/` 下。
第四轮重复诊断与评分见 [第四轮记录](results/grasp-round4-20260912.md)，
多观察取证与正常对照补证见 [第五轮记录](results/grasp-round5-20260912.md)。
私有 `scenario-labels` 只用于评测，不交给诊断 Agent；输入数据本身允许展示实际异常。

## 当前机器上复现

在仓库根目录使用现有 `.venv`。模型调用要求当前环境已配置文本与视觉模型，禁止把密钥放进演示材料。
工位准备和官方 Gazebo 窗口操作见 [工位说明](../harness/gazebo/README.md)。

```powershell
.\.venv\Scripts\python.exe -m harness.gazebo.cli status
.\.venv\Scripts\python.exe -m harness.gazebo.cli run --workspace .runtime/gazebo-pick-cell/projects/rgbd-grasp-faulty
.\.venv\Scripts\python.exe -m harness.gazebo.cli run --workspace .runtime/gazebo-pick-cell/projects/repair-evaluation-20260912
```

最后一个目录是本次模型候选实际落地后的实验副本，不是正常参考工作区。执行会产生新 run ID。
需要导入诊断时，用 `export --run-id <实际返回的ID> --destination .runtime/gazebo-pick-cell/exports` 导出。

启动一次新的只读诊断，输出使用新的时间目录：

```powershell
$demoStamp = Get-Date -Format yyyyMMdd-HHmmss
.\.venv\Scripts\python.exe -X utf8 -u -m harness.evaluate_grasp `
  --bundle .runtime/gazebo-pick-cell/exports/run-20260912T082922Z-f603527a `
  --reference .runtime/gazebo-pick-cell/exports/run-20260912T082034Z-8d407a4a `
  --repository .runtime/gazebo-pick-cell/projects/rgbd-grasp-faulty `
  --access runnable `
  --prompt '检查这次抓取任务的结果，分析最有证据支持的原因、尚不能区分的解释以及下一步。案件里另附一次较早的正常运行作为参考。先诊断，不执行修复。' `
  --output ".runtime/grasp-evaluation/demo-$demoStamp"
```

`--access runnable` 表示可复跑但不读源码，诊断停在软件层；改为 `--access source` 并加
`--repair --repair-turns 2` 时，软件层定位之后的轮次开放源码层并可提交补丁。
需要重演补丁落地时，先从 `projects/rgbd-grasp-faulty` 克隆新的专用副本并绑定它，不要复用已修复的目录。
2026-09-16 之前导出的包需先执行 `python -m harness.gazebo.declare_layers <导出目录>` 补写证据层。

新场景可通过 `python -m harness.gazebo.scenarios physical_occlusion` 或 `stale_capture` 创建，
两者均需 `--workspace .runtime/gazebo-pick-cell/projects/rgbd-grasp-reference`。
同一个仿真工位顺序执行，场景操作结束后自动恢复；不能同时运行互相移动工件的实验。

侧视补证的实际实验脚本和执行方法见第五轮记录。`supplementary/side_view/observer-key-005.png`
是动作结束后的侧视相机画面；该目录中的 `rgb.png` 仍来自原主 RGB-D 相机，两者不能混为同一视角。
案例用于演示取证价值与剩余不确定性，不把相机移动说成 Agent 自主行动。

## 工作台展示保存的修复案件

下面是供现场操作者在终端前台启动的命令；本轮没有启动后台预览服务。
先启动实际 Case API，再在另一个终端启动工作台：

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import uvicorn; from visiondoctor.case import CaseService; from visiondoctor.api.case_api import create_case_app; uvicorn.run(create_case_app(CaseService(Path('.runtime/grasp-evaluation/round6-source-geometry-20260916/cases'))),host='127.0.0.1',port=8000)"
.\.venv\Scripts\python.exe -m streamlit run src/visiondoctor/web/app.py --server.address 127.0.0.1 --server.port 8501
```

打开 `http://127.0.0.1:8501`。右栏自上而下是：常驻状态条（当前阶段 · 现在成立的结论 · 下一步）、
接入条与源码层状态、可折叠的抓取系统图，然后分成两个页签——**宿主核算**（核算隔离、现场复核、证据清单、当前连接）
与**模型主张**（检查依据、分组概览、当前假设与交接说明、候选 diff 与审批）。
讲"模型不主导事实"时直接切换这两个页签：同一张系统图之上，左边的数算出来，右边的话需要证据支持。
系统图折叠起来可以让两栏并排对照。已有批准和落地的候选用于查看，不重复批准应用。
工作台采用实际保存案件进行 AppTest 验证；2026-09-17 又在浏览器里逐项核对了旧帧、深度、复核三个第七轮案件的
核算面板，页面显示与案件里保存的核算结果一致（当时修掉的两处显示缺陷见第七轮记录的"验证"一节）。
展示核算面板时把窗口拉宽：侧栏较窄时检验表最右的"敏感节点"列需要横向滚动。

## 答辩时的能力边界

- 图是抓取任务参考模型，不是自动发现设备，也不是统计训练出的因果网络。
- 结构诊断的模板是人工整理的方程与故障挂接；检验对哪些故障敏感由结构推出，隔离在单故障假设下成立。
  它只对模板覆盖的节点负责，"未检验""不可检测"都不等于正常。
- 感知仅验证已知颜色、形状与正立工装条件；姿态来自配方，位置来自实际 RGB-D。
- 候选重放消费记录的感知位姿；若修改感知算法，必须补原始图像重放或重新运行现场感知。
- 本次修复证明指定工位、A/B 和运行条件下的到位恢复，不能外推复杂物体或完整物理抓取。
- 多次实验规模很小，部分还在开发过程中改过提示和工具；按案例报告结果，不宣称通用准确率。
  图像身份、物理机制和取证效率仍有不足；侧视后的现场检查建议尚无按建议操作后的独立恢复验证。
- 企业部署、通用设备接入和规模化运维不属于本次比赛的重点。
