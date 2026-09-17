# PICK-A17 独立 Gazebo 抓取演示

这是比赛演示的操作环境，不是产品功能，也不接入诊断 Agent 或产品 API。它把一段真实的
Gazebo / ROS 2 节拍采集为可人工导出的观察证据包。

**2026-09-12 当前路径：** 感知程序已消费实际采集的 RGB-D，位置不再由控制器预设。
颜色与已知形状用于估计正立工件的位置，姿态来自工装约束。正常运行
`run-20260912T082034Z-8d407a4a` 的 A/B 到位误差为 3.326 mm / 2.005 mm，均通过现有容差。
新几何故障运行 `run-20260912T082922Z-f603527a` 的 A/B 误差为 71.601 mm / 69.053 mm。
目标深度缺失注入运行 `run-20260912T082341Z-7ad218f4` 中，两件均因感知无位姿而没有发出运动命令。
注入改变了被感知程序消费的数据，不表示已物理模拟反光或透明材质。

第三轮已加入实际场景遮挡与旧帧消费实验，入口为 `harness.gazebo.scenarios`。
新深度注入保留 `raw-depth.npy` 与消费的 `depth.npy`，可定位异常发生在传感器输出之后。
模型修复已在专用副本 `repair-evaluation-20260912` 落地；两次新采集的 A/B 到位均通过。
运行、误诊与边界见 [第三轮自审](../../design/results/grasp-round3-20260912.md)；
完整比赛串讲见 [演示指引](../../design/grasp-demo.md)。

**2026-09-16：** 导出清单为每件材料声明 `layer`。程序读写和加载的记录（`parts/*/algorithm/`、`project/`）
为 `software`，相机与运动记录为 `observation`，诊断产品据此区分软件层证据。此前导出的包可用
`python -m harness.gazebo.declare_layers <导出目录>` 按同一规则补写；只改 `bundle.json`，材料与摘要不变。

演示顺序固定为：故障项目执行 A/B 抓取并失败 → 导出证据包 → 在另一个项目工作区修复代码
→ 在同一工位复跑并成功。抓取判定采用**到位判定**：机器人执行到命令的法兰位姿，harness
用私有的工具模型计算 TCP 偏差；没有伪造的物理吸附或搬运。

**2026-09-16 节拍提速（实测）：** 从点击"触发完整 A/B 机器视觉抓取"到机器人开动，由约 13 秒降到约 2 秒；
A 完成到 B 开动由约 10 秒降到约 1 秒；两件完整节拍由约 70 秒降到约 31–36 秒。到位判定不变，
修复副本 A/B 误差 2.10 / 1.93 mm。原因与改动：

- **容器 CPU 配额 4 → 12**：Gazebo 的相机与 GUI 都是 CPU 软件渲染，4 核时几乎每个调度周期都被限流，
  仿真只有约 0.5 倍实时；12 核后约 0.84 倍（仍受软件渲染吞吐限制）。已在运行的容器可用
  `docker update --cpus 12 pick-cell-harness-gazebo` 生效，新启动的工位自动使用 12 核。
- **常驻工位代理** `ros/cell_agent.py`：在容器内保持相机订阅与 MoveIt 连接，采集和运动不再每次新起
  ROS 进程、加载环境和重新发现话题。启动工位或打开操作台时自动拉起。
- **常驻项目程序进程** `project_worker.py`：感知与指令程序仍按 `python -m` 语义运行项目自己的代码，
  只复用解释器和第三方库导入；项目代码变动时自动重启。
- **旁观视频后台写入**：节拍内 RGB-D 落盘即继续，旁观视频在生成证据包前写完。
  旁观视频现在是 RGB-D 帧之前最近的 10 帧（此前为订阅后收到的前 10 帧），两者都在指令之前。
- **操作台显示进度**：点击后立即显示"点击到机器人开动"、总用时和每个阶段的耗时，运动阶段显示当前动作。

**Gazebo 窗口不出现时（2026-09-16 实测）：** 如果 WSL 运行期间接入或调整了显示器（尤其是缩放比例不同的多屏），
WSLg 可能仍在渲染 Gazebo GUI，却不把窗口显示到 Windows 桌面上，缩放 200% 的屏上尤其容易出现。
执行 `wsl --shutdown`，等 Docker Desktop 自动恢复后重新启动工位即可。工位状态里的 `gazebo_window.visible`
现在检查的是 Windows 桌面上是否真的有 "Gazebo Sim" 窗口；窗口没有出现时，启动工位会明确报错。

## 一次性准备

首次初始化时，在仓库根目录执行：

```powershell
py -3 harness/gazebo/build_demo_project.py
```

第一条命令会生成 `harness/gazebo/ur5e_pick_demo.bundle`，其中有一个正常提交和一个故障
HEAD，并在 `.runtime/gazebo-pick-cell/projects/rgbd-grasp-faulty` 建立默认可运行工作区。为避免覆盖可能已
修复的工作区，已有默认工作区时该初始化会拒绝覆盖。运行时生成的
私有评分资料在 `harness/gazebo/private/`，不会写进观察包或 Git bundle。

工位直接复用已验证的 `visiondoctor/ros-gazebo:jazzy-v1` Docker 镜像，不再构建独立 ROS / MoveIt
镜像。隔离由 harness 自有的容器名、只读场景与探针挂载、独立运行目录和证据导出边界保证；若该镜像
不存在，先恢复或拉取该已验证镜像，再启动工位。

## 操作台

```powershell
py -3 -m pip install -e ".[web]"
streamlit run harness/gazebo/console.py --server.address 127.0.0.1 --server.port 8502
```

浏览器打开 <http://127.0.0.1:8502>。选择“启动工位并打开 Gazebo GUI”后，Docker Desktop
会通过 WSLg 打开官方 Gazebo Qt 窗口；浏览器只展示由场景相机实际采集的图片、短视频与结果，
不会替代或伪造三维视图。

也可使用命令行：

```powershell
py -3 -m harness.gazebo.cli bootstrap-project
py -3 -m harness.gazebo.cli start
py -3 -m harness.gazebo.cli capture
py -3 -m harness.gazebo.cli run --workspace .runtime/gazebo-pick-cell/projects/rgbd-grasp-faulty
py -3 -m harness.gazebo.cli stop
```

## 工位与已验证节拍

场景是一张加大台面：UR5e 固定在可见的台面安装座上，两个窄小工件位于基座前方同一已实测可达区域、
互不重叠的低矮定位座中。RGB-D 和观察相机都指向这一工作区。夹具的 TCP 轴朝下，每个 A/B 节拍都经过
“安全位 → 工件正上方的预抓点 → 下压抓取位 → 抬回预抓点 → 安全位”。正常抓取位的指尖与工件顶面保留
约 4 mm 可见间隙，避免把到位判定伪装成物理接触。A/B 目标来自当前 UR5e 的实时 MoveIt IK 和实际往返验证。
探针提供固定的关节种子以重复求解，抓取目标始终来自所选项目工作区输出的法兰位姿，并以实测关节到位
作为完成依据。

**历史路径已验证（2026-09-02，预设检测位姿）：** 官方 Gazebo Qt GUI 在 WSLg 中可见；RGB-D 与观察相机采集正常。
低矮双工位和向下抓取的故障运行 `run-20260902T134241Z-59a05bb6` 中，A/B 都完成“预抓点 → 下压 →
抬起”并被判为 `missed_pick_pose`，位置偏差为 69.086 mm、69.604 mm。故障代码把逆工具补偿重复两次，
所以 TCP 稳定停在工件上方而非压入工装。使用同一 Git bundle 的正常参考提交复跑
`run-20260902T134059Z-786a6b63`，A/B 都为 `within_tolerance`，位置偏差为 0.946 mm、2.927 mm。
故障包已导出：52 个工件都有时钟字段和匹配的 SHA-256，manifest 不含私有判分词。变更 SDF、工具配置或
A/B 目标后，必须重新完成这两次真实 MoveIt 验证。

正常参考工作区用于对照，**不是 Agent 生成的修复**。
本次实际模型修复位于专用实验副本 `repair-evaluation-20260912`，提交为
`5dac9d5f92ab4b42c5a47a00d06e6e3b808b4e8d`；可以在操作台选择它重新运行。
重新演示候选生成时使用新的故障副本，避免对已经修复的目录重复应用历史补丁。

## 边界

- `harness/gazebo` 不导入产品代码，也不访问 `/api/v1/*`。
- 场景中的 RGB-D、观察相机、程序输入/输出、日志、轨迹和配置版本可被导出。
- 每个 `artifacts[]` 条目都带有 `captured_at` 与 `clock_domain`：相机帧和机器人轨迹优先使用
  `ros_sim_time_s`，其余工件使用 `utc_file_mtime`。因此可用同一仿真时钟对齐相机与轨迹，再用 UTC
  标记程序和导出文件的生成时刻。
- 导出时仅复核 manifest 已声明的 SHA-256；任何不一致都会停止导出。
- 私有评分目标、容差、故障答案和正常补丁不被列入 `bundle.json`，也不被复制到导出目录。
  传感器故障注入标签保存在运行目录之外的 `scenario-labels/`，不导出给产品。
- 操作台只调用本目录的本地控制器；选择的项目工作区可以是人工或后续 Agent 修复后的副本。

## 人工验收

自动化只校验包的脱敏、正常/故障分类、控制器状态和隔离边界。比赛前仍需实际完成：

1. 打开操作台后，确认官方 Gazebo Qt 窗口可见；
2. 用默认故障工作区运行 A/B，确认两组偏差、相机工件、日志、轨迹和 bundle 已生成；
3. 在修复工作区复跑，确认两组都进入声明容差；
4. 检查导出的 `bundle.json` 与文件清单不含私有评分资料。
