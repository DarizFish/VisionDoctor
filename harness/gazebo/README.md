# PICK-A17 独立 Gazebo 抓取演示

这是比赛演示的操作环境，不是产品功能，也不接入诊断 Agent 或产品 API。它把一段真实的
Gazebo / ROS 2 节拍采集为可人工导出的观察证据包。

演示顺序固定为：故障项目执行 A/B 抓取并失败 → 导出证据包 → 在另一个项目工作区修复代码
→ 在同一工位复跑并成功。抓取判定采用**到位判定**：机器人执行到命令的法兰位姿，harness
用私有的工具模型计算 TCP 偏差；没有伪造的物理吸附或搬运。

## 一次性准备

首次初始化时，在仓库根目录执行：

```powershell
py -3 harness/gazebo/build_demo_project.py
```

第一条命令会生成 `harness/gazebo/ur5e_pick_demo.bundle`，其中有一个正常提交和一个故障
HEAD，并在 `.runtime/gazebo-pick-cell/projects/tabletop-faulty` 建立默认可运行工作区。为避免覆盖可能已
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
py -3 -m harness.gazebo.cli run --workspace .runtime/gazebo-pick-cell/projects/tabletop-faulty
py -3 -m harness.gazebo.cli stop
```

## 工位与已验证节拍

场景是一张大台面：UR5e 固定在台面上，两个工件位于机械臂前方且以不同姿态呈现；RGB-D 和观察相机
都指向这一工作区。A/B 的法兰目标及各自的 IK 种子已在当前 UR5e、MoveIt 和该 SDF 布局上实际验证。
种子只用于选择等价 IK 解中的同一可执行关节分支，抓取目标始终来自所选项目工作区输出的法兰位姿。

**已验证（2026-09-02）：** 复用镜像中的官方 Gazebo Qt GUI、相机采集、故障 A/B 往返、修复 A/B
往返和故障 bundle 导出均已实际运行。故障工作区两次均为 `missed_pick_pose`；切换到正常提交后两次
均为 `within_tolerance`。变更 SDF、工具配置或 A/B 目标后，需要重新做同样的真实 MoveIt 验证。

## 边界

- `harness/gazebo` 不导入产品代码，也不访问 `/api/v1/*`。
- 场景中的 RGB-D、观察相机、程序输入/输出、日志、轨迹和配置版本可被导出。
- 每个 `artifacts[]` 条目都带有 `captured_at` 与 `clock_domain`：相机帧和机器人轨迹优先使用
  `ros_sim_time_s`，其余工件使用 `utc_file_mtime`。因此可用同一仿真时钟对齐相机与轨迹，再用 UTC
  标记程序和导出文件的生成时刻。
- 导出时仅复核 manifest 已声明的 SHA-256；任何不一致都会停止导出。
- 私有评分目标、容差、故障答案和正常补丁不被列入 `bundle.json`，也不被复制到导出目录。
- 操作台只调用本目录的本地控制器；选择的项目工作区可以是人工或后续 Agent 修复后的副本。

## 人工验收

自动化只校验包的脱敏、正常/故障分类、控制器状态和隔离边界。比赛前仍需实际完成：

1. 打开操作台后，确认官方 Gazebo Qt 窗口可见；
2. 用默认故障工作区运行 A/B，确认两组偏差、相机工件、日志、轨迹和 bundle 已生成；
3. 在修复工作区复跑，确认两组都进入声明容差；
4. 检查导出的 `bundle.json` 与文件清单不含私有评分资料。
