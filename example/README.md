# 示例工程：Robot Cell Vision

`robot_cell_vision` 是一个可运行的 RGB-D 机器人视觉定位工程，用作 VisionDoctor 的**故障演示样本**。
它以 `robot_cell_vision.bundle`（Git bundle，约 10 KB）的形式随仓库分发。

## 为什么是 bundle 而不是普通目录

这个示例的价值在于它的 **Git 历史**：VisionDoctor 需要对比"上一个正常版本"和"当前故障版本"
两个真实提交。把它作为普通文件提交进父仓库会丢掉历史；作为嵌套 Git 仓库提交则会变成一个
空的 gitlink，克隆后得到空文件夹。Git bundle 用单个文件完整保存历史，一条命令即可还原。

## 还原

```bash
git clone example/robot_cell_vision.bundle robot_cell_vision
cd robot_cell_vision
git log --oneline --all
```

还原后包含两个分支：

| 分支 | 提交 | 内容 |
|---|---|---|
| `main` | `a1ab4c4` → `6da5f58` | 正常版本 → **故障版本（HEAD）** |
| `visiondoctor/INC-5dfd5b4b0c/candidate-01-c5b8c7b7` | `5582338` | VisionDoctor 生成并经人工批准的修复 |

## 工程做什么

深度相机给出目标物体在**相机坐标系**下的位姿，程序结合预标定的相机外参，把它换算到
**机器人基坐标系**，再通过 ROS 2 发布给 UR5e：

```
T_base_object = T_base_camera @ T_camera_object
```

```
src/robot_cell_vision/
  geometry.py       刚体变换工具（4×4 校验、位姿↔矩阵、旋转↔四元数）
  calibration.py    加载并持有相机外参 T_base_camera
  pose_pipeline.py  核心换算 locate_object_in_base()
  application.py    单用例处理入口
  ros_node.py       ROS 2 实时桥接节点
config/cell_calibration.yaml   外参与话题映射
runner.py / test_runner.py     VisionDoctor 执行契约
tests/                          项目自带的 4 个单元测试
visiondoctor.yaml               组件与入口声明（不含故障答案）
```

统一使用米、弧度、`xyzw` 四元数与 `T_target_source` 约定。

## 埋入的缺陷

`a1ab4c4` → `6da5f58` 之间只改了一行，提交信息写的是"在访问边界做归一化"：

```diff
  def base_from_camera(self) -> np.ndarray:
-     return self._matrix.copy()
+     return np.linalg.inv(self._matrix)
```

相机外参被反向使用。后果：目标位置偏差 **909.99 毫米**、姿态偏差 **119.98°**，
机械臂会稳定地停在离目标近一米远的地方。

**而项目自带的 4 个单元测试全部通过。** 因为它们全用单位矩阵构造，而 `inv(I) == I`，
正反向混用在单位矩阵下完全看不出来：

```python
def test_identity_calibration_remains_rigid(self):
    actual = CameraExtrinsics.from_matrix(np.eye(4)).base_from_camera()
    np.testing.assert_allclose(actual, np.eye(4))
```

这正是产线上最难查的一类问题：相机成像正常、标记稳定检出、深度有效、测试全绿——
唯独机械臂抓不准。

## VisionDoctor 给出的修复

修复分支把那一行改回来，并补了两个用**真实部署外参**（带 60° 旋转）的回归测试，
注释直接点破原测试为什么失效：

```python
# Deployed extrinsic (config/cell_calibration.yaml): T_base_camera with a 60° rotation.
# Identity-only calibrations hide forward/inverse mix-ups because inv(I) == I.
```

修复后目标位置偏差降至 **1.19 毫米**。

## 披露

该缺陷由本团队按真实工程中常见的重构失误模式**主动构造**，此处明确披露这一事实。
VisionDoctor 侧不掌握任何缺陷答案——没有已知补丁哈希，也没有缺陷家族开关，
诊断与修复完全由模型基于代码证据独立完成。
