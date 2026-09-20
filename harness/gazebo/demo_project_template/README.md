# PICK-A17 视觉抓取项目

该项目包含 RGB-D 感知、工具补偿和重放入口。它不知道 Gazebo、诊断系统或评分目标。

`pick_demo.perception` 从本次 `rgb.png`、`depth.npy` 和相机记录中估计位置；
颜色、工件高度与工装姿态由 `config/perception.yaml` 声明。适用范围是颜色可分、
已知形状、正立且无遮挡的工件。姿态来自工装约束，不是图像估计的通用六维位姿。
它保存目标框、有效深度比例、输入摘要哈希、采集标识与源帧时间；感知失败不返回虚构位姿。

```bash
python -m pick_demo.replay --input example-input.json --output output.json --log app.jsonl
```

重放输入是感知输出的工件相机位姿；输出是机器人 `tool0`（法兰）命令。部署时 TCP 不是
`tool0`，而是由 `config/tool_profile.yaml` 声明的工具偏移。`config/cell_calibration.yaml`
记录相机与机器人基坐标系之间的静态标定。

当前诊断产品的候选复跑针对这份位姿输入，因此只能验证命令阶段的软件修改；
感知修改要重新运行 `pick_demo.perception` 或完整工位节拍后验证，不能只复用旧位姿。
