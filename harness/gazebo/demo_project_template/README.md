# PICK-A17 视觉抓取项目

该项目只包含视觉定位、工具补偿和重放入口。它不知道 Gazebo、诊断系统或评分目标。

```bash
python -m pick_demo.replay --input example-input.json --output output.json --log app.jsonl
```

输入是视觉系统输出的工件相机位姿；输出是机器人 `tool0`（法兰）命令。部署时 TCP 不是
`tool0`，而是由 `config/tool_profile.yaml` 声明的工具偏移。`config/cell_calibration.yaml`
记录相机与机器人基坐标系之间的静态标定。
