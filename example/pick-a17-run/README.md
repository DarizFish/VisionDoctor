# PICK-A17：一次真实运行的观察证据包

这个目录和同级的 `pick-a17-project.bundle` 合起来，让**没有 Gazebo 环境的人也能驱动一次完整诊断**：
接入证据、绑定项目版本、做结构核算、在隔离环境复跑、走到候选补丁与人工审批。

| 项 | 值 |
|---|---|
| 运行 | `run-20260912T082922Z-f603527a` |
| 工位 | PICK-A17 视觉引导抓取，UR5e，工件 A/B |
| 运行版本 | `c13954fc031e16f23d2a21de0b7856f169c80109` |
| 现象 | A、B 两件都没到位，位置偏差约 7 厘米 |

## 内容

```text
bundle.json              清单：每件带 sha256、采集时刻、时钟域、证据层
run-summary.json         这次运行的工位判定
parts/A|B/
  capture/rgb.png        RGB 图像
  capture/camera-info.json  相机内参与坐标系
  capture/capture.json   采集记录（帧标识与时刻）
  algorithm/input.json   感知输出，进入目标计算的输入
  algorithm/output.json  算法给出的法兰指令
  algorithm/application.jsonl  程序运行日志
  motion.json            运动反馈
  trajectory.json        轨迹
  result.json            到位结果
project/
  cell_calibration.yaml  相机外参
  tool_profile.yaml      工具补偿
  perception.yaml        工件配方
  revision.json          这次运行用的代码版本
```

深度图、观察相机视频与逐帧图像因体积未纳入，清单只列出保留下来的件，
哈希与采集时刻都是原值。因此 RGB-D 区域测量这一条通路需要完整导出包，
几何指令链的四项检验与隔离在本目录上可以完整跑通。

## 还原被诊断的项目

```bash
git clone example/pick-a17-project.bundle pick-a17-project
cd pick-a17-project
git checkout c13954fc031e16f23d2a21de0b7856f169c80109
```

这个仓库有两个提交：`04312a0` 是正常版本，`c13954f` 是这次运行实际跑的版本。
重放入口是 `pick_demo.replay`，接受 `--input`、`--output`、`--log` 三个参数。

## 驱动一次诊断

启动 Case API 与工作台后，在工作台里依次：

1. 新建案件，描述现象；
2. 接入观察证据，目录填本目录的绝对路径；
3. 连接项目，仓库填还原出来的 `pick-a17-project`，运行版本填 `c13954fc…`，
   重放命令填 `python -m pick_demo.replay --input {input} --output {output} --log {log}`；
4. 与 Agent 对话推进诊断。

## 这次运行的核算结果

绑定 A 件的感知输出、法兰指令、运动反馈、到位结果与项目的标定、工具配置后，
几何指令链模板的结构核算给出：

| 检验 | 结果 | 残差 |
|---|---|---|
| A 指令一致 | 违反 | 70.0 mm |
| B 运动跟踪 | 通过 | 0.9 mm |
| C 到位一致 | 通过 | 0.7 mm |

隔离结论为 `isolated`，剩余候选只有接口（指令计算与接口），七个节点被核算排除。
