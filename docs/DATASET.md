# 地面录像、标注与离线训练

地面采集实现位于 `cnsbridge.experiments.tello.record`；数据准备与训练位于 `cnsbridge.learning.prepare`、`cnsbridge.learning.train`。以下命令通过 `python -m` 直接运行这些模块。

当前先做 [神经活动诊断](NEURAL_ACTIVITY.md)：检查不同刺激下的编码输入和神经放电，不需要填写 `labels.csv`。其中的 `stimuli.csv` 只描述画面刺激，不能代替本文的期望动作标签。

确认信号后，本文的训练实验要回答：冻结连接组产生的神经活动，能否预测未见场景中的人工期望平面动作。这里评估标签预测，不评估飞行距离或避障。

```text
Tello 地面录像与时间戳 → 人工期望动作
         ↓                  ↓
离线 Retina / Brain 重放 → 对齐数据 → 按场景训练与评估
```

## 1. 录制一段试采视频

在项目目录激活环境。关闭手机 App 和其他 Tello 程序，电脑连接 Tello Wi-Fi，飞机固定在地面：

```bash
python -m cnsbridge.experiments.tello.record --scene-id room_a --seconds 30
```

Q 或 Escape 结束预览。录制只连接 SDK、读取视频与遥测，不加载大脑、不发送运动指令。终端会输出 `data/ground_recordings/` 下的新会话目录：

| 文件 | 内容 |
|---|---|
| `video.avi` | 保存的视频 |
| `frames.csv` | 帧序号、播放时间与主机取帧时间 |
| `session.json` | 会话和场景信息 |
| `labels.csv` | 只有表头的待标注文件 |

先保持画面静止至少两秒，再按不同顺序展示以下情形。左右以录像画面为准。

| 场景 | 本轮定义的期望动作 |
|---|---|
| 前方视野通畅 | 缓慢前进 |
| 物体从左侧靠近、右侧空着 | 向右转 |
| 物体从右侧靠近、左侧空着 | 向左转 |
| 正前方靠近的物体遮挡通路 | 停止前进 |

换位置和含糊片段不标注。手机随意左右摆动不能直接定义转向答案。当前视觉输入主要编码运动，不能假设它已经具备静态障碍物识别能力。

## 2. 按实际录像标注

用播放器检查视频，编辑本次会话的 `labels.csv`。使用 **AVI 播放时间**，不能使用墙钟时间。`frames.csv` 同时记录 `video_t_s = frame_index / 20` 与主机取帧时间 `t_s`；采集出现间隙时，两种时间并不相同。

标签区间为左闭右开 `[start_s, end_s)`。以下仅演示格式，必须换成实际录像的时间与动作：

```csv
start_s,end_s,vx_mps,wz_rad_s,reason
3.0,6.0,0.10,0,clear_view
9.0,12.0,0,-0.20,left_approach_right_clear
16.0,19.0,0,0.20,right_approach_left_clear
23.0,26.0,0,0,frontal_obstruction
```

`vx` 用 m/s，`wz` 用 rad/s，正值为左转。这些数值是本轮期望动作的任务定义，不是实测 Tello 速度，也不是已验证的飞行设置。其他四个轴为零。

不同录像使用同一标注规则。未知和未覆盖区间不进入训练，不自动当成停止；重放时仍处理这些画面以保留神经状态的因果历史。不要把旧控制器输出直接当成正确答案。

## 3. 离线提取特征

标注后，无需连接 Tello：

```bash
python -m cnsbridge.learning.prepare data/ground_recordings/SESSION \
  --out data/ground_recordings/SESSION/features.npz
```

将 `SESSION` 替换为实际目录。默认读取 `configs/malecns.yaml`，需要已构建的 `data/malecns_brain.npz`。每段录像重置视觉与神经状态，前两秒播放时间不作为带标签样本；旧运动读出不参与此流程。

默认特征为配置中非空输出群组内的单神经元发放率。`--readout-level group` 可改用群组均值，作为另一个实验；特征模式不同的数据不能混合。只有检查软件流程时才使用 `--brain minifly`，并明确记录这是合成网络结果。

每帧按 50 ms 推进模型，不随计算速度变化；工具报告采集间隙与处理耗时。重放假定相机固定，并输入零偏航角速度，不能将该数据直接等同于飞行中的视觉分布。

## 4. 按场景训练与评估

先在至少三个不同物理场景中各采集若干段录像。同一布置的多段录像使用相同 `scene_id`，不能靠改名字将近乎相同的场景分入训练与测试。先用两个场景训练、一个完整场景评估，作为流程试验；这一小规模划分不足以支持广泛泛化结论。

以下目录代表已经录制、标注并提取特征的会话：

```bash
python -m cnsbridge.learning.train \
  data/ground_recordings/RUN_A/features.npz \
  data/ground_recordings/RUN_B/features.npz \
  data/ground_recordings/RUN_C/features.npz \
  --test-scenes room_c --out-dir data/decoder_experiment_01
```

输出目录包含：

- `candidate.json`：神经活动读出的候选模型。
- `report.json`：留出场景上的误差、动作覆盖和分会话结果。

当前对照包括神经读出、完整视觉特征网格的岭回归、训练目标均值和零动作。重点查看 `vx`、`wz` 误差与转向方向，不要用四个固定为零的轴抬高整体成绩。

标准化只使用训练场景。模型超参数应在看测试结果前确定；不要反复针对同一测试场景调参后仍将其称为未见测试。两个学习基线的特征数可能不同，此对照本身不能隔离生物连接结构的贡献。

## 5. 结果边界与记录

会话、特征文件和训练输出均要求新路径，避免覆盖。保留原视频、标签、场景划分、特征模式、配置和评估报告，使结果可以重放。

新候选模型尚未连接现有飞行工具。取得地面标签预测成绩后，还需要新场景的在线观察、动作后果与闭环试验。地面录像得分不能代替实飞成功率、避障可靠性或跨设备迁移结果。
