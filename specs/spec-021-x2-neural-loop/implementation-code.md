# Spec021 实施报告 — 协议、冻结脑宿主与 ROS 桥

状态：初版代码冻结记录，定向软件检查通过；此实施者未运行完整脑、ROS、VM、训练或真实动作。最终两轮与独立预算见[后续限定修正](implementation-observation-budget.md)，实际r4见[实测报告](implementation.md)和[最终PASS](acceptance.md)。风险 S2，仅新增闭环通信与调度，未修改 Spec018 运动核心、Spec019 来源校验、Spec020 数值后端、训练代码或官方 SDK。

## 实际改动

- `control/x2_protocol.py`：严格 JSON 与内容封印、单写者原子且禁止覆盖的消息、运行 ID/观测序号/原始观测 hash/执行身份绑定。纯调度器围绕原 `X2MotionAdapter` 工作。独立初始站稳观察器逐回调及 tick 检查两路新反馈，间隙、速度、姿态或模式异常重置并建立双流 barrier。空源观察不称为拥有控制权，神经 FORWARD 前不发非零。
- `learning/x2_session.py`：新实验目录、原共享锁与 0.1 秒资源监督、VM/CPU 停止预检、固定历史 54 文件快照/来源/图/checkpoint 校验、原 GPUPolicy 构造、FrozenInference 及 600 步新场景预热。每轮保存输入、原始三轴 action/features/logits、完整 brain/drive 状态与 hash；场景保持脑状态，参数及完整原始图计数前后相同才封存。无优化器。宿主 SIGTERM 转为可由监督器收尾的异常。
- `experiments/x2_neural.py`：默认仅预览；显式 `--execute --sim-only --output` 才准备一个受监督 worker。输出只能是本项目 data/ 下的新目录。
- `experiments/x2_bridge.py`：默认仅预览；实际 ROS 必须 `--execute --sim-only`、domain 47、localhost 1，并已有匹配 ready。约 50 Hz tick，思考期间零速，最多 3 次 1 秒 FORWARD，严格固定 vx>0.03 门限。vz/wz 记录为未执行轴。任何 sticky core failure、协议错误、已知宿主失败、日志写失败、信号都清空意图；无自动续走。
- 反馈与宿主 monotonic 分别标明时钟域；30 秒时限由各端本地等待/执行计算。响应前检查更新的 odom+MC、停稳、水平漂移<=0.02m 和完整姿态角<=0.03rad，保留原始输入采集时间。
- 最终握手为：桥第 3 轮真实停稳 ACK（只报告停稳）→worker 参数/图验证并退出→parent supervise 最后资源样本及退出验证通过→host_ack→桥最终零发布、ROS 清理、bridge_summary→parent 才报告整体成功。任何 worker_result 本身均不足以让桥成功。桥每次 update 检查 host_abort、result、worker_result 已知失败。
- 约束为 host 240 秒、ROS 120 秒（包括错误停车、末尾握手与清理），每轮思考 30 秒、每次停车 12 秒。失败情况下宿主不再产生允许动作的响应，桥保留既有 TTL 与有界归零过程。代码不启停 VM 或其他服务。

## 软件验证

唯一相关子系统命令：

```sh
.venv/bin/python -m pytest -q tests/test_x2_neural.py
```

首次 38 passed / 1 failed：bridge 顶层导入经既有 `control/__init__.py` 间接导入 NumPy，使默认预览的无重依赖检查失败。只将 bridge 控制依赖移至显式 execute，随后 39 passed。之后补上宿主 SIGTERM 进入现有监督器收尾、对应监督失败 abort/无 host_ack/恢复信号检查，再运行同一文件：**40 passed in 0.15s**。没有重复 Spec018/019/020 已通过检查或运行全项目测试。

40 项覆盖：原子禁止覆盖、封印、旧运行/乱序/重复字段/非有限 JSON；固定门限与三轴输出；初始空源观察及无抢源；新双流样本与瞬态速度重置；竞争源、陈旧决策/错误观测/执行身份、姿态与漂移；脑等待超时、sticky core failure、运动期间宿主资源失败；全部 STOP 不可报告工程成功；三轮假时钟闭环必须等待 parent ACK；已消费决策篡改；默认预览无 ROS/Torch/NumPy；桥侧 Python3.10 语法；失败监督不会产生 host_ack。假时钟/消息均为软件验证，不是完整脑或仿真效果。

## 冻结 SHA256

| 文件 | SHA256 |
|---|---|
| control/x2_protocol.py | 132fc5124ad27570860df5f2249668e0d9877ed20ebe7347f48da466daf70def |
| experiments/x2_bridge.py | fae0f5facabd50d4e49f28f5828e6eedba1210545adcccfb8e224f2cd87e2b41 |
| experiments/x2_neural.py | 87f985d7498fccee544054ad1ef73bbfccf06720c0fc26fa3e3864f01b11891c |
| learning/x2_session.py | 24e724134e7f3daf5bdd0c6da6ca97eb75a096d9d3a982aa388c148998d104c8 |
| tests/test_x2_neural.py | 7807b45345a7f47f11f14adecc236bf4ac8b6eada85533b4e45e7f3bf111e41f |

## 联调入口（尚未运行）

宿主先运行，新目录由 CLI 创建：

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -m malecns_lab.experiments.x2_neural --execute --sim-only --output /Users/yang/project/malecns_lab/data/x2_sim/agibot/data/var/x2-neural-loop-20261023-r1
```

等待该目录 `ready.json` 且 `phase.json` 为 `wait_observation` 后，协调者才启动已有私有 GPU 环境、官方 SD 和 GUI 重置。宿主保留全部证据与资源监督；容器映射为 `/agibot/data/var/x2-neural-loop-20261023-r1`。

容器所需最小实际源文件（保持原包目录，不用自制替代 init）：

- `malecns_lab/__init__.py`
- `malecns_lab/control/{__init__,adapters,body,command,legacy,x2,x2_protocol}.py`
- `malecns_lab/experiments/{__init__,x2,x2_bridge}.py`

将上述文件从当前 src/ 拷入 run 目录 `transport_src/malecns_lab/` 并保存清单。容器已有 NumPy（原 control 包依赖）、ROS 与 AimDK；不需要 Torch 或完整图。

```sh
docker --host unix:///Users/yang/project/malecns_lab/data/x2_sim/gpu_probe/runtime-rebuild/docker.sock exec malecns-x2-gpu-trial bash -lc 'source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && PYTHONPATH=/agibot/data/var/x2-neural-loop-20261023-r1/transport_src:$PYTHONPATH ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2_bridge --execute --sim-only --output /agibot/data/var/x2-neural-loop-20261023-r1'
```

先启动 host；先有 ready 再准备仿真；结束由协调者关闭本项目容器和私有 VM。默认 Docker context、Noria、CPU 模拟器及真实硬件均不由这些脚本修改。

## 限制与未验收项

尚无本 Spec 完整脑与官方 Sim 共驻留、实际因果运动或环境最终关闭证据；此报告不主张联调通过。真实工程验收需要至少两轮、至少一次世界+x方向实际前进>=0.2m、其后来自新实际 odom 的输入像素 hash 改变、每次前进停稳及 final handshake。全部 STOP 或后续输入不变明确不能整体成功。不主张目标跟随策略已经学会、真实相机、10Hz/60Hz 完整脑闭环、导航或真机能力。资源监督是采样边界，不是 OS 内存隔离或硬实时保证。
