# M2 官方预设消息兼容实施

2026-10-23。工程实现，待新的独立验收；未运行 Linux 编译、VM、ROS、MPS、MC、仿真或机器人动作。没有修改官方源、旧 run、训练、M4、桌面或 `control/x2_skills.py`；没有 commit。

## 改动与边界

风险 S3：厂商消息版本隔离、构建产物身份、跨进程一次请求及生命周期。范围限于 `experiments/x2_preset_identity.py`、`x2_preset_compat.py`、`x2_preset_helper.py`、现有单技能 `x2_skills.py` / `x2_skill_host.py`、`deploy/x2/preset-compat.py` 与定向 CPU 检查。

- `preset-compat.py` 分离显式 prepare/build/inspect。prepare 校验官方 ZIP 的固定大小/SHA、84 接口源清单和固定证据 SHA，完整提取 `src/aimdk_msgs/` 到新的 `data/x2_sim/packages/aimdk-preset-v1_0_0/source/aimdk_msgs`；原 ZIP 与 official-source 保持不动。工作副本保留全部原文件及完整 SHA 清单，允许官方 CMake install 仅新增 `prebuilt_x86_64/`。已有 v1.1 install 保持原样。
- Linux build 固定 `/workspace/aimdk-preset-v1_0_0`，`bash --noprofile --norc -ec` 和显式白名单环境，仅 source ROS Humble；colcon 仅构建 aimdk_msgs，顺序 executor、编译并行 2、外层 900 秒限额，失败保留日志并禁止自动重试。记录编译器、CMake、ROS setup SHA、干净环境、全 install SHA 和源/工具清单绑定。没有新增依赖安装或资源启动逻辑。
- inspect 是独立、不连接 ROS 图的旧版 Python/type-support 导入及递归字段核验，产生绑定 build/prepared/MC SHA 的 inspection.json。核验消息源、生成 Python 字段及 `/proc/self/maps` 中实际 aimdk_msgs 库，拒绝混合 prefix。新版主进程运行前独立核验 v1.1 状态/查询/输入源/速度/关节类型；官方旧 Set 请求/响应与递归依赖采用已固定静态 MC 布局证据。此证据不等于 DDS 成功。
- 新 helper 独立加载旧消息，仅服务 `/aimdk_5Fmsgs/srv/SetMcPresetMotion`，唯一 call_async 固定 motion=1002、area=2、interrupt=false、ani_path 空、play_timestamp=0，并设置实际请求 header 时间。没有任意服务、shell、路径或动作输入，不重试。
- IPC 为最长 32 KiB JSON 行，固定 run/probe/随机 nonce、方向内序号、ready→request→sent→response/error；拒绝额外键、重复 JSON 键、错身份、错序、非有限值、无效原始消息及重复请求。stderr 总计最多 64 KiB。归档实际请求/响应字段、序列化 CDR hex、task_id、子进程本地单调/墙钟以及父进程接收时钟，不跨时钟域相减。
- 父 runner 保留新版 GetMcPresetMotionState、MC、odom、关节和保护循环；不再创建新版 Set 客户端。helper stdout/stderr/stdin 非阻塞，保护 tick 不等待子进程。父进程从发送 IPC 起保留 2 秒 ACK 限额，helper 的实际 call_async 同样上限 2 秒；原 startup 12 秒、动作观察 12 秒、runner 40 秒保持。
- 异常、IPC 失败、ACK 超时进入原 UNKNOWN 路径。helper 只允许一次请求，正常退出 poll 回收；finally 终止/必要时 kill 并分别限时 0.5 秒回收，始终记录取消资格 false。终止 helper 不等于取消 MC 动作；GPU/VM 的原独立 finally 清理路径保留。
- host 在启动 VM/MC/Sim 前核验并归档 prepared/build/inspection、官方证据和 binding；当前 helper 源必须与构建工具快照一致，缺产物明确失败且不回退。新兼容实测入口限定单 WAVE_RIGHT、无 cancel-after、180/360 预算。非 wave 不启动 helper，保持原路径。准备提示对齐 Pause → Reset，确认直立且暂停后 Run。

## 已执行的检查

唯一一次相关子系统 CPU 检查：

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_x2_preset_compat.py tests/test_x2_skill_duration.py tests/test_x2_skill_budget.py tests/test_x2_skill_records.py -q
```

**17 passed in 0.09s**。覆盖白名单环境、严格 JSON/身份/字段、单次发送、非阻塞空管道、正常响应、两秒超时、重复/坏消息、子进程退出、stderr 上限、回收、新旧字段拼写规范，以及真实 runner 回调到 ACK/task_id/UNKNOWN、非 wave 无 helper。实际 helper 的唯一 call_async 在隔离假 ROS 下检查了六字段固定请求与重复请求不重发。继承的 duration/budget/event 相关检查因本轮修改其 host/runner 才一并执行；通过后没有重复检查或扩大范围。

假 ROS 仅验证程序接口和归属，不证明真实类型支持可加载、DDS/ACK、挥手播放、关节运动、自然完成或取消。部署脚本的 Linux 构建及实际产物导入仍待协调者在新的独立 PASS 后执行。

## 协调者后续操作

以下是已实现入口，**本轮均未执行**：

1. 宿主 `PYTHONPATH=src python3 deploy/x2/preset-compat.py prepare`。固定输出必须不存在；保留所有旧档案。
2. 在协调者拥有、已挂载 `/workspace` 的 Linux x86_64 环境执行 `python3 /workspace/aimdk-preset-v1_0_0/preset-compat.py build`。不启动 MC/Sim，不并行完整脑；先按契约检查至少 1 GiB 系统预留及资源所有权。
3. 同环境 `python3 /workspace/aimdk-preset-v1_0_0/preset-compat.py inspect`，只读核验产物和生成 inspection 记录。失败保持失败，不自动重建、不改厂商源或加装依赖。
4. 新自然挥手实测仍须单独登记，由协调者沿官方 GPU 生命周期启动；仅 WAVE_RIGHT，`--wave-probe --ui-timeout 180 --wall-timeout 360`，既有 30 秒资源清理预留。未取得真实 ACK/播放/关节/停稳证据前不称通过。

本轮不解锁 M4 挥手，不重复已通过的前进/双向转向，不改变取消资格，不代表 P2/P5 完成。
