# M3 冻结缓存恢复实施报告

2026-10-23。状态：实施完成，待新的独立工程验收；**尚未实际运行恢复**。依据 `m3-cache-recovery-contract.md` Ready 实施，风险 S3（训练来源、缓存持久化身份、恢复分支）。本次只改训练 CLI/编排、新增严格缓存准入与最小 CPU 用例；不改 policy/encoder/data/brain、不改 joint 调度，不碰桌面，不修改旧 run、模型、源码快照或任何 locked-test 文件，未 commit。

## 实际修改

- `experiments/x2_behavior_training.py` 增加显式 `--feature-cache`，仅允许 readout。缓存路线固定源 r1、r2 **initial-checkpoint**、seed2401、50train/50validation、200 更新、600 秒；错误或缺省 checkpoint、其他模式/预算/目录均拒绝。没有该参数时走原路径。
- `learning/x2_behavior_cache.py` 新增专职准入/加载。绑定原 r1 registration ID、r2 initial checkpoint ID、policy/input-map ID；原 r1 必须仍是 wall_limit 且无 worker_result。新运行保留独立 registration 和 `cache_manifest.json`，绑定缓存根目录、来源 registration/file hash、完整文件清单、初始 checkpoint、选样顺序、数据及数值 runtime、允许的源码差异。
- 准入逐项核对旧 registration canonical ID、原 37 份来源快照、当前数值来源、dataset manifest 和 train/validation 内容 hash、确定 task-round-robin 顺序、帧/状态目录完整集合。所有 818 个 frame JSON/NPZ 和 918 个 state JSON/NPZ 均逐个读取；检查 ZIP/数组重复与多余字段、解压大小界、精确 float32/int64 shape/dtype、有限性、文件及逐数组 hash、canonical state ID、600+500×frame 时间、实际 Observation/编码及前后状态链。100 个预热初始数组声明必须相同；主机归档时间依顺序严格递增。
- 缓存只提供 `features_mv` 给已有三头。旧 logits、teacher、编码或 state 不能作为新的头输入；其余档案只作完整性/链路证据。状态约 2.5 GB 顺序校验，不复制到新 run、不安装到神经状态；worker 仅保留已验证同一字节中提取的约 8.6 MB 特征。
- 新旧来源差异仅放行 `experiments/x2_behavior_training.py`、`learning/x2_behavior_training.py` 和新增 `learning/x2_behavior_cache.py`；未知新增/删除或数值/输入/教师变化拒绝。训练模块内部 `contract`、`select_episodes`、checkpoint 读写、loss、预测、梯度审计、状态/输出归档、`cached_outputs` 和图计数函数另与旧源码做 AST 完全比较。原 **200 步读出循环本体**也必须 AST 完全一致，保留每步全验证、最低 loss、最早平局和原保存逻辑。
- 父进程准入后封存清单，只复制已核对的 dataset manifest/train/validation 字节；不再为缓存路线调用生成完整 dataset 的函数，因此不读取或重生成 locked-test。worker 使用前重新进行完整准入，与新登记的清单逐文件/身份比较；使用同一已验证字节解析得到的特征，拒绝校验后替换。任何失败直接结束，没有坏帧跳过、部分恢复或重新算脑 fallback。
- worker 沿既有完整图/严格 checkpoint 恢复，保留真实 graph hash/结构和全部七组参数校验，然后冻结 theta、创建 fresh Adam。神经保护在完整图/策略构造之前安装，对 policy forward/initial_state/_step/warmup 与 brain step/reset/forward/warmup 的存在入口计数并拒绝。恢复成功须记录所有计数为零、初始七组参数身份、初始 Adam 状态为空、200 次更新与 200 条验证 history、theta 不变以及正式 selection/validation-scores。
- 父进程准备与审计纳入同一次 600 秒墙钟；准入每次文件读取前核对剩余时间。后续仍使用既有唯一完整脑锁、MPS 无 CPU fallback、CPU4/interop1、RSS/MPS 预算与至少 1 GiB 系统预留，仿真预检和关闭规则不变。

## 实际检查

只执行一个受影响 CPU 检查命令：

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_x2_behavior_cache.py -q
```

**8 项通过，6.26 秒。**没有失败重跑、全仓检查、MPS/完整图加载、真实缓存重审、训练或仿真。

- 完整 50/50、818 frame / 918 state 档案 fixture 覆盖父准入及 worker 原字节复核、无 locked-test 读取、保留全部特征。
- fixture 仅将状态数组宽度通过 pytest monkeypatch 改小，不实例化神经图，生产 API 没有维度/身份 override，也没有合成网络回退；生产参数保持 166700 节点、3377 输入、2628 特征、18 个 pending 数组。
- 拒绝 frame/state hash 篡改、实际 Observation 换值、checkpoint/状态步数不符、目录缺失/多余项、episode 顺序改变、数值来源改变，以及 sealed 清单后仅改 JSON 空白的替换。
- NPZ 多余/重复字段、对象数组、非有限、错误 dtype/shape 被拒绝；路径读取本身禁止 locked-test，并逐级拒绝符号链接。
- 从当前源码抽取**原封不动的 readout 200 次循环 AST**，只用 CPU 三头 fixture 和原 `read_features`、`loss_for`、梯度审计运行：七组初始参数完整恢复、Adam 初始为空、四轮固定 50 条顺序、每次验证 50 条、200 次更新/history 均成立。人为验证 loss 平局用例选择最早第 101 次；这只是控制流检查，不是候选模型分数。theta 不变，循环内零神经调用；尝试调用被禁的 forward/initial_state/_step 会明确失败。

最后仅检查限定范围 diff。上述 CPU 用例不替代真实缓存/graph/checkpoint 的后续运行准入，不声称 r1 失败已修成成功。

## 待独立验收及实际运行

独立验收可核对上述边界、源码允许集合、原 readout 循环及检查证据，无需重复通过的 CPU 用例。工程通过后由协调者登记新的输出目录并串行执行一次：

```sh
PYTHONPATH=src .venv-train/bin/python -u -m malecns_lab.experiments.x2_behavior_training \
  --execute --mode readout \
  --feature-cache data/x2_behavior_readout_20261023_r1 \
  --checkpoint data/x2_behavior_probe_20261023_r2/initial-checkpoint \
  --train-episodes 50 --validation-episodes 50 --max-updates 200 --wall-seconds 600 \
  --output data/<协调者登记的新目录>
```

该命令尚未执行。本次恢复的真实审计耗时、资源峰值、200 更新终态、验证选择及零神经调用证据只能在新 run 完成后报告。原 r1 保持失败，其第 149 或 168 次参数不会用于恢复；没有补造其 worker_result/selection。恢复完成也仅得到模拟反馈/工程教师的冻结读出基线，不通过 P3/P4/P5 或真机门槛；joint 仍需独立决定。
