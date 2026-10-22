# Spec 016 独立验收

日期：2026-10-23。结论：**PASS，限官方仿真中的站立、短时前进与最终稳定停止。**

本次由新的独立验收者复用实施证据审查，不是新一轮动作验证。只阅读 Spec、Plan、实施报告、只读采集脚本、发布与组件日志、两轮采集结果、关闭记录及截图；对关键轨迹区间做一次离线核对。没有启动 VM、Docker、ROS 或 UI，没有发送动作、运行测试、构建、训练或重复既有检查。仅新增本验收文件，未代替实施者修改其他文件。

## 逐项结果

| 条目 | 判定 | 依据与范围 |
|---|---|---|
| AC1 隔离及可追溯 | PASS | 实施记录显式使用私有 Docker socket、GPU 容器与 LIGHTER_HOME，关闭记录保留默认 context 为 desktop-linux、CPU 容器 exited。当前 Git 变更仅文档；采集器只订阅两话题，未创建控制发布器或服务客户端。命令、组件日志及两轮原始记录保留；官方组件和模型沿用既有版本，本次未改。独立审查未重新对忽略目录内的全部官方二进制做散列复核。 |
| AC2 稳定站立 | PASS | attempt2 的运动前窗口 9.886 s、472 个里程计样本；MC 为 STAND_DEFAULT，body_status=1。高度约 0.6531 m，报告最大倾斜 3.676°；重置后站立截图与姿态数据一致。首次 SD 服务成功却仍倒地明确排除为站立通过证据。 |
| AC3 实际前进 | PASS | velocity2.log 非零指令为前进 0.20、横移 0、角速度 0，发布约 4.999 s。原始轨迹区间跨度 4.980 s，按起始姿态投影前向位移 2.118200 m，大于 0.2 m；MC 进入 LOCOMOTION_DEFAULT，body_status 持续为 1，报告高度 0.632–0.676 m、最大倾斜 4.541°。这是实际位移门槛通过，不是精确速度跟踪通过。 |
| AC4 最终稳定停止 | PASS | 发布零速度持续 12.200 s，末 3 秒窗口内 143 个离散样本的端点跨度为 2.972 s，水平漂移 0.000347434 m，低于 0.1 m。该窗口 MC 为 STAND_DEFAULT、body_status=1；最终截图保持直立。归零后总余程 0.205995 m 已单独记录，不把最终稳定等同于立即停住。 |
| AC5 如实记录及关闭 | PASS | 首次倒地、GUI 重置、第一轮采样缺口及一次补测均保留。shutdown.txt 保存实际停止调用输出：GPU 容器 exited、私有 VM Stopped.、CPU 容器 exited、默认 context desktop-linux。Sim 停止窗口的 terminate/Aborted 已披露，不能称优雅退出；本次没有为复查启动环境。 |

## 核对结果与限制

- 关键区间的样本数、单调接收时间跨度、水平位移和起始身体前向投影与 analysis.json 一致；MC 状态与对应区间相符。采集器记录实际消息与接收时间，按 odom ≤50 Hz、MC ≤10 Hz 抽样，这不是物理频率或控制频率。
- attempt1 在 wall time 1792692684.906 s 到达 240 s 时限，首次非零发布在 1792692704.383 s，存在约 19.48 s 的明确缺口，因此补测有具体依据。attempt2 完整覆盖本次采用的站立、前进和零速观察区间；两份 summary 的 errors 均为空。
- 首次 SD 后原始数据仍显示高度约 0.076 m、STAND_DEFAULT 与 body_status=2；倒地截图一致。GUI 重置后站立只支持恢复初始姿态后的控制，本次没有验证自主起身。
- 指令 0.2 m/s 对应本轮按接收时间计算的平均前向速度约 0.425 m/s，不能称速度已校准。物理仿真时钟与墙钟关系未测，不能把速度差异直接归因于官方配置中的某一参数。
- 截图仅证明拍摄时刻；连续运动与直立结论依赖原始轨迹及 MC 状态。recorded-motion.png 不作为非零速度阶段画面证明；未验证转弯、长期稳定性、MaleCNS 闭环或真机。

没有发现阻碍本 Spec 通过的记录矛盾。速度映射、停车余量、自主起身和官方进程退出异常属于保留的后续问题，不扩张本次通过范围；无需追加实验或重跑检查。

## 证据位置

- `logs/x2_docker/20261023_motion_mvp/`：observe.py、observer-implementation.md、analysis.json、set-sd.log、set-ld.log、shutdown.txt、stand-from-floor.png、stand-after-reset.png、recorded-stop.png。
- `data/x2_sim/agibot/data/var/x2-motion-20261023/`：attempt1/ 与 attempt2/ 的 samples.jsonl、summary.json，velocity.log、velocity2.log、sim.log、mc.log。

路径均相对项目根目录；运行资产位于既有忽略目录，未纳入 Git。
