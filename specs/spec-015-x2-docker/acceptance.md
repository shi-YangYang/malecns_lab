# Spec 015 验收

2026-10-22；独立验收者 `/root/x2_deployment_acceptance` 只读审查，结论 **PASS（组件环境与查询通信）**。没有修改实现、操作 Docker 或重复构建测试。

| 范围 | 证据 |
|---|---|
| 官方包来源 | `data/x2_sim/manifest.json`、`downloads/*.provenance.json`：三包地址、版本、大小、本地 SHA-256 |
| 独立环境 | `deploy/x2/`、`runtime.json`：amd64、4 CPU/4 GiB、本机 6080、无特权与宿主 socket；既有容器与基础探针快照一致 |
| 构建 | `build.log`、`build-launcher.log` 成功；`aimdk-build.log` 两包完成，耗时 8 分 28 秒；未验收全部 C++ SDK |
| 实际运行 | `sim-final.log`、`mc-final.log`：新镜像启动器运行官方组件；`data/x2_sim/agibot/data/var/x2-ready.png` 显示 MuJoCo 与 X2 |
| 通信 | `get-mc-action.log` 返回 `PASSIVE_DEFAULT / 100`；`odom-once.log` 包含实际 ROS 状态消息 |
| 失败保留 | `mc-run.log` 保留生成 dump 权限失败；`mc-dump-permission.json`、启动器记录修复；后续成功，源 YAML 与官方二进制未修改 |

除明确路径外，日志均位于 `logs/x2_docker/20261022_103048_install/`，本地数据不随 Git 提交。审查包括实际截图与配置/日志，不仅依赖状态描述。

限制：被动模式下机器人倒地，没有站立/行走/速度命令验收；编译结束后仍有调度迟到。只能确认环境和查询链路，不证明实时控制、全部协议兼容、MaleCNS 控制或真机适配。文档已按此范围更新。S2 验证限于部署链路，没有全仓测试、训练、硬件或推送。本次没有 commit。

## 同日资源调整

独立验收者 `/root/x2_resources_acceptance` 只读审查，**PASS（资源配置与重启后的查询）**。Compose 与 `logs/x2_docker/20261022_latency/runtime-resources.json` 一致：8 CPU、6 GiB、4 个 llvmpipe 工作线程；前后 5 秒样本支持 CPU 约 3.4→4.5 核、限流周期 4/50→0/50，无 OOM。模式查询仍返回 `PASSIVE_DEFAULT / 100`。截图 Info 分别为 5、7 FPS；视角、接触状态和未固定的客户端条件限制比较，卡顿未解决，不扩大到动作或实时控制结论。审查仅使用现有配置、日志和截图，没有重复运行 Docker、测试或采样；noVNC 条件措辞已按建议统一。

## 同日 Mac GPU 探针

独立验收者 `/root/x2_gpu_final_acceptance`，**PASS（探针记录与失败后恢复）；X2 GPU 功能仍为 FAIL**。仅审查现有文件与截图，没有启动容器/VM、重复运行测试或发送动作指令。

- `virtualgl-install-probe.log` 的 renderer 为 Zink / Venus / Apple M5；已实际查看 `gears-vgl.png`，三色齿轮像素可见。此证据仅支持小型 OpenGL 场景，不支持 X2 性能结论；黑窗路径的帧数未作为通过证据。
- `lighter-machine.log` 保留 Metal `sampler` 名称遮蔽与管线编译失败；`x2-sim-gpu.log` 和 `x2-sim-discrete-no-cache.log` 均有 Venus ring fatal、官方 `start_sim.sh:169` 内部程序 Aborted。文档正确地未将外层启动脚本退出码 0 当作仿真成功，未声称修复或提速。
- `cleanup-status.log` 显示 Lighter 未运行、原 6080 HTTP 200；`restored-query.log` 返回 `PASSIVE_DEFAULT / 100`；`restored-desktop.log` 为 llvmpipe。已查看 `data/x2_sim/agibot/data/var/x2-gpu-probe-restored.png`，原 MuJoCo 场景恢复且机器人仍倒地，不能解释为站立或行走通过。
- `before.json` / `after.json` 的默认 context 均为 `desktop-linux`；两个无关容器的 ID、状态、起止时间与重启次数一致，仅项目 X2 容器被重启。GPU 草稿位于 Git 忽略的 `data/x2_sim/gpu_probe/deployment-draft/`，正式 `deploy/x2/` 没有 GPU 入口。
- 现有变更范围与日志未显示官方 SDK、MC、Sim 源码/二进制或模型被修改，实际调用仍指向原官方包；本次为记录审查，没有额外执行完整 SDK 文件散列审计。当前交接、Spec 和使用文档均明确兼容实验失败、原 CPU 环境恢复，未把 GPU 草稿作为可用部署交付。

以上日志除另注外均在 `logs/x2_docker/20261022_gpu/`。恢复通过不覆盖 GPU 功能失败；卡顿、实时控制及 MaleCNS 接入仍未解决。

## 同日画面、物理与 MC 时序诊断

独立验收者 `/root/x2_timing_acceptance`，**PASS（一次只读观测、统计与结论边界）**。仅检查已有采集脚本、日志、资源快照和图片；没有操作 Docker、重新采样、运行测试或发送动作指令。

- 采集器只订阅两个状态话题；5 秒预热后，以单调时钟限定 30 秒接收区间，截图每 5 秒一次。主机脚本保存这次运行新增的 MC 日志，退出码为 0，采集结果无错误。
- 已查看 Info 拼图与最后一张原图：7 次显示均为 2 FPS，机器人仍倒地。仿真时间增加 29.695 秒，两个端点的截图捕获间隔为 29.994280263 秒，支持约 0.99 倍实时的估算；低帧率读数滞后已注明，不能解释为精确物理步计数或浏览器交互延迟。
- 按相同 UTC 区间重新读取保存的 MC 日志，得到 115 行：106 次调度迟到，其中 18 次达到所在线程周期、最大 22,150 微秒；另有 9 次指令发布耗时告警，最大 22,521 微秒。所有 `miss %` 均等于当次迟到量除以周期，不是丢包率或全部周期的失败率。
- 资源快照支持平均 5.268 个逻辑核、300 个周期中 0 次容器 CPU 限流、内存 3.633–3.656 GiB，宿主交换空间前后相同。ROS 接收统计为里程计 497.47 Hz、MC 公共状态 10 Hz，未被当作物理仿真频率；接收明细未逐条落盘，因此不能从现有文件定位 104.93 ms 最大接收间隔的具体时刻。
- 第 25 秒截图捕获与保存约 202 ms，附近出现多条较大 MC 告警；现有结果未分离观测开销与原有调度抖动。文档保留该限制，没有把时间接近实时或查询成功升级为站立、行走、神经闭环或真机控制通过。

证据：`logs/x2_docker/20261022_timing/` 的协议、采集与主机脚本、`mc-window-raw.log`、`mc-window-filtered.log`、`analysis.json`、主机快照及拼图；`data/x2_sim/agibot/data/var/x2-timing-baseline/` 的 `summary.json`、`resource_samples.json` 与原图。使用文档、交接和 Spec 的本次结论一致；尚未执行无渲染、无截图的调度对照。仅追加本验收记录，未修改业务或已有运行证据，无 commit/push。

## 同日关闭特效、线程对照与 GPU 兼容绕行

独立验收者 `/root/x2_optimization_acceptance`，**PASS（关闭特效、CPU 线程选择及失败后恢复）；X2 GPU 功能仍为 FAIL**。只审已有配置、采集代码、原始日志及图片，没有操作 Docker/VM、重新采样、运行测试或发送动作指令。文档按 S0 审查，本节不修改实现或既有证据。

- `observe_ab.py` 的截图位于正式 30 秒区间外，期间仅订阅状态并采集资源。按保存的 MC 原始日志及各组 UTC 起止时间独立复核，4/6 线程分别为 158/628 次调度迟到，最大 26.219/41.866 ms；容器 CPU 分别约 4.892/5.059 核，均无配额限流。四张端点原图的 Info 均为 7 FPS，窗口与视角一致，机器人被动接触状态近似匹配。
- GUI 仿真时间需除以约 31.10/31.09 秒截图捕获间隔，得到约 0.99/0.97 倍实时，不能除以正式统计的 30 秒。顺序短样本没有显示 6 线程收益，保留 4 线程有本次证据支持；未把告警差异写成线程数导致抖动的因果证明，也未视作动作或实时控制验收。
- 两次 GPU 命令分别关闭窗口多重采样与配置副本中的离屏多重采样；`gpu-sim-offscreen0.log:58` 明确加载副本。`no-msaa-overlay.json` 记录唯一差异为 `scene.xml` 增加 `quality offsamples="0"`，其余 49 个文件散列一致；未把副本当作正式模型或已验证修复。
- `gpu-final-machine.log` 保留 `texture2d_ms<float> sampler` 遮蔽 Metal 类型的编译错误，两次 Sim 日志均有 Venus ring fatal 和内部程序 Aborted。文档明确 GPU 未接通、没有 GPU 帧率，尚未重编图形兼容库，未把外层脚本成功退出当作通过。
- `restored-runtime.json` 包含 `LP_NUM_THREADS=4` 及四个 llvmpipe 工作线程；`restored-query.log` 返回 `PASSIVE_DEFAULT / 100`。实际查看 `data/x2_sim/agibot/data/var/x2-optimized-restored.png`，Shadow、Reflection、Skybox、Haze 均关闭，Cull Face 保留，Info 为 7 FPS。开关仅对当前窗口生效，重启 Sim 后需再次关闭，文档已注明。
- `private-runtime-final.log` 显示 Lighter 未运行；前后运行快照的默认 context 均为 `desktop-linux`，两个无关容器的 ID、状态、起止时间和重启次数一致，仅本项目容器重启。记录与恢复没有发现阻塞问题，性能和站立/行走能力仍未解决。

除另注外，本节日志均在 `logs/x2_docker/20261022_threads_gpu/`；各组统计与原图在 `data/x2_sim/agibot/data/var/x2-threads{4,6}/`。没有重复已通过的验证、全仓测试、训练或 commit/push。

## 同日 GPU 兼容层重建（未通过）

独立只读审查（子代理，未构建、未操作 Docker/VM、未改文件）核对补丁、构建、崩溃与恢复证据，结论 **FAIL（记录一致性）；GPU 功能仍 FAIL**。主要问题：`vkCreatePipelineCache → readData` 的 backtrace 当时未落盘；文档引用的 MoltenVK/Lighter 散列是中间构建；查询结果无原始日志；`spirv_reflect` 缺失导致的首次链接失败输出未保留。审查确认补丁库与离线 Metal 回归、`-ObjC` 崩溃复现、`user(locn1_3)` 转储、CPU 恢复和官方组件未改均成立，未发现把失败写成成功。

已按审查意见补证据并更正：新增 `machine-cache-readdata-backtrace.log`（同次复现含 sampler 报错）、`final-cpu-query.log`；文档改用最终产物散列（MoltenVK `71612cc9...`、Lighter `9ac09c87...`、app 内副本 `d42190ba...`）并注明调试探针；`spirv-build-provenance.json` 更新 library 散列并保留补 reflect 前的值；误导性的 `spirv-rebuild-with-reflect.log` 改名为 `.stdout.json`。未再次送审；随后继续修复并跑通，见下一节。

## GPU 图形链路修复（2026-10-23，通过；未再送审）

在上一节记录修正后继续修复，最终原版 X2 在 `zink Vulkan 1.2 (Virtio-GPU Venus (Apple M5))` 上正常显示，Info **60 FPS**；同一 GPU 容器内 MC 查询返回 `PASSIVE_DEFAULT / 100`；成功运行无 `ring fatal`/`Aborted`，缓存重建后的再次启动正常。修复包括：`sampler` 保留字、补编 `spirv_reflect.cpp`、`-ObjC` 链接 category、fragment 分量 varying 打包、清空 Mesa/zink 磁盘缓存。关键证据：`spirv-result.json`、`component-fix-fs.metal`、`machine-cache-readdata-backtrace.log`、`gpu-sim-componentfix.log`、`gpu-sim-cleancache.log`、`gpu-sim-warmcache.log`、`gpu-query-clean.log`、`final-artifact-hashes.txt`；截图在 `data/x2_sim/agibot/data/var/`。

限制：被动倒地，无站立/行走/速度指令，无 MaleCNS 闭环；60 FPS 为 GUI Info 读数，非端到端时延。结束时 GPU VM/容器停止，CPU 容器、Sim、MC 恢复并查询 `PASSIVE_DEFAULT / 100`。按用户要求未再送独立验收；没有训练、动作指令或 commit/push。
