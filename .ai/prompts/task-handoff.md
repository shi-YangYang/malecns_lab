# 当前交接 — X2 已关闭特效、保留 4 线程；GPU 绕行仍失败

更新：2026-10-22。工作目录 `/Users/yang/project/malecns_lab`；部分工具默认 cwd 仍旧路径，运行命令显式指定工作目录。

## 当前任务

用户授权开始线程对照及 GPU 兼容验证，并要求关闭阴影等。已关闭 Shadow、Reflection、Skybox、Haze；4/6 线程各测一次 30 秒，正式区间不截图，两组相同默认相机并在 GUI Reset 后观察被动状态。两组 Info 均为 7 FPS，CPU 4.89/5.06 核，MC 迟到日志 158/628 次、最大 26.219/41.866 ms，均无容器 CPU 限流；没有 6 线程收益，保留 4 线程。单次顺序样本不构成线程数与告警的严格因果结论。详见 [AGIBOT](../../docs/AGIBOT.md)。

GPU 复用原隔离环境，先关闭 VirtualGL/Mesa 窗口多重采样，再使用官方配置的独立副本仅设 `visual/quality offsamples=0`。两次均仍发生 Metal `texture2d_ms<float> sampler` 命名遮蔽、管线编译失败及内部 Sim 中止，未接通 X2 GPU。副本其余 49/50 文件逐字节相同，原包不改。GPU 容器及私有 VM 已停止，恢复原 CPU 容器 `malecns-x2-x2-1`、Sim、MC 和 4 线程；当前窗口重新关闭特效，重启 Sim 后须再次关闭。桌面为 `http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale`。没有站立/行走验收。下一候选是修补 SPIRV-Cross 保留字并重编外围静态库/私有运行时，本轮未做；不要重复已失败的环境变量绕行或本次有效 A/B。

## 环境与复现

- 源配置 `deploy/x2/`，ROS Humble + Xvfb/llvmpipe/noVNC，8 CPU/6 GiB 限额、4 渲染线程、512 MiB 共享内存；仅发布本机 6080，ROS domain 47/localhost。Docker VM 当前总上限约 10 CPU/7.75 GiB，没有调整全局配置。
- 下载资产 `data/x2_sim/`：MC/Sim v1.0.0-20260522，共同型号 `lx2501_3_t2d5`；AimDK v1.1.0-gfaf4c94。只源码构建 `aimdk_msgs` 与 `py_examples`，不是完整 C++ SDK 验收。
- `manifest.json`、`downloads/*.provenance.json` 保留下载地址和本地 SHA-256；没有官方版本兼容矩阵，查询通过不代表全部接口兼容。
- `x2-component sim` 按提示输入 0；随后 `x2-component mc`。两者都在本项目容器内执行，调用官方脚本。MC 启动器仅修复已观察到的生成 dump 文件权限并清除真机模式环境变量。
- 安装日志 `logs/x2_docker/20261022_103048_install/`；资源调整记录 `logs/x2_docker/20261022_latency/`，含前后短样本、实际限额、启动日志和查询。首次基础探针记录仍保留。不要重复已通过的构建。
- GPU 证据 `logs/x2_docker/20261022_gpu/`，包含发行包校验、真实齿轮截图、原始 X2 失败及恢复日志。`lighter-machine.log` 中已确认独立资源索引配置生效，但仍有 sampler 冲突。
- 时序证据 `logs/x2_docker/20261022_timing/`：协议、采集脚本、主机前后快照、同区间 MC 日志和分析；统计及原图在 `data/x2_sim/agibot/data/var/x2-timing-baseline/`。GUI 时间用于近似实时倍率，ROS header 未用作物理时钟，MC `miss %` 未解释为丢包率。独立审查只使用已有证据。
- 新线程/GPU 证据在 `logs/x2_docker/20261022_threads_gpu/`，含 `comparison.json`、两个线程组的原始 MC 日志、GPU 两次精确启动参数、源码调查、独立配置副本 diff/散列及恢复记录。截图、逐条 ROS 接收与资源统计在 `data/x2_sim/agibot/data/var/x2-threads{4,6}/`。`x2_no_msaa/` 副本仅供失败复现，不作为正式启动配置。
- 私有运行时与资源均在忽略的 `data/x2_sim/gpu_probe/`；始终显式设置 `LIGHTER_HOME=.../runtime` 和 Docker `--host unix://.../runtime/docker.sock`，不改默认 context。当前私有 VM 已关闭，无自启动。GPU 部署草稿移到其中 `deployment-draft/`，未构建、不支持直接运行。

## 已确定方向

- 设备为智元 **X2**，采用官方 Sim + MC + AimDK；随后赴上海现场适配。
- 当前只有 Mac，没有远程 Ubuntu。独立 linux/amd64 下官方组件已启动并完成查询；不能据此宣称实时性能或现场版本匹配。
- 不采用 RoboJuDo-Plus 等第三方运动控制器。先搭建 X2 官方环境并验证手动运动指令，再接 MaleCNS；单轴稳向不是前置门槛。

## 证据与资产

- 完整参考为 166,700 节点、25,582,938 内部有向连接、124,177,617 突触计数；原始发布边界见 [MODEL_V2](../../docs/MODEL_V2.md)。
- 目标跟随全 GPU 160 窗为历史训练过程结果，未证明最终策略可靠控制。
- 单轴 joint/frozen 各 80 窗已完成，固定模型验证失败，锁定测试未开启。控制失败、原始 run 与数值源码快照仍保留；见[历史结果](../../docs/EXPERIMENT_RESULTS.md)。不重跑旧训练或诊断。
- data/、logs/、apps/atlas/public/data/、.venv、.venv-train 均为本地忽略资产，不删、不提交、不改旧 run。
- 单轴 run 在 data/yaw_joint_20261021_r1/ 与 data/yaw_frozen_20261021_r1/，含 sources 快照。

## 环境与约束

Noria 不得停止，Docker 不得为本项目重启或更换后端。本轮只重启本项目组件及项目私有 GPU VM，切换 GPU 探针时停 CPU 项目容器，随后恢复；默认 Docker context 不改。线程样本峰值内存为 3.68/3.61 GiB，6 GiB 只是上限。完整脑任务仍遵守 1 GiB 系统预留，不与仿真并行启动完整脑训练。

## 验证与提交

此前 [Spec 014](../../specs/spec-014-x2-repository-cleanup/spec.md) 清理已提交为 `e81b2e5`，未推送。本次 S2 仅验证新部署链路：shell 语法、Compose 配置、镜像/接口构建、组件启动、画面与查询/odom通信；[独立验收](../../specs/spec-015-x2-docker/acceptance.md) PASS，没有运行全仓测试或训练。本次未 commit/push。

同日资源配置变更按 S0 审 diff；另完成用户所需前后性能短样本及重启后查询。独立只读验收 PASS，保留卡顿未解决的限制，未追加构建或全仓检查。

同日 GPU 环境尝试按 S2 验证 Vulkan → 可见 OpenGL → 官方 X2；前两层通过，最后一层失败并停止。原 CPU 环境恢复检查通过。独立验收 PASS（记录与恢复），X2 GPU 功能仍 FAIL；不覆盖此前 CPU 组件安装的 PASS。未重复 live check、构建或全仓测试。

同日时序诊断完成一次有效 30 秒采样；文档为 S0，只审 diff。独立验收 PASS（测量记录与结论边界），使用已有原始日志、截图和统计，没有重复运行。ROS 单条接收记录未落盘，最大接收间隔无法定位到具体时刻；当前结论限于被动场景。未 commit/push。

同日关闭特效、4/6 线程对照和两次 GPU 多重采样绕行已结束。新隔离采集脚本 S1，一次 AST 检查通过；显示配置副本 XML 解析与字节 diff 通过；无全仓检查或训练。独立验收 PASS（关闭特效、CPU 选择、失败记录与恢复），GPU 功能仍 FAIL。最终截图为四特效关闭、7 FPS，恢复查询为 `PASSIVE_DEFAULT / 100`，4 个 llvmpipe 工作线程已确认；默认 context 和无关容器前后状态相同。以上部署、实测与文档随本次提交归档，未推送。
