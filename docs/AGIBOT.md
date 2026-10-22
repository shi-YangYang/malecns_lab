# X2 官方仿真与现场接入

当前选择 **智元官方 X2 Sim + MC + AimDK**。Mac GPU 仿真已完成官方动作验证和[项目动作适配器验收](../specs/spec-018-x2-motion-adapter/acceptance.md)：受控意图经自定义输入源前进，归零后结合里程计与MC确认停稳。官方SDK、MC、Sim与模型未改；完整MaleCNS闭环、精确速度映射与真机适配尚未完成。仿真按需启动，当前GPU容器、私有VM和CPU仿真均已关闭。

[最终适配器实测](../specs/spec-018-x2-motion-adapter/implementation.md)：约2秒非零目标带来整轮约0.901m前向位移，首零至确认停车约5.09秒；最终连续1.007秒的最大水平速度0.0173m/s。目标0.2不是实测速度，这一轮的位移与停车时间不是通用校准或刹停保证。失败集成与修复证据均保留。

下一阶段为[完整脑冻结推理基线](../specs/spec-019-x2-neural-inference/spec.md)，先在仿真关闭时核对最终训练参数、实测完整图前向时延与资源，再制定真实反馈联调协议。当前官方配置没有相机图像发布；计划使用明确标注的里程计驱动虚拟标记，不冒称真实相机。

## 官方组件

| 组件 | 作用 |
|---|---|
| `sim_mujoco` | 机器人物理仿真、关节与惯性状态反馈 |
| `mc` | 官方运动控制，将高层运动目标转换为关节指令 |
| AimDK | 应用侧感知与运动接口；具体可用接口以对应版本为准 |

官方仿真文档要求 **x86_64 Ubuntu 22.04**。Apple Silicon Mac 的容器默认架构为 ARM，本项目使用 `linux/amd64` 转译，已有软件渲染与私有 GPU 图形路径。当前没有验证 Mac 与官方 Linux 部署等价，也没有更换第三方步态控制器。

- [官方开发站](https://x2-aimdk.agibot.com/zh-cn/latest/)。
- [官方仿真部署](https://x2-aimdk.agibot.com/zh-cn/latest/sim_contents.html)。
- [SDK 与组件下载](https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html)：`mc-x86`、`sim_mujoco-x86` 在 Extras 区。
- [运动与关节接口](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/index.html)。
- [Docker Desktop 设置](https://docs.docker.com/desktop/settings-and-maintenance/settings/)：转译能力取决于宿主、虚拟化后端及设置；不能据此推断 X2 已兼容。

## Mac 基础探针（2026-10-22）

在 ARM64 Mac、Docker Desktop 的 `desktop-linux` 上完成一次隔离探针：

| 检查 | 实测结果 |
|---|---|
| 镜像与系统 | `ubuntu:22.04`，镜像 `linux/amd64`，容器报告 Ubuntu 22.04、`dpkg` 架构 `amd64` |
| 实际程序 | Bash ELF 头为 64 位 x86-64（`e_machine=62`）；Bash 5.1.16、glibc 2.35 及 awk 整数/浮点断言均执行成功，退出码 0 |
| 隔离与清理 | 限制 1 CPU、512 MiB，无网络、无宿主挂载、只读根文件系统；容器通过 `--rm` 删除 |
| 既有容器 | 前后 ID、状态、起止时间与重启计数一致；未停止其他服务或重启 Docker |

镜像摘要：`ubuntu@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956`。本地原始命令、日志与前后快照位于 `logs/x2_docker/20261022_102511_amd64_probe/`，不纳入 Git。检查镜像时需用 `docker image inspect --platform linux/amd64`；本机未指定平台时返回的是架构字段为空的多架构索引。

该结果只证明 Ubuntu amd64 基础用户态可执行，未识别具体转译器，也未验证 X2 二进制、ROS 2、图形显示、通信或仿真实时性。基础镜像已保留，本阶段没有常驻测试容器，没有安装 X2 或启动训练。

## 官方组件实测（2026-10-22）

本节首次实测使用 4 CPU、4 GiB 内存限额和 2 个 llvmpipe 渲染线程；结果不代表后续资源调整后的性能。

| 组件 | 本地版本与结果 |
|---|---|
| MuJoCo HAL | 官方 `sim_mujoco-x86-v1.0.0-20260522`；窗口显示 X2，已读取 `/aima/hal/odom/state` 消息 |
| MC | 官方 `mc-x86-v1.0.0-20260522`；以 `mc_sim.yaml` 启动，查询返回 `PASSIVE_DEFAULT` / `100` |
| AimDK | 官方 SDK `v1.1.0-gfaf4c94`；在 amd64 中源码编译 `aimdk_msgs`、`py_examples`，两包完成耗时 8 分 28 秒，官方 `get_mc_action` 获得回复 |
| 图形 | Xvfb、Mesa llvmpipe、OpenGL 4.5；通过 noVNC 在本机浏览器显示，没有使用 Mac GPU |

仿真型号为两个官方包共同提供的 `lx2501_3_t2d5`。SDK 压缩包名带 `aarch64`，本次使用其源码构建 x86 接口，没有执行 ARM 预编译库，也没有编译全部 C++ 示例。官方未提供这三个版本的兼容矩阵；查询成功只证明本次查询接口兼容，不代表所有动作接口通过。

首次 MC 启动在 Mac 挂载目录生成 `mc_sim.yaml.dump` 后报权限错误；失败文件权限为 `0200`。`x2-component mc` 仅预建此生成文件并赋予所有者读写权限，随后调用官方脚本；不修改源配置或控制器。启动器会清除 `AGIBOT_SOFTWARE_HOME`，确保使用仿真模式。

**限制：**没有发送站立或速度指令。默认 `PASSIVE_DEFAULT` 下机器人已倒地，画面不代表站立通过。SDK 编译结束后仍观察到 MC 的 `missed wakeup time` 告警；目前只能确认组件与查询通信可用，尚未达到实时控制验收。下一步应重置仿真，验证官方站立、运动和停止，并测量控制节奏，再接神经模型。

原包及本地实测 SHA-256 保存在 `data/x2_sim/downloads/` 和 `data/x2_sim/manifest.json`；记录位于 `logs/x2_docker/20261022_103048_install/`。这些文件不纳入 Git。下载源为上方官方索引，散列是本地计算结果，不是厂商签名。部署源码见 [deploy/x2](../deploy/x2/compose.yaml)。

## 桌面性能与资源调整（2026-10-22）

因桌面与视角交互迟缓，将本项目容器提高到 8 CPU / 6 GiB，渲染线程从 2 改为 4；已重建本项目容器并恢复 Sim、MC，查询仍返回 `PASSIVE_DEFAULT / 100`。没有重启 Docker、重编译 SDK 或修改原模型。Docker VM 当时总上限为 10 CPU / 约 7.75 GiB；容器额度是使用上限，不是预留全部资源。

| 观察 | 原配置 | 新配置 |
|---|---|---|
| MuJoCo Info 帧率快照 | 约 5 FPS | 约 7 FPS |
| 5 秒样本容器平均 CPU | 约 3.4 核 | 约 4.5 核 |
| 样本 CPU 限流周期 | 4 / 50 | 0 / 50 |
| cgroup 内存占用 | 约 3.30 GiB | 约 3.10 GiB |

两次均为 1280×800 桌面、最大化窗口、关闭阴影和反射；重启后视角、接触状态不同，且未固定 noVNC 客户端连接条件，因此仅为资源调整前后快照，不能视为严格性能基准或端到端延迟验证。原始开启阴影/反射时曾观察到约 2 FPS。没有发生内存上限事件或 OOM；增加内存额度不等于渲染提速。当前仍是 amd64 转译与 CPU 软件渲染，尚未解决流畅交互或实时控制。

证据：`logs/x2_docker/20261022_latency/` 下的 `resources-before.json`、`resources-after.json`、`runtime-resources.json`、`get-mc-action.log`；截图为 `data/x2_sim/agibot/data/var/x2-resource-{before,after}.png`。阴影/反射开关仅在本次窗口生效，重启仿真后可在 Rendering → OpenGL Effects 再关闭；未进行网格减面。

## Mac GPU 探针（2026-10-22，未通过 X2 验收）

在不改官方 SDK、MC、仿真二进制、原模型和物理参数的前提下，尝试项目内独立 [Lighter 0.13.7](https://github.com/fieldwork-ai/lighter/releases/tag/v0.13.7)，通过 Venus/MoltenVK 把容器 Vulkan 接到 Apple M5。此路线是第三方兼容实验，不是智元对 macOS GPU 的支持承诺。发行包按 GitHub release 的 SHA-256 校验，安装器 Apple 签名校验通过；安装与状态均在 `data/x2_sim/gpu_probe/runtime/`，没有切换默认 Docker context、安装自启动或重启 Docker Desktop。

| 阶段 | 实际结果 |
|---|---|
| amd64 Vulkan | `Virtio-GPU Venus (Apple M5)`，真实 GPU 设备初始化通过 |
| OpenGL 兼容 | 原 Ubuntu Mesa 23 缺内核 CROSS_DEVICE；Mesa 26 要求当前 MoltenVK 缺失的 nullDescriptor；改用签名 PPA 的 Mesa 25.0.7，枚举 OpenGL 2.1 |
| 可见画面 | 直接 Xvfb 呈现失败；关闭 Kopper 的高帧数对应黑窗，不能算显示通过。加入 [VirtualGL 3.1.5](https://github.com/VirtualGL/virtualgl/releases/tag/3.1.5) 的 EGL 回读后，截图确认齿轮实际可见，renderer 仍为 Apple M5 |
| 原版 X2 | Metal 编译报告纹理变量 `sampler` 遮蔽同名类型，随后 Venus ring fatal，仿真进程中止；官方启动脚本退出码为 0 不能代表内部程序成功 |
| 定向兼容尝试 | 宿主 `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=0` 已由日志确认生效；禁用 Mesa 缓存后仍出现同类名称冲突，停止继续尝试 |
| 恢复 | GPU 容器与私有 VM 已停止；原本机 6080 桌面和 Sim/MC 恢复，模式查询 `PASSIVE_DEFAULT / 100`；渲染仍为 llvmpipe |

驱动还报告缺少 logicOp、custom border color 和 depth clip 能力；没有完成 X2 画面正确性、帧率或动作验收。该实验不能说明 GPU 版 X2 更快，也不能说明官方 Linux GPU 路线不可用。后续需先修复或等待图形兼容层解决上述着色器问题，再重新验证；不要把尚未跑通的 GPU 环境作为机器人控制基础。

证据位于 `logs/x2_docker/20261022_gpu/`：`downloads.json`、`vulkan-amd64-proxy.log`、`glx-mesa25-gpu-readback.log`（黑窗路径）、`virtualgl-install-probe.log`、`gears-vgl.png`（实际可见像素）、`x2-sim-gpu.log`、`x2-sim-discrete-no-cache.log`、`lighter-machine.log`、`restored-query.log`、`cleanup-status.log`、`before.json` / `after.json`。失败部署草稿仅保存在忽略的 `data/x2_sim/gpu_probe/deployment-draft/`，没有作为正式入口交付；移动后的相对路径不可直接使用。没有 commit/push。

## 画面、物理与控制时序实测（2026-10-22）

在已恢复的 CPU 容器内连续观察 30 秒（北京时间 12:52:45.585–12:53:15.585）。1280×800 桌面、最大化 MuJoCo 窗口，阴影和反射开启；机器人保持被动倒地状态。仅打开 Info、读取状态和日志，没有重启服务、修改 SDK 或发送运动指令。

| 测量 | 结果与含义 |
|---|---|
| 画面 | 7 次 Info 快照均为 **2 FPS**；没有测量浏览器鼠标到画面的端到端延迟 |
| 仿真时间 | `2613.238 → 2642.933 s`，推进 29.695 s；截图捕获间隔 29.994 s，估算 **约 0.99 倍实时**。低帧率造成读数滞后，不是精确物理步计数 |
| 里程计接收 | 14,924 条，**497.47 Hz**（配置 500 Hz）；接收间隔 p95 为 2.27 ms，最大 104.93 ms。最大间隔是本次观察者的接收间隔，不等于已证明的发布停顿或丢包 |
| MC 公共状态接收 | 300 条，**10.0 Hz**（配置 10 Hz）；接收间隔 p95 为 100.43 ms，最大 110.77 ms |
| MC 调度 | **106 次** `missed wakeup time` 日志，其中 18 次迟到达到该线程的一个周期；最长迟到 **22.15 ms**（腰部指令线程，周期 2 ms） |
| MC 指令发布 | **9 次** `Publish joint command too slow`，最长耗时 **22.52 ms**；该时段没有其他类别的 MC 日志，不据此断言不存在控制故障 |
| 资源 | 容器平均使用 **5.27 个逻辑核 / 8 核限额**，其中 Sim 约 4.50 核、MC 约 0.49 核；300 个限流检查周期内 **0 次 CPU 限流**。内存 3.63–3.66 GiB / 6 GiB，宿主交换空间未增长 |

MC 迟到事件按线程分别为：手臂 37、腿部 39、腰部 23、头部 1、腿部里程计 5、调试发布 1。上述次数只统计日志事件，不是全部执行周期的失败率；日志 `miss %` 是单次迟到量除以该线程周期。未获得 MC 调度实现源码，无法确认告警阈值、日志抑制及补执行策略。

**结论：当前场景的物理时间接近实时，但显示明显迟缓，控制调度存在迟到记录；尚未证明站立或行走时也能稳定运行。** 配置中的物理步长为 1 ms，仿真、GUI 和状态发布各有执行线程，ROS 接收频率不能替代物理实时倍率。来源为 `sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml`、`simulator/default.yaml` 和 `mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml`（相对于 `data/x2_sim/packages/`，前两个文件属于同一机器人配置目录）。

采样本身有开销；第 25 秒截图捕获与保存约 202 ms，附近出现多条较大的 MC 延迟，尚不能分离采样影响与原有调度抖动。当前结果不支持“多分配 CPU 就会解决”的判断。下一项有针对性的实验是关闭画面渲染但保留官方物理与 MC，使用不截图的低开销观测比较调度；此对照尚未执行。

原始证据：`logs/x2_docker/20261022_timing/` 下的 `protocol.json`、`observe.py`、`run_observation.py`、`host-before.json` / `host-after.json`、`mc-window-raw.log`、`mc-window-filtered.log`、`analysis.json` 和 `info-contact-sheet.png`；接收与资源统计、7 张原图位于 `data/x2_sim/agibot/data/var/x2-timing-baseline/`。日志与图片不纳入 Git；只执行了一次有效采样。

## 关闭特效与线程对照（2026-10-22）

关闭 Rendering → OpenGL Effects 中的 **Shadow、Reflection、Skybox、Haze**，保留纹理与 Cull Face。1280×800 桌面、最大化窗口、默认相机，两组均在 GUI Reset 后以相同流程观察被动状态；仅切换本项目 Sim 的 `LP_NUM_THREADS`，MC 保持运行。每组正式记录 30 秒，截图放在统计区间前后，避免此前截图干扰调度统计。

| 指标 | 4 个渲染线程 | 6 个渲染线程 |
|---|---:|---:|
| 前后 Info 帧率 | 7 / 7 FPS | 7 / 7 FPS |
| 容器平均 CPU | 4.89 核 | 5.06 核 |
| CPU 限流周期 | 0 / 300 | 0 / 300 |
| MC 调度迟到日志数 | 158 | 628 |
| 最大调度迟到 | 26.219 ms | 41.866 ms |
| GUI 时间估算实时倍率 | 约 0.99 | 约 0.97 |

这次短样本没有体现 6 线程的收益，选择保留 **4 线程**。两组为顺序执行的单次样本，重启和被动接触状态仍可能影响结果，不能据此证明增加线程必然导致告警增加。开启特效的旧样本为 2 FPS，关闭后的本次为 7 FPS；两者条件和时段不同，不作为严格提速比例。`comparison.json`、各组原始日志、接收时间序列与截图均保留。

GPU 又做了两项针对性验证：第一项使用 VirtualGL `VGL_SAMPLES=0 / -ms 0` 和 Mesa `DRI_NO_MSAA=1` 关闭窗口多重采样；第二项在独立配置副本中仅增加 `<quality offsamples="0"/>` 关闭 MuJoCo 离屏多重采样，经官方 `start_sim.sh -c` 加载。副本 50 个文件中其余 49 个逐字节相同，原始文件未修改。两次仍在 `texture2d_ms<float> sampler` 遮蔽 Metal `sampler` 类型处失败，随后内部 Sim 中止；**GPU 尚未接通，也没有可用 GPU 帧率**。阴影等 UI 开关无法在中止前应用于该 GPU 窗口。

相关依据：[VirtualGL 3.1.5 多重采样选项](https://github.com/VirtualGL/virtualgl/blob/3.1.5/doc/index.html#L3766-L3806)、[Mesa 环境变量](https://docs.mesa3d.org/envvars.html#envvar-DRI_NO_MSAA)、[SPIRV-Cross 同类命名冲突](https://github.com/KhronosGroup/SPIRV-Cross/issues/293)。下一候选修复是外围 SPIRV-Cross 的保留字处理，但本地 Lighter 静态链接 MoltenVK，需要重建相关库和私有运行时；本轮未执行，不承诺其解决后续全部兼容问题。

GPU 容器与私有 VM 已停止，恢复原 CPU 环境、4 线程并重新关闭上述特效。这些开关是当前窗口状态，重启 Sim 后需要再次关闭。没有修改官方源码、二进制、物理或 MC 参数，没有发送站立/速度命令，也没有 commit/push。

证据位于 `logs/x2_docker/20261022_threads_gpu/`：协议、`comparison.json`、`threads4/`、`threads6/`、`threads6-runtime.json`、`gpu-investigation.json`、`no-msaa-overlay.json`、两次 GPU 启动参数与日志、最终机器日志及恢复记录。两组原图和统计在 `data/x2_sim/agibot/data/var/x2-threads4/` 与 `x2-threads6/`；独立配置副本在 `data/x2_sim/agibot/data/var/x2_no_msaa/`，仅供复现失败，不作为正式入口。

## GPU 图形兼容层重建（2026-10-22，通过）

用户要求继续推进。在项目内重建外围图形栈后，**原版 X2 已在 Apple M5 GPU 路径上显示并稳定运行**；官方 SDK、MC、Sim、模型和物理参数未修改。补丁与构建在忽略目录 `data/x2_sim/gpu_probe/rebuild/`，证据在 `logs/x2_docker/20261022_gpu_rebuild/`。最终二进制含调试探针（着色器转储、崩溃 backtrace、跳过缓存的开关），属于实验构建，不是发布产物。

| 步骤 | 结果 |
|---|---|
| SPIRV-Cross 补丁 | `sampler` 加入 MSL 保留字；补入遗漏的 `spirv_reflect.cpp`；针对 fragment 输入的分量 varying 增加打包；静态库 `2b1bd3...` |
| MoltenVK 1.4.2 | 源码重建并链接补丁版 SPIRV-Cross，`0a7469...` |
| Lighter 0.13.7 | 重编主机程序，必须带 `-ObjC` 链接 ObjC category；二进制 `b90247...` |
| 原版 X2 画面 | `zink Vulkan 1.2 (Virtio-GPU Venus (Apple M5))`，场景正常；Info 显示 **60 FPS** |
| MC | 同一 GPU 容器内启动 MC，查询返回 `PASSIVE_DEFAULT / 100` |
| 稳定性 | 成功运行中无 `ring fatal`/`Aborted`，Sim 持续运行；重复启动（缓存已重建）同样正常 |

修复链（按发现顺序）：

1. **保留字冲突**：SPIRV-Cross 的 MSL 输出把纹理变量命名为 `sampler`，遮蔽 Metal 类型。补丁将其重命名为 `sampler0`，离线回归由 FAIL 变 PASS。
2. **静态库遗漏**：首次重建漏编 `spirv_reflect.cpp`，Lighter 链接时报 `CompilerReflection` 未定义；补入后通过。
3. **ObjC category**：静态链接 MoltenVK 时需 `-ObjC`，否则 `MTLSamplerDescriptor setBorderColorMVK:` 未加载，VM 创建 sampler 时崩溃。
4. **分量 varying**：Zink 生成的 fragment SPIR-V 在同一 location 同时有 `float4` 与 `float` 分量输入（`slot_4` / `slot_4_c3`），SPIRV-Cross 输出两个 MSL 成员，Metal 拒绝。补丁对 fragment 输入启用分量打包，并让带 `Component` 的变量先于整向量变量生成，复用单个 `float4 m_location_N` 成员；真实转储着色器已验证为单一成员。
5. **持久化缓存**：X2 首次运行仍报旧 `sampler` 错误，backtrace 为 `vkCreatePipelineCache → MVKPipelineCache::readData → MVKShaderLibraryCache::addShaderLibrary → compile failed`。数据来自 **Mesa/zink 的磁盘着色器缓存**，其中保存了补丁前生成的 MSL；清空 `/root/.cache/mesa_shader_cache*` 一次（或设置 `MESA_SHADER_CACHE_DISABLE=true`）后，补丁版翻译器重新生成有效缓存，后续启动无需再清。

复现（在项目根目录，GPU VM 与 CPU 容器不同时运行）：

```bash
docker stop malecns-x2-x2-1
cd data/x2_sim/gpu_probe
LIGHTER_HOME=$PWD/runtime-rebuild ./runtime-rebuild/bin/lighter start
docker --host unix://$PWD/runtime-rebuild/docker.sock start malecns-x2-gpu-trial
printf '0\n' | docker --host unix://$PWD/runtime-rebuild/docker.sock exec -i \
  -e DRI_NO_MSAA=1 -e VGL_SAMPLES=0 -e LIBGL_KOPPER_DISABLE=true \
  malecns-x2-gpu-trial /opt/VirtualGL/bin/vglrun -d egl0 -ms 0 /usr/local/bin/x2-component sim
```

首次切换到新二进制时，若仍出现 `sampler` 报错，先执行一次：

```bash
docker --host unix://$PWD/runtime-rebuild/docker.sock exec malecns-x2-gpu-trial \
  rm -rf /root/.cache/mesa_shader_cache /root/.cache/mesa_shader_cache_db
```

限制：机器人保持被动倒地，没有发送站立/行走/速度指令；没有实现 MaleCNS→AimDK→MC 闭环；60 FPS 是 MuJoCo GUI 的 Info 读数（vsync 上限），不是端到端控制时延；私有 VM 8 CPU/6 GiB、GPU 容器 4 GiB，仍是 amd64 转译 + Venus/Zink 实验路径，不等于现场性能。结束时 GPU VM 与容器已停止，CPU 容器、Sim、MC 恢复，查询 `PASSIVE_DEFAULT / 100`；CPU 侧特效默认开启、未重新关闭。

证据：`logs/x2_docker/20261022_gpu_rebuild/` 的 `spirv-result.json`、`component-fix-fs.metal`、`machine-cache-readdata-backtrace.log`、`machine-gpu-success.log`、`gpu-sim-componentfix.log`、`gpu-sim-cleancache.log`、`gpu-sim-warmcache.log`、`gpu-query-clean.log`、`final-artifact-hashes.txt`、`compiled-msl/`；截图 `data/x2_sim/agibot/data/var/x2-gpu-componentfix-1.png`、`x2-gpu-fps-info.png`、`x2-gpu-ui-effects-off.png`、`x2-gpu-cleancache.png`、`x2-gpu-clean-fps.png`、`x2-gpu-warmcache.png`。没有训练、动作指令或 commit/push。

## 本机启动与停止

在项目根目录运行。已安装的包位于 `data/x2_sim/packages/{aimdk,mc,sim_mujoco}`；仅本项目容器使用这些目录。当前部署配置限制 8 CPU、6 GiB、512 MiB 共享内存，llvmpipe 使用 4 个渲染线程；不发布 ROS 端口，不停止其他服务。

```bash
cd /Users/yang/project/malecns_lab
docker compose -f deploy/x2/compose.yaml up -d
```

浏览器打开 [X2 仿真桌面](http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale)。桌面默认不自动启动机器人组件。

终端一启动仿真，按提示输入 `0` 选择 `lx2501_3_t2d5`；窗口出现后保留此终端：

```bash
docker compose -f deploy/x2/compose.yaml exec x2 x2-component sim
```

终端二启动 MC；首次没有型号缓存时同样选择 `0`，保留此终端：

```bash
docker compose -f deploy/x2/compose.yaml exec x2 x2-component mc
```

终端三查询模式：

```bash
docker compose -f deploy/x2/compose.yaml exec -T x2 bash -lc 'timeout 20s ros2 run py_examples get_mc_action'
```

停止本项目环境（保留下载、构建产物和日志）：

```bash
docker compose -f deploy/x2/compose.yaml down
```

如需从已下载的源码重建，只构建本阶段使用的两个包：

```bash
docker compose -f deploy/x2/compose.yaml build
docker compose -f deploy/x2/compose.yaml up -d
docker compose -f deploy/x2/compose.yaml exec -T x2 bash -lc 'cd /workspace/aimdk && CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 colcon build --packages-select aimdk_msgs py_examples --executor sequential --cmake-args -DBUILD_TESTING=OFF'
```

## 接入顺序

1. 在独立容器验证 x86 程序及官方组件可启动，不停止或改动正在使用的 Noria 服务。
2. 先通过官方指令验证仿真 X2 的站立、前进、转弯和停止，并记录身体反馈。
3. 接入感知 → MaleCNS → 动作解码 → AimDK → MC → X2 的反馈闭环；明确相机、坐标、控制周期、时延和失效停止。
4. 与上海现场的具体 X2 型号、固件、MC/AimDK 版本及接口权限对齐，再进行现场适配。

既有 `BodyVelocity` 是 FLU 坐标的 SI 速度目标，不能不经核对直接当作 AimDK 参数。平衡和步态由官方 MC 负责；MaleCNS 的任务输出需要独立验证。单轴稳向实验已停止作为接入前置，结果见[历史实验](EXPERIMENT_RESULTS.md)。当前受控STOP/FORWARD适配器已通过；神经输出接入与真机验证尚未完成。
