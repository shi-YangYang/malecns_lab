# 当前交接 — GPU 仿真运行中；CPU 仿真已停止

更新：2026-10-23。工作目录 `/Users/yang/project/malecns_lab`；运行命令显式指定工作目录。

## 当前任务

2026-10-23 按用户最新要求切换到 GPU 并保持运行：CPU 容器 `malecns-x2-x2-1` 已停止；私有 GPU VM、`malecns-x2-gpu-trial`、Sim 和 MC 已启动，浏览器画面已确认，MC 查询返回 `PASSIVE_DEFAULT / 100`。当前桌面为 `http://127.0.0.1:6081/vnc.html?autoconnect=true&resize=scale`。本次未重新测量 FPS，也未发送运动指令。启动与查询日志在 `logs/x2_docker/20261023_gpu_switch.frGXfA/`，组件日志在 `data/x2_sim/agibot/data/var/gpu-switch-20261023-frGXfA/`。

此前修复结果：

用户要求继续推进 GPU。外围图形栈已重建并修好，**原版 X2 已在 Apple M5 GPU 路径上正常显示**：`zink Vulkan 1.2 (Virtio-GPU Venus (Apple M5))`，Info **60 FPS**；同一 GPU 容器内 MC 查询返回 `PASSIVE_DEFAULT / 100`；成功运行无 `ring fatal`/`Aborted`，缓存重建后再次启动正常。详见 [AGIBOT](../../docs/AGIBOT.md) 的「GPU 图形兼容层重建」与 [Spec 015](../../specs/spec-015-x2-docker/spec.md)。

修复的五个问题：

1. SPIRV-Cross 的 `sampler` 保留字冲突；补丁后 MSL 使用 `sampler0`。
2. 静态库漏编 `spirv_reflect.cpp`，`CompilerReflection` 未定义。
3. Lighter 链接 MoltenVK 必须带 `-ObjC`，否则 ObjC category 未加载，VM 创建 sampler 时崩溃。
4. Zink 的 fragment 分量 varying（同一 location 的 `float4` + `float`）；补丁启用 fragment 输入的分量打包，真实转储着色器变为单个 `float4 m_location_1`。
5. Mesa/zink 磁盘缓存保存了补丁前 MSL；清空一次 `/root/.cache/mesa_shader_cache*` 后重新生成，之后无需再清。

## 环境与复现

- 正式 CPU 部署 `deploy/x2/`，8 CPU/6 GiB、4 个 llvmpipe 线程；桌面 `http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale`。
- CPU 容器 `malecns-x2-x2-1` 当前已停止；`final-cpu-query.log` 是上一轮恢复时的 `PASSIVE_DEFAULT / 100` 历史记录。
- GPU 使用独立运行时 `data/x2_sim/gpu_probe/runtime-rebuild/`（独立 `LIGHTER_HOME`，rootfs/data.img 为 APFS 克隆），未替换已安装签名 app bundle。GPU 容器 `malecns-x2-gpu-trial`，4 GiB，端口 6081。
- 按用户最新要求保持 GPU 运行、CPU 停止；默认 Docker context 仍为 `desktop-linux`，未操作无关服务。
- 复现命令与一次性缓存清理见 [AGIBOT](../../docs/AGIBOT.md) 的 GPU 一节。

## 证据与回滚

- `logs/x2_docker/20261022_gpu_rebuild/`：SPIRV 补丁与回归、分量 varying 转储对比（`component-fix-fs.metal`）、缓存 backtrace、成功/重复启动日志、MC 查询、最终散列（`final-artifact-hashes.txt`）、`compiled-msl/`。
- 成功截图在 `data/x2_sim/agibot/data/var/`：`x2-gpu-componentfix-1.png`、`x2-gpu-fps-info.png`、`x2-gpu-ui-effects-off.png`、`x2-gpu-cleancache.png`、`x2-gpu-clean-fps.png`、`x2-gpu-warmcache.png`。
- `data/x2_sim/gpu_probe/rebuild/`：SPIRV-Cross 源码与两处补丁、MoltenVK 构建树、Lighter 源码与产物；`runtime-rollback/` 保留原安装二进制副本。
- 最终散列：SPIRV `2b1bd3...`、MoltenVK `0a7469...`、Lighter `b90247...`。二进制含调试探针（转储、backtrace、跳过缓存的 env），不是发布产物。

## 已确定方向与边界

- 设备为智元 **X2**，官方 Sim + MC + AimDK；随后赴上海现场适配。
- 官方 SDK、MC、Sim、模型与物理参数未修改；GPU 路径仍是第三方兼容实验，不代表现场性能。
- 被动场景：没有发送站立、行走或速度指令，没有 MaleCNS→AimDK→MC 闭环。
- 不采用第三方运动控制器；下一步先验证官方站立/运动/停止，再接 MaleCNS。

## 环境与约束

Noria 不得停止，Docker 不得为本项目重启或更换后端。GPU VM 与 CPU 仿真容器不同时运行；当前按用户要求保持 GPU 运行。默认 context 不改；完整脑任务遵守 1 GiB 系统预留，不与仿真并行。

## 验证与提交

此前重建按 S2：补丁离线回归、全栈重建、原版 X2 GPU 画面与 60 FPS、MC 查询、缓存重建后的重复启动、CPU 恢复查询。未做全仓测试或训练。按用户要求未再送独立验收；此前一次只读审查指出的记录问题已补证据并更正。

本次仅切换已有运行环境：CPU 停止、GPU 组件启动、浏览器画面和一次 MC 查询均已确认；交接更新为 S0，仅检查 diff，无代码修改、构建或训练。相关文档按用户要求本地提交，未推送；GPU 补丁源码、运行时、二进制和日志仍保留在本地忽略目录，本次提交不包含这些资产。

下一步：在 GPU 或 CPU 环境验证官方站立/运动/停止并记录反馈，再设计 MaleCNS 到 AimDK 的接口。
