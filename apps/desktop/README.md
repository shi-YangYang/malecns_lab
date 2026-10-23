# MaleCNS Lab 桌面工作台

左侧显示官方 X2 GPU 仿真，右侧显示同次脑活动；已封存记录可以回放。启动应用只加载已有胞体坐标，不自动启动仿真或完整脑。

## 安装与打开

需要 Node >=22.12，在本目录执行：

```sh
npm ci
npm run build
npm start
```

已有构建直接 `npm start`。构建复制本地 `apps/atlas/public/data/atlas.json` 的坐标和来源，不分发模型或实验记录。浏览器 preview 只能查看界面。

当前任务按钮存在无动作问题，原因尚未定位；训练和实测按用户要求暂停。旧纵向运行记录不证明当前多行为链路可用，实际结果见 [history.md](../../history.md#spec-024)。

## 候选配置与运行

主进程固定启动 `malecns_lab.runtime.x2.behavior`。宿主会话与 GPU 生命周期在 `src/malecns_lab/runtime/x2/`，容器内 ROS/AimDK 协议在 `experiments/`。[GPU 部署说明](../../deploy/x2/README.md)是环境维护入口；私有安装仍在 `data/x2_sim/gpu_probe/runtime-rebuild`，容器名 `malecns-x2-gpu-trial`，画面端口 6081。

操作者在本地 `data/x2_behavior_desktop.json` 指定实际 checkpoint（不入 Git）：

```json
{"mode":"qualification","checkpoint":"data/<已登记运行>/<实际候选checkpoint目录>"}
```

路径必须位于项目 `data` 内，经过完整性、策略和输入身份检查；缺少配置明确失败，不回退旧三轴模型。当前模式只是候选验证，右挥手因取消资格缺失保持关闭。renderer 不能选择任意路径或直接发送 MC 命令。

后续获准运行时，准备成功后按 Pause、Reset，确认直立且暂停后 Run，再点击“开始”。任务按钮只发送任务枚举，“结束互动”进入脑；“紧急停止”或关闭窗口独立请求停车并等待资源清理，没有最终确认仍显示失败。不得以进程退出当作停车。仿真不用时关闭以减少发热，不退出 Docker Desktop 或停止 Noria。

默认限制：总墙钟 600 秒、GPU 共驻留 240 秒、清理预留 35 秒、最多 50 次决策，每请求最多 8 次决策/120 秒，系统至少预留 1 GiB。无实际相机或外部目标时保留缺失视觉；身体反馈取官方 odom/MC，不补造成功标签。

## 脑活动与回放

- 脑图按 manifest 的真实 body_id 映射；点云颜色是群组均值，栅格显示选定 32 个神经元的真实 spike。横轴是模拟时间，每批通常 5 ms，不等于控制周期；隐藏时暂停拉取，显示跳批和采集丢批分开记录。
- 记录选择器只列固定目录中已确认资源清理的终态实验。`telemetry/manifest.json` 保存图、checkpoint、神经执行身份与来源；回放隐藏实时 noVNC，未记录部分不补造。
- `simulation_capture/` 按同机主机时间对齐采样画面、任务/状态事件与脑批次。只选不晚于游标的数据；画面超过 1 秒或有空档显示“此处未采集”，脑批次超过 2 秒显示“上一批”。noVNC 截图可能包含断线提示，截图成功不证明像素新鲜。
- `simulation_video/` 为无音频 VP8 WebM；终态记录可切换采样/视频。首帧媒体时间锚点尚未实测，`media_anchor=null`，视频模式显示“视频未对齐”并清空配对脑批次；联合查看使用采样模式。

## 采集与安全约束

| 项目 | 固定边界 |
|---|---|
| 视口采样 | 仅本应用 noVNC，2 Hz、单在途、JPEG 质量 70；最长 240 秒/480 张，最大 1280×800；单帧 512 KiB、图片总量 256 MiB |
| 原生视频 | 仅唯一可见 noVNC canvas，15 fps 请求、2.5 Mbit/s、每秒分块；最长 240 秒、单块 8 MiB、未确认数据 16 MiB、文件 256 MiB |
| 档案预算 | 图片/视频各自索引上限 4 MiB，启动至少留 512 MiB 磁盘；遥测预算 16 MiB 不变 |
| 封存 | 停车优先，图片/视频共用最多额外 2 秒；缺末块、stop 确认或完整提交保持 partial，不补造 summary，不接受跨 run 迟到数据 |

renderer 无 Node/shell 或任意文件权限；preload 只暴露固定操作。noVNC 使用独立内存 session 的 WebContentsView，只允许 `127.0.0.1:6081` 的 HTTP/WebSocket，禁止导航、弹窗、下载和权限。视频开始前校验四个 noVNC 文件 hash；不采集整屏、不申请系统录屏权限。记录读取限制固定 ID、整数帧号、符号链接、文件大小、身份与 SHA-256；视频使用固定同源 Range 地址。

隐藏、断线、重载、写入失败或预算超限会产生空档/终止；恢复须确认资源清理后新建 run。采样与录制的边界测试不能代替真实截图、编码、解码、同步、共驻留性能或动作验收。检查按变更选择 `electron/*.test.mjs` 中相关文件，不默认重跑所有测试。历史验证范围统一见 [history.md](../../history.md#spec-022)。
