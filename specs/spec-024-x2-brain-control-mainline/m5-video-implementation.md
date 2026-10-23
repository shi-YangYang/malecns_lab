# M5b 原生连续视频实施报告

日期：2026-10-23。实施者：`roadmap_m5_video_implementation`。状态：**桌面工程实现与限定检查完成，待新的独立审查；没有实际视频或同步验收结论。** 前置为 M5a 最终独立 PASS。风险 S3（隔离 preload、IPC、持久化封存、媒体读取协议与退出交互）。

## 修改范围

本轮只修改桌面子系统与本报告，保留原有未提交工作。新增：

- `apps/desktop/electron/simulation-preload.cjs`：独立隔离沙箱内的原生 canvas/MediaRecorder 采集。
- `apps/desktop/electron/video.mjs`：来源与身份、写入、时钟证据、封存、读档与有界媒体响应。
- `apps/desktop/electron/video.test.mjs`：定向 IPC、归档、媒体及隔离 preload 契约检查。

接线改动：`electron/main.mjs`、`electron/controller.mjs`、`src/main.ts`、`src/types.ts`、`index.html`、README。`electron/recording.mjs` 只把 `seal/finalize` 接受的绝对截止时刻传入已有封存流程，使 M5a/M5b 共用同一 2 秒预算；原 pending marker、批次绑定与游标逻辑不变。未改 Python、模型、SDK、MC、Sim、noVNC、依赖或锁文件；未改运行中训练来源，未 commit/push。

## 实际实现

专用 preload 无 contextBridge 暴露、无页面 postMessage/自定义事件入口、无 executeJavaScript 或 RFB/绘图方法改写。主世界无控制或文件 API。保留全部 sandbox/contextIsolation/webSecurity、导航/权限/下载/网络边界。只能使用 `#noVNC_container` 下唯一可见且已连接的原 canvas，要求 noVNC connected 状态、尺寸 1–1280 × 1–800；替换、移除、尺寸改变、隐藏或 track mute/ended 结束本段。

固定请求 `captureStream(15)`，一个视频 track、无音频，`video/webm;codecs=vp8`、2.5 Mbit/s、1000 ms timeslice；不支持或实际编码错误均终止，不回退其他来源/编码。页面只能影响自身显示像素，不能据视频断言动作成功；该像素仍可能包含 VNC 虚拟桌面的控件或终端。

主进程每次录制前从固定 6081 读取四个源文件，核对可行性报告中的 SHA-256；逐文件最多 1 MiB/2 秒、总校验最多 5 秒、拒绝重定向。录制身份来自同次 registration/ready/telemetry/official assets，额外绑定 noVNC、录制源码、Electron/Chromium。每条返回都核对专用 WebContents、mainFrame、完整入口 URL、document generation 与随机 token；事件序号、大小、类型、时钟身份单独检查。重载撤销旧授权，旧块不会进入新 run。

最多一个在途写入与一个待发块；每块 8 MiB、应用未确认数据 16 MiB、视频 256 MiB、索引 4 MiB、最长 240 秒、磁盘至少 512 MiB。ACK 仅在原始编码块与元数据写入后返回；ACK 超过 1500 ms 或队列超界即停止并标不完整。该额度不保证 Chromium 编码器内部缓冲上界。

`simulation_video/recording.webm.partial` 排他创建并顺序追加，`events.jsonl.partial` 持续保存已确认事件/块位置、大小、hash 与时间。只有收到最终 dataavailable、stop、全部在途写入结束、身份再核对后才发布完成文件/索引/summary。durable `seal_pending.json` 始终优先于 summary；摘要已进入异步 link 后跨过期限也不能变完整。失败保留 partial、未完成标记与 best-effort 原因/事件/时钟记录；不删除原始资产。

控制停止先于 recorder stop。主进程在控制终态/资源关闭前先停止媒体录制，以免把正常清理导致的视图隐藏当成主动录制断线。独立停止、关窗、renderer 异常仍先请求既有控制停车；封存只在控制清理确认后等待。图片与视频并行使用同一绝对截止时刻，最多额外 2 秒，不能各自增加 2 秒。视频状态不修改停车或资源确认。

回看路由仅固定 `workbench://app/media/<record-id>/recording.webm`，资源未清理时拒绝访问；只接收 GET/HEAD、单一合法 Range，错误范围 416，无任意路径、查询、目录列举、file 或 base64 视频。读取逐级拒绝符号链接、验证身份/索引/所有块与整文件 hash；同一个文件描述符提供输出，并在输出每个编码块前再检查 hash，避免校验到播放期间内容改变。内存最多一个有界编码块，流输出每次最多 64 KiB。现有 scheme 仅增加 stream 能力，CSP 仅增加 `media-src 'self'`。

左侧面板顶部增加“采样／视频”和播放/暂停，保留两侧布局，不恢复大标题、底部播放条或历史小字。切记录/切模式/隐藏暂停，运行时不能打开历史视频或发送历史任务。原有采样回放及没有视频的旧脑记录仍保留。

## 时间能力与明确限制

记录随机 nonce、host send/receive 纳秒字符串、preload performance.now/timeOrigin/clock ID，保存 start 调用/事件、每块 timecode（若有）、事件时刻/host 接收/写入与 stop。开始/运行期间/结束均请求往返取样，原始时间保持独立域；pixel/source freshness 为 unknown。

**本实现不能从 MediaRecorder 的 start/dataavailable 事件证明第一编码帧的媒体时间对应哪个主机时刻。** 因此 `media_anchor` 和 `alignment` 固定为 null，不以假零偏移或首个块时刻构造锚点。视频模式明确显示“视频未对齐”，清空右侧配对批次；需要同次主机时间记录时选择原 M5a 采样。契约的“无锚点仅视频”分支已实现；**有真实可核对锚点后的“近似对齐＋误差范围”联合视频播放尚未实现，不能据此标 M5 同步 PASS。** 协调者已被明确通知这个边界，后续须依据真实采集的锚点方案再限定实现/验收。

真实编码器是否支持、实际编码方式/帧率、WebM 解码/时长/seek、可观察运动、断线、首帧锚点、录制与 MPS 共驻留开销，均未实测。fixture 字节不是可播放视频，更不是仿真行为证据。

## 实际检查

1. 首个相关 Node 命令：

   `node --test --test-name-pattern='video|late capture after bounded seal|summary publication already in IO' electron/video.test.mjs electron/recording.test.mjs`

   结果 **7/10 PASS，3 FAIL**。两项 M5a 超时回归通过；本轮修改触及其 seal 截止参数，因此只运行这两项相关检查，未重复其他已通过 M5a 用例。三项失败分别为测试误用 `assert.rejects` 捕获同步抛错，以及两个 fixture 在 best-effort 不完整记录尚在写入时删除临时目录导致 ENOTEMPTY。产品错误路径并未被这些结果判成功。

2. 限定返修时一次编辑命令误用了工作目录，修改未应用，随后限定命令复现相同三项失败（4 项中 1 PASS/3 FAIL）；该次不是新的产品故障，也没有据此扩大检查范围。

3. 修正同步断言、让测试清理等待可观察的不完整写入 Promise，并补实际隔离 preload 背压拒绝用例后，限定运行：

   `node --test --test-name-pattern='video archive binds|video refuses wrong|video failure requests|video isolated preload|video preload permits' electron/video.test.mjs`

   结果 **5/5 PASS**。来源/发送者/Range/晚摘要等先前已过且无相关修改的用例没有机械重跑。合计 11 个不同相关用例各有通过证据，**不是声称重新完整运行 11/11**。

4. 一次现有 `npm run build`：prepare:assets、`tsc --noEmit`、Vite **通过**；9 模块，Vite 455 ms。没有新增或升级依赖。构建只准备已有图谱静态资产，没有启动脑。

5. 只读检查当前 diff、退出顺序与类型/UI 接线。文档补充属 S0，不再追加 build 或测试。

没有启动 Electron、私有 GPU VM、旧 CPU 仿真、MPS/完整脑或硬件；没有停止 Noria 或改变 Docker。实际训练运行不受本实施调用影响。

实施在此停止，交由新的独立验收者审查工程边界与上述时间能力限制。真实 M5a/M5b、P5 行为与 M6 上海真机分别待验证；本报告不替代这些验收。
