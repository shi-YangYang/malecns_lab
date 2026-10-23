# M5b 连续视频：受限 noVNC canvas 录制可行性

日期：2026-10-23。状态：**静态可行，尚未实施或实测**。本次仅核对本地源码与官方 API 并新增此报告；没有执行页面脚本、启动 Electron/仿真/脑、采集视频、运行测试或 build。M5a 的“不注入 noVNC 脚本”仍有效，本报告不改变其范围，也不接管其实施者文件。

## 建议结论

建议独立登记 M5b：为现有 simulation `WebContentsView` 增加**专用、隔离、沙箱内的 preload**，仅对该视图中实际显示 VNC 像素的 canvas 调用 `captureStream()`，由 Chromium 自带 `MediaRecorder` 编码，主进程通过有界专用 IPC 持续写入本次 run。这个路径无需 `desktopCapturer`、`getDisplayMedia`、整屏权限、FFmpeg 或新依赖，也无需修改 SDK、MC、Sim 或 noVNC 的 JS 文件。

这是一条有源码和 API 依据的实现候选，不是已能录制的结论。增加 preload 是新增受信任代码和 IPC 面，必须作为 M5b 单独设计、实施和独立审查；不能以“不改 noVNC 文件”为由规避 M5a 当前禁止注入的范围限制。

与 `desktopCapturer` 相比，canvas 路径可明确限定像素对象；后者列举屏幕/窗口源，官方文档不保证独立 `WebContentsView` 是可选窗口，macOS 屏幕采集另需用户同意。此项目的最小路线无需扩大到屏幕源。[Electron desktopCapturer](https://www.electronjs.org/docs/latest/api/desktop-capturer)

**像素边界：**录制的仍是本应用 noVNC canvas 内的专用 X11 虚拟桌面；其中若出现终端或其他远端窗口，也会进入视频。它不采集用户其他 macOS 应用，但也不能称为只有 MuJoCo 场景的离屏渲染。noVNC HTML 工具栏、断线弹窗等 canvas 外层元素不会进入该视频，必须另外归档状态事件；M5a 页面截图仍有不同用途。

## 本地依据及其限制

- `apps/desktop/electron/main.mjs` 当前 simulation 没有 preload；它使用独立非持久 partition `malecns-simulation`，`nodeIntegration=false`、`contextIsolation=true`、`sandbox=true`、`webSecurity=true`、`backgroundThrottling=true`。权限请求/检查全部拒绝，弹窗、导航、重定向、webview 和下载受限。主界面的 `preload.cjs` 不能直接复用给 simulation，否则会把实验控制能力带入 noVNC 页面。
- `boundaries.mjs` 的网络限制是精确 loopback 主机与端口（HTTP/WS，127.0.0.1:6081），不是精确页面路径；现有主界面 IPC `trustedFrame` 则只认 `workbench://app/index.html`。新视频 IPC 需要另立 simulation 发送者检查，不能放宽原主界面检查。
- 本地 Electron 依赖为 45.0.0。`deploy/x2/desktop.sh` 将 `/usr/share/novnc` 交给 websockify，Xvfb 为 1280×800。GPU 构建草案以 `malecns-lab-x2:local` 为基底，但草案不证明当前 GPU 容器字节相同。
- 本次先只读确认默认 Docker 中 `malecns-x2-x2-1` 为 **exited**，再用 `docker --context desktop-linux cp …:/usr/share/novnc -` 将文件流读入内存核对；没有启动它。GPU 私有 VM 关闭，未为取源码开启。下表是该停止容器的实际文件，不冒称当前 GPU 容器来源已核实。

| 停止容器内文件 | 实际实现与 SHA-256 |
|---|---|
| `/usr/share/novnc/app/ui.js` | 第 1018 行将 `#noVNC_container` 交给 RFB；`49dd68cd630ba75cca74602b8950ed84d14da39458819387c0ef76880d4c9172` |
| `/usr/share/novnc/core/rfb.js` | 第 156–165 行动态创建无固定 ID 的可见 canvas；第 183 行交给 Display，第 413 行挂入容器，第 436 行断开时移除；`708f8abd828b5b45c0cfdf101c51c39506a4210e2c5af5dd605493e67ac7c9cd` |
| `/usr/share/novnc/core/display.js` | 第 47 行取得可见 canvas 的 2D context，另建隐藏 backbuffer；第 293 行将 backbuffer 画到可见 canvas；网络解码数据通过 ImageData 或 data 图像进入；`768bb889ffbfb8c3db327be0c7b9b6039f80b59bd192f4424aeaa358905678de` |
| `/usr/share/novnc/vnc.html` | 页面入口；`720ee6e3090aa60863e8133d5ccc8f3aa3ca2a26513855a0fc3ff42f5da754ae` |

因此不能硬编码 `#noVNC_canvas`，也不能录隐藏 backbuffer 或全页第一个任意 canvas。建议专用 preload 在 DOM 就绪后，等待 `#noVNC_container` 下**唯一、已连接、尺寸非零**的 HTMLCanvasElement；缺失、多个、替换或尺寸越界均拒绝/终止，而非猜测另一个来源。首次实际运行须核对当前 GPU 服务的对应文件身份与 DOM。

`captureStream()` 输出 canvas 位图的 MediaStream；来源被污染时会抛 `SecurityError`，后续污染还可能使 track 静音。本地所见绘制路径支持 origin-clean 的可能性，但静态阅读不能证明运行中从不污染；禁止关闭 webSecurity 或绕过该检查。[MDN captureStream](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream)、[W3C canvas capture](https://w3c.github.io/mediacapture-fromelement/#html-canvas-element-media-capture-extensions)

## 最小隔离与 IPC 方案

1. 新增独立 `.cjs` simulation preload，保留全部既有隔离设置。preload 可访问 DOM 和有限 Electron API；沙箱不提供普通文件系统/子进程能力，也不适合直接加载 Node ESM。使用 Web API 做编码、`Uint8Array` 做二进制传输，写盘留在主进程。[Electron preload](https://github.com/electron/electron/blob/main/docs/tutorial/tutorial-3-preload.md)、[Sandbox](https://www.electronjs.org/docs/latest/tutorial/sandbox)、[ESM](https://www.electronjs.org/docs/latest/tutorial/esm)
2. preload 不向 noVNC 主世界暴露 `contextBridge` API，不接受页面 `postMessage`、自定义 DOM 事件或页面可设置的参数作为开始/停止/路径。仅监听主进程发送的固定录制消息。不读取页面的 `UI.rfb`、不 monkey-patch RFB/绘图方法、不调用 `executeJavaScript`；通过隔离世界原生 DOM 接口持有已验证 canvas。页面仍能影响其自身像素，因此视频是显示证据，不能成为行为成功的权威判据。[Electron Context Isolation](https://www.electronjs.org/docs/latest/tutorial/context-isolation)
3. 主进程确认当前 run 身份、资源归属、可见性及准入后，向**当前 simulation 主 frame**发出带 run generation/随机 token 的开始消息；页面 preload 自行启动没有资格。每条返回消息均验证 `event.sender === simulation.webContents`、`senderFrame === mainFrame`、完整规范化入口 URL、当前 document generation、token、序号、类型和大小。URL 必须匹配固定 `/vnc.html` 与登记查询参数，不能只凭 6081 同源；导航/重载立即撤销 generation，迟到 chunk 不进入下一 run。token 不替代发送者验证。[Electron security](https://www.electronjs.org/docs/latest/tutorial/security/)
4. 二进制先由 Blob 转为有界 ArrayBuffer/Uint8Array，再走具体 chunk IPC，禁止传 DOM 对象、Blob/File、MediaStream 或 `ipcRenderer` 本身。IPC 只接受固定元数据和二进制，主进程选择既定 run 目录；无任意路径、URL、capture 对象、命令或新增任务能力。主进程写入完成才 ACK，preload 释放相应引用；不使用同步 IPC。[Electron ipcRenderer](https://www.electronjs.org/docs/latest/api/ipc-renderer)
5. 保持所有媒体设备/显示权限拒绝、原网络白名单和导航限制。此路径不调用设备/屏幕采集 API；若实际运行出现权限门槛、编码错误或 origin 错误，明确失败，不能自动改走 desktopCapturer。

## 编码、预算与封存建议

以下是供 M5b 契约冻结的一组最小参数，**尚未采纳为运行配置或测得性能**：原 canvas 尺寸不超过 1280×800，不另建缩放/补帧 canvas；`captureStream(30)`、单一视频 track、无音频；固定尝试 `video/webm;codecs=vp8`，码率请求 4 Mbit/s，`start(1000)` 分块。先查 `MediaRecorder.isTypeSupported`，不支持则停止登记，不静默切换编码。即使返回 true，资源不足仍可编码失败；VP8 是否使用硬件、实际帧率、温度和与 MPS 共驻留的开销均未验证。[MDN isTypeSupported](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/isTypeSupported_static)

建议沿用本轮最多 240 秒、不增加 VM 时间；单 blob 8 MiB、应用未确认数据合计 16 MiB、文件 256 MiB、索引 4 MiB、磁盘预留 512 MiB。先检查 Blob.size 再转 ArrayBuffer；每个 generation 最多一个待写 IPC 和一个待发送块，超额终止并保留不完整档案，不丢块后宣称完整。码率只是请求，timeslice 也不是块大小或墙钟上限，浏览器仍可能内部积累较大编码缓冲；这些限制约束本应用接收/排队/落盘，不能声称严格限制了 Chromium 编码器内存。[MDN dataavailable](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/dataavailable_event)

新增独立 `simulation_video/` schema 和额度，绑定 M5a 同样的 run/神经身份、官方资产、录制源码、noVNC 文件来源及 Electron/Chromium 版本；不改变旧遥测或采样档案。以排他新建的 `recording.webm.partial` 顺序追加原始编码块，索引保存序号、偏移、长度、hash、源事件时刻和主进程接收/写入时刻。每块不一定可单独播放；只有完整 recorder 输出的全部块合并后才有规范上的可播放要求，不能把每个 blob 命名成独立完成视频。[W3C MediaStream Recording](https://w3c.github.io/mediacapture-record/#mediarecorder-methods)

隐藏、最小化、canvas 移除/换尺寸、track mute/ended、页面异常、写盘/ACK 超时或预算到界都结束当前录制并记原因。最小版本不 pause/resume，不自动拼接跨 gap 的视频；恢复需新段/新明确登记。不要为后台录像关闭节流、保活屏幕或延长 GPU 工作时间。

正常控制结束、紧急停止或关窗先发既有停车/资源清理，再请求 `MediaRecorder.stop()` 并停止媒体 track；等待最终 dataavailable、stop 和所有 ACK 后才封存，完成条件与机器人停止条件分别记。封存最多占用既有额外 2 秒；超时/崩溃保留 partial、原因和已写块，不能把资源清理卡在编码器上。最后原子更名/写 summary，缺少末块、序号不连续或超时都不能标 complete。[MDN stop](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/stop)

## 时间锚点与缺帧：能承诺到哪里

- `captureStream(30)` 是采集请求率，不证明 30 个新物理帧/秒，更不证明 60 fps。canvas 只在有更新且满足捕获条件时产生帧；VNC 延迟、编码掉帧和静止内容都会影响结果。不得从 blob 数量、rAF 次数或 2 Hz JPEG 推导编码帧率；M5a 的 JPEG 不用于生成这段“连续原生录制”。[W3C canvas capture](https://w3c.github.io/mediacapture-fromelement/#html-canvas-element-media-capture-extensions)
- preload 的 `performance.now()`、MediaRecorder/BlobEvent 媒体时间、Node `process.hrtime.bigint()`、Python 主机单调时间和 Linux MC 时间并非可直接互减的同一数轴。主进程与 preload 要用带 nonce 的往返采样保存两端时刻、偏移估计和往返不确定区间，开始/期间/结束均留证；新主机纳秒用十进制字符串。renderer 重载重建时钟身份。[MDN performance.now](https://developer.mozilla.org/en-US/docs/Web/API/Performance/now)
- 记录 start 调用/事件、每个 BlobEvent 的 `timecode`（若实现提供）、事件 `timeStamp`、接收时间及 stop。`timecode` 相对于该 recorder 首批数据，不是物理仿真时间；`start()` 调用也不是第一帧已编码的证明。没有第一帧到主机时间的已验证锚点时，只可给出主机时间范围/近似关联，不能用零偏移制造精确神经同步。可在实际验证中使用同流的本地 video 帧回调核对 mediaTime 与呈现时刻，但它也只是本地呈现，不是 Sim 源帧时刻，且增加开销，不作为本次静态结论。[MDN BlobEvent.timecode](https://developer.mozilla.org/en-US/docs/Web/API/BlobEvent/timecode)、[MDN requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)
- 单靠 MediaRecorder 事件不能区分“场景静止”和“VNC 上游冻结”，也不能得出全部编码掉帧数。能够确认的 disconnect、canvas 移除、mute、隐藏、超时均记录 gap；其他源新鲜度保持 unknown。真实测试需对照同次可见运动与 MC/odom 原始事件，不能以视频存在替代运动证据。
- 首版验收分别报告主机持续时间、可解码媒体时长、实际可观察帧/间隔、已知 gap、锚点不确定度与未知项；不假设 WebM 的 duration/seek 元数据一定完善。若媒体缺可验证时间锚点，保留视频单独播放与原始事件查看，联合游标不宣称精确同步，也不把 M5 同步验收判 PASS。

## 回放与实施前必须冻结的边界

当前 `index.html` 没有视频 `media-src`，workbench 协议只服务已列出的静态素材。视频回放不能简单把整个数百 MiB 文件当 IPC 返回、塞成 data URL，或加入任意 file 路由。M5b 应另定**只读、同 workbench origin 的受限媒体 URL**：只接受有效 record ID/固定视频名，真实路径/符号链接/封存/hash/身份检查，GET/HEAD 和单一有界 Range，严格 MIME、无目录列举，不接受 renderer 指定路径。沿 `media-src 'self'` 即可，不增加外网/任意 blob/file 权限；实现该窄路由是 M5b 新的 S3 边界，不属于 M5a 原实现。

新视频路径与 M5a 要由一个实施者在其工程审查完成后整合，避免并改 `main.mjs`、controller、preload 或关闭生命周期。只修改桌面子系统和相应规范；M1/M3、训练来源、SDK/MC/Sim 不变。文件 hash/身份不匹配时仍拒绝，不从旧 run 借用画面。

后续必要验证为一个桌面定向检查（发送者/URL/frame/generation、块大小与背压、序号/末块、写入故障、迟到数据和停止优先），TypeScript 改动才加一次现有 build；不重复 M5a 已通过用例。工程通过后，仅在训练结束和 M2/M3/M4 准入满足的已登记真实运行内录制一段，再关闭仿真离线验证播放、同次身份、时间限制、断线/关闭档案及资源开销。没有实际解码的真实连续视频，不能标 M5b 完成；P5 任务评分与上海真机仍分别验收。

本报告为 S0：仅检查新增文档内容，不运行程序验证；实际 GPU noVNC 来源、origin-clean、MediaRecorder 编码/停止行为、帧率、播放与时间锚点都是明确待实测项。
