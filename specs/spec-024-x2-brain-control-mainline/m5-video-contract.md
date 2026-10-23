# M5b 同次原生视频录制与回看

状态：Ready；**仅在M5a独立工程PASS后开始实施**。2026-10-23。依据[m5-video-feasibility.md](m5-video-feasibility.md)。风险S3，文件范围仅桌面子系统和实施报告；不改Python、脑模型、SDK、MC、Sim或noVNC源码，不改运行中的训练来源。实际GPU运行另登记，不能在当前完整脑训练期间启动。

## 目标与来源

补齐实际视频文件及离线回看，不能用M5a的JPEG序列拼装冒充原生录制。使用本应用专用simulation WebContentsView的隔离沙箱preload，对`#noVNC_container`下唯一、已连接、尺寸非零且不超过1280×800的可见canvas调用captureStream；禁止任意canvas、隐藏backbuffer、整屏/系统窗口录制和自动来源回退。canvas替换/尺寸变化/污染/移除即结束并保留原因。

视频像素来自专用VNC虚拟桌面，可能包含其中的官方控件或终端；不含外层noVNC断线提示。来源和断线状态均明确归档，不称纯场景渲染或60fps录像。M5a独立采样仍承担同次主机时间回放；视频不是脑或动作成功的判据。

实际运行前，从已经启动的固定6081服务有界读取vnc.html、app/ui.js、core/rfb.js、core/display.js，记录hash并核对可行性报告中的来源；每文件最多1MiB，单请求2秒，总来源校验最多5秒。不为读取而启动任何额外资源。来源不同则明确拒绝本版录制，不猜DOM或解除webSecurity。

## 唯一编码与预算

固定请求`captureStream(15)`、一条视频track、无音频；MIME为`video/webm;codecs=vp8`，请求2.5Mbit/s，MediaRecorder按1000ms分块。isTypeSupported为false或实际编码失败均终止，不静默切编码/改为整屏捕获。15fps只是请求，实际帧率与编码器实现待实测。

最长240秒、不延长现有VM时间；单块8MiB、应用未确认数据16MiB、视频文件256MiB、索引4MiB，开始至少512MiB磁盘预留。最多一个在途写IPC和一个待发块，写入ACK后释放；队列/块/磁盘超限即结束并明确不完整。该预算不宣称严格约束Chromium内部编码缓冲。

仅在本次run身份已核对、控制运行且视图可见时开始。隐藏/最小化、disconnect、track mute/ended、canvas异常、页面重载、ACK超时、写盘错误均结束当前段；首版不pause/resume、不自动拼接跨gap、不后台保活。恢复需新run，原档案保留。

## 隔离、归档与退出

- 新建独立simulation preload，不复用主界面控制preload，不向页面主世界暴露API，不接受页面postMessage/DOM事件驱动录制；不使用executeJavaScript，不改RFB/绘图方法。
- 保持nodeIntegration=false、contextIsolation=true、sandbox=true、webSecurity=true及原权限/网络/导航/下载限制。主进程固定消息才能授权录制；每条返回检查simulation WebContents、主frame、完整固定URL、document generation、随机token、序号与大小。页面重载撤销授权，迟到块不得进入下一run；不接受renderer提供的路径、URL、命令。
- 独立simulation_video目录和v1 schema，绑定M5a相同的record/run/session/神经execution/checkpoint/policy/input_map、官方资产、noVNC来源、录制源码和Electron/Chromium身份；单技能ID不冒充神经execution。
- 排他创建recording.webm.partial，按序追加原始块，记录偏移/大小/hash及时间。收到最终dataavailable、stop、全部写入ACK才封存；缺块/乱序/超时保留partial和原因，不补造完成文件。半成品不得当完整视频提供联合回放。
- 正常结束、紧急停止、关窗及renderer异常均先走既有独立停车/资源清理，再停止recorder和track。图片与视频封存并行共享最多额外2秒，不能各加2秒或阻塞停车。视频状态不得覆盖机器人或资源状态。

## 时间与回看

记录preload performance时钟身份与主进程nonce往返原值/区间，保留start调用/事件、每块timecode（若有）、事件时刻、主机接收时刻及stop；新主机纳秒为十进制字符串。媒体时间、preload时间、主机时间与远端MC时间不能直接混减。像素没有源仿真时间，不承诺物理帧同步；无法分辨静止与上游冻屏时保留unknown。

完整视频在同一run记录里可选择回看，保留M5a的主机时间采样回放与旧无画面记录。只在具有可核对时间锚点及误差范围时关联脑批次，并只使用可确认时刻之前的真实批次，显示“近似对齐”及范围；没有锚点则明确只回看视频，不能用假零偏移把脑数据拼上。无论哪种路径都不插值脉冲、不混入实时noVNC或其他run。原生视频的实际解码、时长/seek、锚点与已知gap需真实验收，静态API可用不能代替。

不恢复大标题、底部播放条或历史小字；在现有面板顶端使用必要的画面类型选择和播放/暂停。暂停/切记录即停播放；实时运行时不能打开历史视频或发历史任务。

视频使用受限同源workbench媒体路径，固定record ID和文件名；仅封存且身份/hash核对的档案可读，逐级拒绝符号链接/路径越界。GET/HEAD、单一合法Range与正确长度/MIME，按有界流返回，禁止整文件base64/任意file路由/目录列举。CSP仅补`media-src 'self'`，不给bypassCSP或外网权限；按[Electron协议文档](https://www.electronjs.org/docs/latest/api/protocol)为现有scheme声明stream能力。

## 验证与停止

由一个实施者承担全部耦合桌面变更。一个相关Node检查命令覆盖独立发送者/URL/frame/generation、来源/身份、背压/块顺序/末块、迟到数据、停止优先、受限媒体读取及Range；使用注入编码数据仅验证契约，不冒称真实视频。修改TypeScript/入口后运行一次现有构建，失败限定修复；不机械复跑M5a已经通过且未改变的检查。

新的独立审查者审阅代码和真实检查；工程PASS后停在待实测。训练退出、M2/M3/M4相关准入满足后，由协调者把一段真实录制并入已登记的短M4运行，结束即关闭仿真，然后离线验证原生视频解码、同次身份、时间限制和资源开销。M5a/M5b、P5任务成功和M6上海现场分别记录；不得只因视频文件存在而宣布主线完成。
