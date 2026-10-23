# M5b 原生视频独立验收

日期：2026-10-23。验收者：`roadmap_m5_video_review`，未参与实施。结论：**FAIL，存在 1 项正常停止时的末块判定问题，须限定返工。** 风险仍为 S3（隔离 preload、IPC、持久化与媒体协议）。本次仅新增本报告，没有修改业务代码、Python 或训练来源，没有 commit。

## R1：停止前已排队的 timeslice 会被误判为最终数据

**[P2] 阻断 M5b 工程验收。** `apps/desktop/electron/simulation-preload.cjs:74–77` 用 `a.stopping && a.recorder.state === 'inactive'` 判断当前 `dataavailable` 是否末块。但 `stop()` 同步令 recorder 进入 inactive，`dataavailable` 和 `stop` 随后异步派发；此前已经排队的普通 timeslice 事件也可能在 stop 调用之后才执行。该事件和真正的最终数据都会被标成 `final: true`。

这不是需要编码器故障才能发生的输入。W3C 的 [MediaStream Recording 方法规范](https://www.w3.org/TR/mediastream-recording/#methods) 明确区分同步状态变化与异步事件：timeslice 将数据事件排入任务队列；stop 先使 recorder inactive，再排入最终 dataavailable 和 stop。因此，状态 inactive 不能单独证明某一个数据事件就是末块。

主进程 `apps/desktop/electron/video.mjs:101–107` 在接收第一个误标末块后令 `finalChunk=true`，真正的末块随后会被拒绝为“视频末块或时间无效”。`receive()` 的错误路径会转为 incomplete；完整且正常结束的录制无法封存。当前保护没有把丢失数据误标为成功，但不能因此把正常停止能力判 PASS。

本验收作了两次最小 CPU 复现，未执行浏览器、Electron 或编码器：

1. 使用 `node:vm` 执行**当前生产 preload 文件**，注入 MediaRecorder 事件对象。先启动并确认 started，再调用主进程 stop 命令使状态变 inactive，然后依次派发“先前已排队的 timeslice”、真正最终 dataavailable、stop。每次等待微任务与 ACK，未超过任何块大小或队列预算。实际 IPC 输出为：

   ```json
   [
     { "kind": "started", "sequence": 1 },
     { "kind": "chunk", "final": true, "sequence": 2 },
     { "kind": "chunk", "final": true, "sequence": 3 },
     { "kind": "stopped", "sequence": 4, "final_sequence": 3 }
   ]
   ```

2. 调用当前 `SimulationVideo.prototype.write()`，在第一个错误末块已接受的状态下输入真正末块；`assert.rejects(..., /视频末块或时间无效/)` 通过，证实接收侧确实拒绝它。

现有 `video.test.mjs:107` 的 Recorder fixture 在 `stop()` 方法内部同步派发最终 dataavailable 和 stop，没有覆盖上述已排队 timeslice 的交错；其既有通过结果不证明这个停止边界。

限定修复要求：用最终 stop 事件与其之前的最后数据事件建立末块证据，不能仅凭 recorder 当前状态判定。保留原始块顺序、所有 ACK、既有背压/大小/共享期限及缺末块拒绝规则，不通过放宽“多个 final”来掩盖问题。至少用异步 stop 及已排队 timeslice 的场景贯通生产 preload 和接收侧，确认正常封存；不需要重复来源、媒体 Range 等未变检查，也不要求本次启动真实仿真。按 AGENTS，由新的实施者返工，再由新的独立验收者复核。

## 本轮已审阅且未发现直接阻断的问题

- **采集来源与隔离**：独立 simulation preload，未向页面主世界暴露桥、未接收页面消息驱动操作，未改写 RFB。仅选 `#noVNC_container` 内唯一、连接、可见且尺寸受限的 canvas；固定 VP8、15 fps 请求、无音频，无替代来源。主进程有界读取并核对四个 noVNC 源文件，保留来源未知/断线边界。
- **IPC 授权**：`trustedVideo()` 加 main handler 核对专用 WebContents、mainFrame、完整固定 URL、document generation、随机 token；事件额外核对序号、大小、类型与时钟身份。重载撤销旧授权，未接收 renderer 路径、命令或任意 URL。
- **预算与不完整终态**：preload 保持最多一个在途块与一个待发块、16 MiB 未确认数据、1500 ms ACK 期限；主进程文件/索引/磁盘预算明确。封存先保留持久化 pending 标记；摘要迟到不会单独取得完整资格，超时保留 partial。此结论不消除 R1。
- **停车优先与共同期限**：`main.mjs:67–70` 先写独立 stop；在控制终态/清理时停止录制。`sealArchives()` 为 M5a/M5b 传入同一个绝对截止时刻并行等待，最多额外 2 秒；视频不会覆盖停车和资源状态。M5a 仅扩展 seal 接收截止参数，原 pending、批次 hash 和游标绑定仍保留。
- **媒体读取**：固定同源 record ID/文件名路由、只允许 GET/HEAD、单一合法 Range；活动实验期间拒绝。逐级路径/符号链接约束及 O_NOFOLLOW 保留；核对身份、索引、块和整文件 hash，同一描述符输出前重新校验块，最多一个有界块且分段输出。CSP 只增加同源 media，scheme 增加 stream，没有任意文件路由。
- **回看与兼容**：视频模式清空 current/history，经 `brain.update(null)` 清空活动，隐藏神经批次游标；轮询不会向回放灌入实时批次。显示“视频未对齐”，切记录/模式/隐藏时暂停；M5a 采样与无画面的旧脑回放分支保留。没有恢复用户删除的大标题、底部播放条或历史说明。

上述为本次代码审阅范围内的结论，不是实际采集、播放或仿真动作成功证据。

## 证据范围与停止

已阅读当前 AGENTS、使命、技术栈、项目规则、交接、Spec 索引、M5b 契约/实施报告、M5a 最终独立验收，以及当前桌面采集、主进程、controller、renderer、媒体协议及相关测试。继承实施报告中 **11 个不同用例各自已有通过证据及一次 build 通过**，没有把它写成独立重跑 11/11；未机械重复这些检查。本验收只为 R1 执行上述两个小型 CPU 复现，没有新增业务测试或运行全仓检查。

`media_anchor` / `alignment` 仍为 null，是契约允许的视频单独回看分支；真实首帧锚点、带误差范围的联合视频播放尚未完成。真实编码支持、可播放 WebM、时长/seek、实际断线、帧率与共驻留开销均未验证。没有启动 Electron、GPU VM、旧 CPU 仿真、MPS/完整脑或硬件，没有停止 Noria 或改变 Docker。

**M5b 本轮工程 FAIL；M5 同步、P5 行为验收、M6 上海真机均不能据本报告判 PASS。** R1 已具体定位并给出限定复现，本验收在写完报告后停止，不继续扩展探索。
