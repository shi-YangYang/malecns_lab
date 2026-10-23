# M5b 停止边界修复最终独立验收

日期：2026-10-23。验收者：`roadmap_m5_video_final_review`，未参与原实施、首次审查或返工。结论：**PASS，原 M5b 验收唯一 R1 已关闭；M5b 工程验收通过，实际录制仍待验。** 原 [FAIL 报告](m5-video-acceptance.md)保留，修复证据见 [限定返工报告](m5-video-stop-repair.md)。

本轮仅复核正常停止的末块判定与直接关联的接收、封存边界。局部修复为 S1，M5b 整体隔离、IPC 与归档风险仍为 S3。没有修改业务代码、训练来源或其他报告。

## R1 修复成立

- **末块证据来自 stop 事件。** `simulation-preload.cjs:74–89` 不再根据 recorder 的 inactive 状态标记 final。调用 stop 后，最新数据事件保留为 held 候选；下一数据事件到达时，前一非空候选释放为普通块。真正的 recorder stop 事件才将最后候选标记为唯一 final，并在其后入队 stopped。因此，停止前已排队但停止后才派发的 timeslice 不会抢占末块身份。
- **字节与 ACK 顺序未放宽。** `simulation-preload.cjs:31–57` 的单一 pump 按队列顺序转换和发送 Blob，每个事件等待匹配序号的 ACK 后才推进。held 候选不会先发；stopped 不能越过最终块 ACK，`final_sequence` 对应实际发送的最后块。空中间事件不写入字节，空最终事件仍可作为末块证据。`video.mjs:81–115` 仍拒绝并发写、序号不连续、第二个 final 和缺少 final 的 stopped。
- **背压仍包含候选末块。** 在途块、待发块及 held 候选统一计入 `a.blobs` 和 `pendingBytes`。单块 8 MiB、合计最多两块/16 MiB、事件队列上限以及 1500 ms ACK 期限未增大；超限进入明确失败，不丢弃非空块换取成功。正常完成的保证以未触及这些既有预算为前提。
- **没有 stop 就不能封存。** 缺少 recorder stop 时，最后候选保持 held，不发 final 或 stopped；有 stop 但没有候选时走缺少最终数据的故障路径。主进程封存仍同时要求 started、finalChunk、stopped、非空文件、无故障及期限内完成，pending 标记未撤销前读档始终 incomplete。

## 关联保护保持成立

`main.mjs:67–70` 先执行独立控制 stop，再停止图片和视频采集；返工没有改这一顺序。`sealArchives()` 在 `main.mjs:89–97` 只建立一个绝对截止时刻，图片与视频并行使用同一最多额外 2 秒预算，逐块 ACK 不重置该预算。`video.mjs:126–147` 期限到达即关闭当前档案资格，迟到摘要也不能越过持久化 pending 标记成为完整档案。

页面重载仍递增 document generation 并撤销旧视频，主进程每条消息仍验证专用 WebContents、main frame、固定 URL、当前 generation 和随机 token（`main.mjs:131,173–180`，`video.mjs:42–43`）。超时关闭后 authorization 为 null，旧 pump 即使结束异步转换或收到迟到 ACK，也不能把数据写入新 run；已经接收的异步写仍受 closed 检查约束。本次修复没有引入跨 run 队列、重试或自动恢复。

## 检查证据与范围

已阅读当前 AGENTS、使命、技术栈、项目规则、交接、Spec 索引、M5b 契约、原 FAIL、返工报告，以及生产 preload、相关测试、接收/封存和主进程停止路径。

继承返工实施者一次定向 CPU 命令的 **4/4 PASS** 记录，并核对当前测试内容；**本验收没有重新执行这些测试，也没有运行 build 或全仓检查**。新增回归确实执行生产 preload，接入实际 `SimulationVideo.receive()`、文件写入与读档路径，覆盖：

- 异步 stop、先前排队的 timeslice、真正末块以及 stop 事件；两个尾部 ACK 分别延迟；只有真正末块为 final，最终文件等于原始三段非空字节顺序连接。
- 最终 ACK 前不存在 stopped，读档仍 incomplete；全部确认后才 sealed，final_sequence 与末块一致。
- 缺失 stop 事件时到期 incomplete，候选块和停止确认均未发送。
- 原固定采集参数、正常结束顺序与两块背压用例在本次队列修改后通过。

未发现需要另作复现的具体疑点，因此不追加或机械重复检查。原审查已通过且本次未改变的来源、媒体 Range、路径约束及界面分支沿用原结论，不冒称本轮重新完成全面验收。

## 保留限制与停止

`media_anchor` 与 `alignment` 仍为 null，当前为**视频单独回看**；不把视频时间与神经批次假定对齐。真实首帧锚点、编码/解码、可播放 WebM、时长/seek、实际帧率、断线行为和共驻留开销尚未验证。注入事件与固定字节通过，只证明工程协议和归档顺序。

没有启动 Electron、GPU VM、CPU 仿真、完整脑、MPS 或硬件；没有操作 Noria、Docker 配置或进行 commit/push。M5 同步实测、P5 行为验收与 M6 上海真机均不能据此判 PASS。本报告完成即停止。
