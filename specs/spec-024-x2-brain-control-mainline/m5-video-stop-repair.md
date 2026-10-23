# M5b 正常停止末块判定限定返工

日期：2026-10-23。实施者：`roadmap_m5_video_stop_repair`，不同于原实施者与验收者。对应 [M5b 独立验收](m5-video-acceptance.md)唯一 R1。本轮为 S1：隔离 preload 内部停止队列的局部修复；M5b 整体的 S3 隔离、归档和媒体边界不变。状态：**限定修复及定向回归完成，待新的独立验收者；未作实际录制结论。**

## 修改与保证

仅修改 `apps/desktop/electron/simulation-preload.cjs`、相应 `video.test.mjs` 与本报告，保留其他已有修改。没有修改 `video.mjs` 接收/封存协议、main 停车与共享期限、TypeScript、Python、训练来源、依赖或锁文件。

不再用 `recorder.state === 'inactive'` 判定数据末块。主进程 stop 命令发出后，preload 将最新的 dataavailable 放在已有有界队列中暂缓发送；下一数据事件到达时释放上一块为普通块。只有真正的 recorder stop 事件才将仍保留的最后数据块标记为唯一 `final: true`，随后排入 stopped。没有 stop 事件时不会发送候选末块或停止确认；没有 stop 调用后的数据事件时仍明确报缺少最终数据。

候选末块仍计入原有最多两块（在途与待发合计）、16 MiB 未确认数据、单块 8 MiB 与事件队列上限。原始非空数据按事件顺序发送，不重编码、不拼改字节。零字节中间事件与原录制阶段处理一致，不产生文件字节；零字节的最终事件仍保留并作为末块。候选块只有在队列头释放后才进入发送，stopped 仍等待所有前序块的逐一 ACK，不能越过最终块确认。

未放宽接收侧唯一末块、连续序号、缺停止确认拒绝、generation/token、迟到事件、pending 标记或封存限制。停车优先和图片/视频共享总 2 秒均未改变。

## 真实检查

只执行一次最相关 CPU 命令，工作目录 `apps/desktop`：

```sh
node --test --test-name-pattern='video isolated preload|video preload permits|video preload stop boundary' electron/video.test.mjs
```

结果 **4/4 PASS，0 FAIL**。回归执行生产 `simulation-preload.cjs`，其中新增停止边界用例贯通实际 `SimulationVideo` 接收、写入及读档：

- Recorder.stop 同步变 inactive，随后异步依次派发零字节中间事件、已排队普通 timeslice、真正末块与 stop；两个尾部块的 ACK 分别受控延迟。
- stop 前候选未发送；只有 stop 后最后数据标 final；最终 ACK 前没有 stopped，读档仍 incomplete。全部确认后 sealed，最终文件等于三段原始非空字节按顺序连接，只有一个 final，final_sequence 正确。
- 只收到候选数据而始终没有 stop 事件时，封存期限到达仍 incomplete，未发布完成档案。
- 原 preload 固定采集参数、正常结束顺序和两块背压检查在本次队列修改后分别通过。

原测试 Recorder fixture 的 stop 也改为异步派发事件，不再在 stop 方法内同步生成尾部事件。检查后人工核对本次 patch；未扩大或重复测试。TypeScript/入口未改，未重新 build；未重复来源、Range、M5a 等未修改检查。

## 限制与停止

测试使用注入事件与固定字节，只证明停止协议及字节顺序，不能证明实际 VP8 编码、可播放视频、帧率、同步锚点或仿真动作。没有启动 Electron、GPU VM、CPU 仿真、完整脑或硬件；没有操作 Noria、Docker 配置，也没有 commit/push。

本轮在报告完成后停止，交新的独立验收者复核 R1；原 M5b 验收 FAIL 在该复核通过前不改写为 PASS。M5 实际录制/同步、P5 行为与 M6 上海现场仍需分别验收。
