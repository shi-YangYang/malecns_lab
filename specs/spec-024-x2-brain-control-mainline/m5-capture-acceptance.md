# M5a 同次画面采样与回放独立验收

日期：2026-10-23。结论：**FAIL，限定返工两项工程问题**。验收者 `roadmap_m5_capture_review` 未参与实施。风险 S3，范围仅桌面采样、持久化、读取/IPC、主机时间关联与生命周期；未修改业务代码、Python、模型或训练来源，未启动 Electron、脑、仿真或硬件，未 commit。

本结论不评价候选模型的行为能力。即使下列问题修复并通过工程复验，真实采样、同次回放、M5b 连续视频、P5 任务效果和 M6 真机仍分别待验。

## 必须修复的问题

### R1 — 超时后的迟到 summary 发布会把不完整档案重新认成成功封存

**P2；位置：** `apps/desktop/electron/recording.mjs:42–48, 177–196`。

`exclusiveFile()` 在执行 `await link(temporary, target)` **之前**检查 `stillCurrent()`，但不能阻止这次异步发布在随后发生的封存超时之后完成。`finalize()` 的 2 秒定时器将内存状态改为 `incomplete` 并递增 generation，但已在途的 summary 发布仍可落地；落地内容是之前生成的 `state=sealed, complete=true`。读取端只信任这个 summary，因此同一档案在本次进程内报告超时，稍后或重启后却被认成完整封存。这违背“超时保留不完整、迟到结果隔离”的契约；控制停车状态本身没有被改写。

**实际最小 CPU 复现：** 在临时同身份档案中采样一张注入 JPEG，只将 `node:fs/promises` 的 `link` 对 `summary.json` 的发布延迟 2300 ms，其他操作正常。直接调用现有 `SimulationCapture.seal()`，随后调用现有 `readCaptureArchive()`：

```json
{"status":{"state":"incomplete","frames":1,"reason":"画面封存超过 2 秒，保留未完成档案","complete":false},"generation":2}
{"state":"sealed","complete":true,"summary":"sealed"}
```

**限定修复要求：** 使封存超时成为读取端也必须尊重的终态，不能由在途发布覆盖；封存决定与迟到文件的有效性须受同一 generation/终态规则约束。只在调用异步 `link` 前检查 token 不足。补一个“最终发布已经进入 I/O 后再超时”的定向用例；已有“capture Promise 迟到”用例没有覆盖该窗口。不得通过延长 2 秒预算规避。

### R2 — 联合时间索引没有绑定实际读取的神经批次，批次变化后仍可显示未来数据

**P2；位置：** `apps/desktop/electron/recording.mjs:208–212, 247–265`；`apps/desktop/electron/controller.mjs:123–128`；`apps/desktop/src/main.ts:224–235`。

封存只把批次序号和当时的 `host_monotonic_ns` 写入回放索引，没有绑定该批次文件的内容 hash。打开档案验证图片/索引及 manifest 等来源，却不重新验证索引对应的批次文件；`readBatch()` 返回当前文件，renderer 的 `validBatch()` 也没有检查返回批次的序号/时间是否与本次游标选择的条目一致。因此文件损坏、错放或封存后被修改时，仍可在原来的时间点展示较晚批次，且画面档案继续标为完整。现有 run/session/checkpoint 校验不能发现同身份内的这个错误。

**实际最小 CPU 复现：** 在上述临时档案封存后，只将 `telemetry/batches/000001.json` 的 `host_monotonic_ns` 从 `9000000000` 改为 `20000000000`，其他身份与序号不变。现有读取器返回：

```json
{"archiveState":"sealed","indexTime":"9000000000","batchTime":20000000000}
```

在 10 秒的画面游标处，当前 `replayAt()` 仍据 9 秒索引选择批次 1，而后续 `readBatch()` 会返回 20 秒文件；静态核对 renderer 没有拒绝该对应关系。该例不需要修改图片、索引、summary 或任何其他 hash。

**限定修复要求：** 在独立画面/联合回放档案中绑定所引用批次的内容身份，并在实际返回用于联合显示的批次时验证 run/session、序号、来源时间与索引的一致性，拒绝晚于游标或已改变的文件。保留旧遥测 schema、旧记录兼容与 16 MiB 额度，不修改冻结的 Python 来源。验证应覆盖“封存后同身份批次内容/时间改变”，不能只测跨 run 身份或游标选择函数的静态数组。

## 逐项审查

| 项目 | 结果 | 依据及范围 |
|---|---|---|
| 采集对象与权限 | PASS（静态工程） | `main.mjs` 只持有隔离 simulation WebContentsView，使用 `capturePage` 的 `stayHidden=true, stayAwake=false`；无整屏权限、任意 URL、网页注入或新依赖。已有网络、导航、下载与 sandbox 限制保留。 |
| 单在途、额度、缺帧 | PASS（工程） | `tick()` 单在途、不追赶；独立时长定时器；图像/索引/磁盘/尺寸限额明确；隐藏、最小化、加载错误触发 gap，恢复后成功采样才结束 gap。实际编码耗时和 2 Hz 达成率未测。 |
| 紧急停止、关闭与资源状态 | PASS（静态工程，封存终态除外） | `stopExperiment()` 先调用控制器停止，再冻结采样；`safeToClose` 仍要求子进程退出和资源清理确认；采样不改写控制清理状态。新 prepare 在初始化/封存尝试未结束前拒绝；单对象固定原 root，旧 capture 不换入新 run。 |
| 2 秒封存与迟到隔离 | **FAIL — R1** | 已复现迟到 summary 将超时档案重新认作完整。已有迟到 capture 用例只覆盖发布前的等待。 |
| 运行与策略身份 | PASS（静态工程） | 登记、brain_ready、遥测 manifest、神经 session/execution、checkpoint/policy/input_map 交叉核对；官方资产和采集源码 hash 绑定；采样期间重读来源身份。 |
| 神经与技能身份区分 | PASS（静态工程） | 神经 `execution_id=session_id`；帧上下文的 `skill_execution_id` 来自采样前已知的上个脑批次，并注明 `last_available_neural_batch_input_feedback`，未后填未来技能 ID。主机 execution 事件使用独立技能字段。 |
| IPC、路径、图片完整性 | PASS（工程） | frame IPC 仅 record ID 与整数序号；逐级符号链接检查、固定路径、有界读取、图片及索引 SHA256、JPEG 边界；非法画面档案拒绝联合对齐并保留独立脑记录路径。未放宽旧遥测额度。 |
| 主机时间与远端时间区分 | PASS（静态工程） | 新纳秒为十进制字符串；旧 Number 保留且标注近似转换。仅纳入任务/行为日志主机时刻，不混入远端 bridge 原始时钟；明确像素无源仿真时间。Node/Python 实际时钟比较仍待真实运行。 |
| 回放只取过去数据 | **FAIL — R2** | 选择函数对未改变的索引向后匹配；实际读取的神经批次未与索引绑定，已复现同身份时间变化仍被接受。 |
| 动作段无新脑批次、老化与 gap | PASS（工程） | 主游标包括图片和主机事件，未限定为脑批次；画面超过 1 秒或 gap 不显示，脑批次超过 2 秒标“上一批”，不插值脉冲。 |
| 旧记录与 UI 范围 | PASS（静态工程） | 无 capture 目录返回 absent，旧逐批读取保留；回放隐藏实时仿真，保持左右布局和现有小型导航，没有恢复删除的底部播放条/顶部大标题/历史说明。 |

## 实际检查与证据边界

- 阅读了当前 AGENTS、使命/技术栈/项目规则/交接、Spec 索引、M5a 契约与实施报告，以及相关桌面改动。只读核对 Python 生产者的身份、事件与准备顺序，没有修改或执行 Python。
- 采用实施报告及协调者提供的既有结果：Node 20/20，后补改动相关采样用例 8/8，TypeScript 与 Vite 构建通过。审查了用例内容，**未机械重跑**这些检查。它们没有覆盖 R1 的最终文件发布在途超时和 R2 的封存后批次变化。
- 本验收额外执行两个小型 CPU 命令：先确认当前 Node 的内建 `link` 注入只影响本进程，再用一个临时档案脚本同时复现 R1/R2，实际约 2.42 秒。临时 fixture 随脚本清理；项目 run、模型、训练、仿真与历史资产未动。
- 未做实际 Electron/noVNC 采样、真实画面回放、性能或同机跨语言时钟实验。上述 PASS 均不等于真实采样通过，注入 JPEG 不是仿真截图证据。

## 交回与停止

按项目流程将 R1/R2 交新的实施者限定返工，再由新的独立验收者审查；本验收者不代改业务代码。修复之外不增加功能、不重做已通过检查。M5a 工程尚未通过，M5/P5/M6 均未完成。
