# M5a 采样回放返工独立验收

日期：2026-10-23。结论：**PASS，限 R1/R2 修复及 M5a 桌面工程**。验收者 `roadmap_m5_capture_final_review` 未参与首次实施或返工。此前 [FAIL](m5-capture-acceptance.md) 的两项问题已按 [返工报告](m5-capture-repair.md) 修复；没有发现本次修复引入的直接阻断问题。

风险等级仍为 S3（封存终态、持久化完整性与 IPC），审查范围限定桌面相关子系统。仅新增本报告；未修改业务代码、Python 或冻结训练来源，未启动 Electron、仿真、GPU/MPS、完整脑或硬件，未 commit。

## R1：超时终态不会被迟到摘要反转

**PASS。** `apps/desktop/electron/recording.mjs:98–103` 先发布绑定身份和 generation 的 `seal_pending.json`，再发布声明新封存协议的 manifest。读档路径在读取成功摘要之前检查该标记；标记仍存在时只返回 `incomplete / complete=false / replay=null`（263–267 行）。因此，已在异步 `link` 中的成功摘要即使超时后落盘，也不能单独取得完整档案资格。

`finalize()` 的超时路径递增 generation，并保持 pending 标记。摘要发布返回后再次检查超时、generation 和期限，只有检查通过才在同一事件循环段同步删除 pending 并更新状态（181–209 行）；其间没有另一个可被定时器穿过的异步提交窗口。I/O 异常同样保留不完整标记。读取还核对 summary 与 manifest 的 generation（270 行）。

新回归用例直接拦截本进程的 `node:fs/promises.link`，让 summary 发布已经进入 I/O 后跨过既有 2 秒预算，再释放并确认磁盘摘要确为 `sealed`；随后实际 `readCaptureArchive()` 仍返回不完整，内存 generation 不回退（`recording.test.mjs:102–123`）。该用例覆盖了首次验收复现的窗口。

同步操作只限一次本地元数据删除；不涉及同步图片写入。采样封存仍位于控制清理确认之后，`stopExperiment()` 仍先请求独立停止，再冻结采样；采样状态不能覆盖 `safeToClose`、停车或资源清理状态。2 秒仍是异步等待预算，不能据此声称操作系统文件调用具有硬实时保证。

## R2：实际联合批次与索引、游标绑定

**PASS。** 封存索引从同一次受限文件读取取得批次内容、字节数和 SHA256，核对文件名与内容序号，并同时记录原始 Number 时间与转换后的主机时间（`recording.mjs:220–226`）。读档逐条拒绝重复批次并核对实际文件（283–302 行）。同身份内的时间或内容变化不能继续作为有效联合回放。

实际 IPC 链路贯通为 `preload.readBatch(id, sequence, cursorIndex)` → `main.mjs` 的 `records:batch` → `ExperimentController.readBatch()` → `readCapturedBatch()`。主进程校验整数游标，打开并验证档案，再根据该游标选择不晚于游标的最后批次；请求其他或未来批次会被拒绝。随后对**最终返回的文件字节**重新检查 hash、字节数、run/session/神经 execution/checkpoint/input map、序号及原始/索引时间（`controller.mjs:123–133`、`recording.mjs:304–316`），覆盖打开档案与取批次之间的文件变化。

renderer 联合路径实际传入游标，先运行 `validReplayBatch()` 再更新图像与脑显示；校验批次身份、序号、原始时间、索引时间以及“不晚于游标”（`src/main.ts:222–236`、`src/data.ts:41–44`）。校验失败不会把请求的游标和非法批次写入当前显示状态。

新增回归经过真实控制器读档/读批次路径，覆盖同身份时间由 9 秒变为 20 秒、保持时间而改变内容、打开后的变化、未来批次及非法游标；renderer 校验覆盖错误序号、时间及早于批次的游标（`recording.test.mjs:125–152`）。这不是只测静态选择数组。

## 保留的边界

- 旧两参数脑回放仍保留。无新封存协议的旧画面档案明确不完整、不做联合对齐；其独立脑记录仍可读取。没有修改 Python 遥测 schema 或旧 16 MiB 档案额度。
- record ID、整数序号/游标、固定目录、符号链接拒绝、有界读取和图片 hash 校验保留；没有新增任意路径、URL、命令或采集权限。
- 两侧布局与小型导航保留，没有恢复顶部大标题、底部播放条或历史说明小字。没有实现连续视频。

## 实际检查与停止

已阅读当前 AGENTS、使命、技术栈、项目规则、交接、Spec 索引、M5a 契约、首次 FAIL、返工报告，以及当前封存/读取/IPC/renderer 代码和相关测试。仅只读核对 Python 遥测字段，没有执行 Python。

采用实施者及协调者提供的已通过证据：`node --test electron/recording.test.mjs` **10/10**，以及一次 `npm run build` **通过**；审查了新增回归的注入位置、实际业务调用与断言。按项目停止规则，未机械重跑这些检查，未增加测试或运行全仓检查。报告中没有把继承的检查结果表述为本验收者重跑结果。

真实 `capturePage`、noVNC 采样、同次真实画面回放、采样性能和 Node/Python 时钟比较仍未运行；fixture JPEG 不是实际仿真证据。M5a 工程通过不代表 M5b 连续视频、P5 行为效果、模型能力或 M6 上海真机通过。

R1/R2 已满足限定验收要求，本验收工作在报告完成后停止；后续阶段由协调者依据各自契约推进。
