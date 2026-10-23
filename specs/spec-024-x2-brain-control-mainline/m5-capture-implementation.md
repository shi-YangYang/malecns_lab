# M5a 同次采样与回放实施报告

日期：2026-10-23。状态：实施完成，待新的独立验收；**未进行真实采样，M5/P5 未完成**。依据 `m5-capture-contract.md`，风险 S3（主进程采集、受限档案读取、IPC 与关闭生命周期），检查仅限桌面子系统。本次未修改 Python、训练来源、候选、官方组件、协议旧 schema 或训练预算；未启动 Electron、仿真、完整脑或硬件，未 commit。

## 实际修改

- 新增 `apps/desktop/electron/recording.mjs`。采样只由主进程传入隔离 noVNC WebContents 的 `capturePage(undefined, { stayHidden: true, stayAwake: false })`；无整屏采集、权限申请、网页注入或新依赖。运行身份、候选 checkpoint、policy/input map、神经 execution/session、官方资产与采集源码 hash 先核对并冻结。采集期间重新检查来源身份；单技能 execution 保留在独立字段中，已知技能上下文注明来自上个脑输入反馈，不推断当前身体正在执行什么。
- 固定目标 2 Hz、单在途且不追赶漏帧；JPEG 70、按比例最多 1280×800、不放大，240 秒/480 张、单张 512 KiB、图片总 256 MiB、索引总 4 MiB，开始至少留出 512 MiB 磁盘。独立截止定时器不依赖 capture Promise 返回。隐藏/最小化/加载异常即时记 gap，恢复后直到成功采样才关闭 gap。
- 每次固定 run 下独立 `simulation_capture` v1 manifest、JPEG、帧索引、回放索引及 summary。新帧经独占临时文件与原子 no-overwrite 发布，成功才入索引；旧 run 不覆盖。时间使用主机单调纳秒十进制字符串；既有 Python Number 字段原样保留，并标注换算精度。`task_requests`、`behavior_events` 只取可核对的主机时间；Linux bridge 原始单调时间不进入共同时间轴。
- 空图、写入/身份错误或额度达到上限先调用既有独立停车入口，再停止采样，不算普通任务成功。关窗/紧急停止/renderer 退出同样先请求控制停止；资源清理确认后才额外等待封存，最多 2 秒。超时 generation 失效，保留不完整档案，迟到内容不能入新 run；档案状态不修改控制器停车或资源状态。新实验须等待旧采样封存尝试结束。
- 主进程新增唯一具体读取 IPC `records:frame(recordId, integerSequence)`；`records:open` 返回有界元数据和时间索引。路径逐级拒绝符号链接、文件大小和 hash 校验，run/session/checkpoint/policy/input-map 交叉核对；只逐张返回 JPEG base64，不暴露路径/URL/capture/shell。缺 summary 明确 incomplete；非法画面档案拒绝联合对齐，仍保留单独的旧脑回放能力。原遥测 16 MiB 限额未变。
- renderer 沿现有面板前后按钮按主机时间浏览；没有新脑批次的技能执行段仍可查看后续图片。只显示游标之前最后帧/批次，图片超过 1 秒或 gap 显示“此处未采集”，脑批次超过 2 秒显示“上一批”；不插值脉冲、不加载实时仿真、不预载图片集。旧无画面档案继续逐批浏览。没有恢复底部播放条、大标题或历史说明小字。
- 更新 README 使现行候选入口与历史 Spec022 记录区分；准备提示按协调者最新流程改为“先 Pause、Reset，确认直立暂停后 Run，再点击开始”。不改变控制预算或任务逻辑。

## 实际检查与限定修复

1. `node --test electron/recording.test.mjs electron/controller.test.mjs electron/boundaries.test.mjs`：**20/20 通过**，约 2.45 秒。包括 8 项新增采样/回放用例，以及 12 项现有控制器和路径/IPC 边界检查。截图对象为测试注入，不是 Electron 真实截图。
2. 原计划在此命令前落盘的一组采样可见性/截止处理修改因工作目录错误未写入（FileNotFoundError，未改其他文件）；测试实际运行的是当时已有实现。随后将这组必要修改正确写入，并补充封存结束前拒绝下次 prepare。只重跑修改相关的 `node --test electron/recording.test.mjs`：**8/8 通过**，约 2.46 秒；没有重跑控制器/边界旧用例。
3. 一次 `npm run build`：素材准备通过；TypeScript 发现 `CaptureFrame` 类型导入遗漏，构建停在类型阶段。只补该 type import，再执行尚未完成的 `./node_modules/.bin/tsc --noEmit && ./node_modules/.bin/vite build`：**类型检查通过、Vite 构建通过（454 ms）**，未重复素材准备。
4. 最终仅检查本范围 diff，不运行全仓测试、Python 检查、lint、第二个脑或 GUI。

定向用例实际覆盖：单在途/无追赶、隐藏 gap 与恢复、时间边界、空图/单帧大小/磁盘预留、运行身份变化、独立停止回调先于冻结、缺 summary、2 秒封存超时后迟到结果隔离、索引/图片篡改、ID/帧号和符号链接越界、旧已封存脑记录兼容，以及 renderer 游标不会选未来批次/图片、动作阶段无新脑批次仍可看图和空档/老化呈现。没有用注入图片测试声称真实 noVNC 已采样成功。

## 限制与后续验收

- 真实 capturePage 成功率、编码耗时、实际帧间隔、CPU/磁盘开销、断线画面、同机 Node/Python 时钟比较和本次动作像素尚未实测。manifest 明示当前 macOS 同机路线假设及 actual-run 比较待验证；没有像素的仿真源时间，只承诺主机时间关联。
- 采样可能保存 noVNC 断线提示或停滞画面。页面加载成功/截图成功不证明 VNC 像素更新；界面不会把这些归为行为成功。
- 有界读取主机事件各最多 16 MiB、单事件 256 KiB、回放事件最多 10000；达到限额保留不完整档案，不无限扩容。采样总索引仍须不超过 4 MiB。发生封存写盘错误或超时可能保留 `.partial`/未入索引帧，不补造成功汇总。
- 程序结束但资源清理未确认仍沿原入口拒绝新实验/保留失败窗口；画面封存不是停车证明。
- 本子阶段需要新的独立验收者静态审查本实现和上述检查证据。实际采样与同档案回放需在 M3 结束、候选及相关动作准入满足后，合并到协调者登记的单次 M4 运行；不可为录制单独启动第二个脑或重复 GPU 运行。
- M5b 连续视频未交付，25 次 P5 任务与上海 M6 真机仍按原门槛处理。
