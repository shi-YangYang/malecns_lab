# Acceptance

## Result

PASS

独立验收 Agent 未参与实施，仅审查本次迁移及已有验证证据；未修改业务代码。基线为 `8546f6c`。

## Spec Coverage

| 标准 | 结果 | 证据 |
|---|---|---|
| AC1：职责实现归位 | PASS | `control/` 拥有动作契约、数值适配、旧读出、校准和控制限制；`learning/` 拥有读出、数据与离线流程；`experiments/tello/` 拥有设备实验，试飞的策略、子进程及会话分别提取到 policy、worker、session。七个旧工具均只导入并调用真实 `main`。 |
| AC2：依赖与导入兼容 | PASS | 新实现使用真实业务路径，`learning` 单向依赖 `control`，未发现包反向导入 tools 或旧兼容层。旧 motor、safety、calibrate 使用显式同对象导出；新旧身份回归已通过。低层无 SDK/OpenCV 顶层导入；check 的可选依赖移至参数解析之后，保留实际检查流程。 |
| AC3：原断言与离线回归 | PASS | 对照原测试差异，变化限于导入及真实实现定位；未移除或弱化控制、限幅、超时、降落顺序、数据及学习断言。原 spawn 用例改为真实 `worker.BrainProcess`，目标为可导入的 `worker._brain_main`。实施报告提供 144 passed、251 subtests passed、退出码 0 的确定结果。 |
| AC4：旧命令及路径语义 | PASS | 七个脚本均保留原入口与参数处理；新增测试在其他 cwd 下阻止 djitellopy/cv2 导入，执行七个帮助、两个无起飞标志入口及原 CLI。`source_project_root()` 根据本包实际 src 路径与 pyproject 定位，无法满足源码布局时明确失败，不把 site-packages 猜成项目根。trial_config 仍属于 trial，PROJECT patch 作用于真实配置函数。 |
| AC5：数据、身份与数值行为 | PASS | 对照迁移前后文件，动作及旧读出实现保持一致；学习实现仅更换导入，prepare 的哈希计算和持久化内容保持一致。实施报告提供四个来源文件字节不变、原 trial/hover 函数与类 AST 一致，以及原 RecordingBrain 独立基准；新增测试核对旧 source ID、特征顺序和 50 ms 窗口。实际审查确认清理仍先零 RC、最多两次 land、end，再停止 worker 和写日志，策略保持固定墙钟窗口。未发现实验资产或配置改动。 |
| AC6：文档与交付记录 | PASS | README、ARCHITECTURE、DATASET、TELLO、技术栈与规则对应实际布局；Decision 002 记录迁移目的和兼容策略，Spec/Plan/实施报告描述一致。保留旧控制与新学习读出的证据边界，没有宣称新增实飞、底盘验证或训练成果。 |

## Validation

- Risk Level: S3。涉及多个业务子系统、Python 导入兼容、源码资产定位及 spawn/试飞监督边界。
- Evidence Reviewed / Checks Performed:
  - 阅读 AGENTS、使命、技术栈、项目规则、交接、Spec、Plan、Decision 002、实施报告及验收流程。
  - 审查已跟踪差异及全部新增源码/测试；按原文件到新实现映射直接比较，确认除必要导入、根路径定位及 check 延迟导入之外未重写业务。
  - 审查原测试修改、新增迁移用例及实际 mock 目标。time 仍为共享标准库模块，现有假时钟覆盖提取后的 session；trial 的配置、worker 构造和 CSV 写入 patch 仍作用于实际调用位置。
  - 采信实施报告中的一次确定 pytest 结果、来源字节与 AST 证据；首次结果捕获不完整的说明符合允许重跑条件。验收未重复这些已经通过的检查。
  - 对本次新增源码、测试和治理文件执行 `git add -N` 以纳入差异，执行一次 `git diff --check`，退出码 0。该检查发生在本验收记录写入前。
- 未运行额外测试、build、lint，未连接硬件、下载数据或训练模型。

## Issues

未发现本次迁移引入的阻塞问题。

## Regression Risks

- 实验流程继续要求源码 checkout 与 editable 安装；普通 wheel 不包含工作区模型、配置和日志。这一边界已明确记录。
- 离线回归与等价迁移审查不能证明实体飞行效果；现有控制和学习方法的局限保持原状。
- 兼容入口保留迁移用途，未来移除需要另行定义范围。

## Required Rework

无。协调 Agent 可更新完成状态并按既有授权创建本地提交；无需追加无具体依据的验证。
