# Acceptance — MaleCNS Lab 命名迁移

## Result

**PASS**

2026-10-20，由未参与本轮实现的独立验收 Agent 审阅 Spec 009、Plan、实施报告、实际新文件、已跟踪差异与当前安装元数据。验收覆盖仓库命名和入口整理；不构成模型学习效果、GPU 续训或实体控制验收。

## Spec Coverage

| 标准 | 结论 | 证据 |
|---|---|---|
| AC1：新包、CLI、editable 与模块入口 | PASS | `pyproject.toml:6,35,41` 分别登记新发行名、console script、资源包；`src/malecns_lab/__main__.py` 使用当前包的 CLI。独立读取两个 venv 的 `malecns_lab-0.1.0.dist-info/{direct_url.json,entry_points.txt}`，均为本 checkout 的 editable；旧 dist-info、旧 bin 和 `src/cnsbridge` 均不存在。实际 CLI/import 运行证据见 `implementation.md:60`。 |
| AC2：执行路径、worker、示例、测试与当前命令 | PASS | `train_gpu.py:63` 等子进程模块引用使用 `malecns_lab`；三个完整图 benchmark 的当前源码闭包/路径同步更名。搜索运行源码、测试、示例、配置和 manifest 后，旧名仅剩持久化契约、已声明的历史 `git show` 路径、数值来源归属。测试中的新入口覆盖见 `tests/test_business_layout.py:13`。 |
| AC3：工作区定位与离线检查 | PASS | `src/malecns_lab/paths.py:11` 按新 src layout 定位，`config.py:25` 从新包读默认资源；`test_business_layout.py:33` 在外部 cwd 拦截 Torch、OpenCV 和设备 SDK 提前导入。实施报告中的定向离线测试、两个环境不借助 PYTHONPATH 的安装检查和普通 wheel 检查覆盖此风险。 |
| AC4：格式、历史资产与严格来源准入 | PASS | 源码逐文件审阅未发现名称/路径之外的逻辑变化；`gpu_contract.py:120` 的完整来源/运行时准入保留，`gradient_probe_evidence.py:61` 仍读取旧 commit 的旧包路径。实施者已逐项比对 197 个文件的 versioned format 字符串；`docs/MIGRATION.md` 明确不能用新源码直接续跑旧 hash-bound 实验。历史 Spec/Decision、已有审计、LICENSE 在本轮差异中未改写；当前 handoff 只加新任务段落，旧正文保留。 |
| AC5：导航与证据边界 | PASS | README、`docs/README.md`、`docs/MIGRATION.md`、`docs/AGIBOT.md`、AGENTS、constitution、项目规则和 Decision 016 使用一致的新名称/路径。WebGL、G2/X2 接入明确尚未实现；训练指标与固定策略成绩、旧 Tello 实飞与新策略效果分开。G2 手册标明第三方托管及版本未确认，X2 官方文档与部署条件单列。 |
| AC6：版权、Git 与用户文件 | PASS | `THIRD_PARTY.md` 只变本项目展示名，来源 URL 与署名保留；HEAD 仍为 `732ddc0`，remote 为空。独立读取预存未跟踪文件并计算 SHA256，与实施前登记的 `bbcd8795d53e03ca4e77138192dea2b49d02a45a65b8046e023fab268f98a075` 一致。没有在验收中提交、推送或修改它。 |

以上源码路径未写全时，均相对于仓库根目录；`train_gpu.py`、`gpu_contract.py` 与 `gradient_probe_evidence.py` 位于 `src/malecns_lab/learning/`。

## Validation

- Risk Level：**S3**。跨包导入、分发/资源、worker 模块和源码身份的迁移涉及共享基础设施；未改变神经数值逻辑。
- 独立执行的审阅/检查：
  - 阅读固定治理入口、Spec/Plan/Decision/实施报告及当前文档；检查未跟踪的新包与新文档，而非只看 `git diff`。
  - 以 `HEAD:src/cnsbridge/...` 对照迁移后的 138 个包内文件，分类命名/路径替换及保留的历史标识，未发现其他逻辑变动。
  - 审阅 manifest、资源定位、worker、benchmark 源码闭包、测试增量和来源验证代码；分类当前代码中的全部旧名称残留。
  - 只读检查两个 venv 的安装元数据和新旧入口文件、Git HEAD/remotes 及用户原文件摘要。
- 采用并审阅实施者已经完成的验证记录（`implementation.md:35–69`），未机械重跑：
  - 一次完整测试收集：761 项。
  - 32 个不同定向测试通过。首次同进程执行为 31 passed / 1 error；该 error 是原有 fixture 重复配置 PyTorch interop 线程，唯一受影响节点在独立进程 1 passed，其他已通过节点未重跑。不能将它写成一次完整无错误的测试执行。
  - 两个 venv 的外部 cwd、移除 PYTHONPATH 后的新 import、版本、CLI、资源及工作区定位检查通过。
  - 普通 wheel 构建和清单检查通过，包含新包及 defaults，不包含旧包或本地研究资产。
  - versioned format 字符串逐项比对、diff 空白及新增文档本地链接检查通过。
- 未执行新测试、全图加载、训练、性能基准、硬件操作或机器人 SDK 安装；未重新计算全部被忽略科研资产的哈希。本次资产保护判断依据受限变更范围、来源校验代码及实施记录，不把源码迁移验收解释成全部历史数值重新复现。

## Issues

无阻塞本批验收的问题。

既有两个测试 fixture 在同进程重复设置 interop 线程的问题已如实记录，受影响节点已独立通过，与本次命名迁移无关，不在本批扩大修复范围。

## Regression Risks

- 旧 `cnsbridge` 导入及 CLI 有意停止工作，外部脚本应使用新入口；没有兼容转发层。
- 名称迁移改变源码摘要；旧训练/诊断 run 会继续受到严格来源约束。使用原提交恢复原实验，未来参数迁移另立协议。新命名后的 `--help` 通过不等于旧 GPU run 可直接继续运行。
- 普通 wheel 不携带工作区数据。现有设备/实验工具仍要求源码 checkout 和 editable 安装。
- 本批没有新机器人的驱动、WebGL 应用或新训练策略的实体成绩。

## Required Rework

无。协调者可按本报告更新 Spec/Plan 与当前任务状态并交付；无需追加已通过检查。
