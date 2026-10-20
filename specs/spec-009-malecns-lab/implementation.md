# Implementation Report

## Summary

2026-10-20 完成 `CNSBridge / cnsbridge` → `MaleCNS Lab / malecns_lab` 的工程命名迁移。当前 package、CLI、资源加载、动态导入、子进程模块和源码定位使用新名；不提供旧转发包。README 与文档索引区分现有研究工具、历史实验和拟议的 WebGL／智元接入。

本报告是实施记录，不替代独立验收。没有 commit、push、GitHub 仓库创建或硬件操作。

## Files Changed

- `src/cnsbridge/` → `src/malecns_lab/`，138 个被版本控制的包内文件保持同一职责；源码逐文件比对确认只包含允许的名称／路径改动。
- `pyproject.toml`：分发名、console script 和 package data 统一为 `malecns_lab`。
- `tests/`、`examples/`：导入、mock／动态导入、模块命令、源码闭包和路径同步迁移；`test_business_layout.py` 增加现有训练／诊断入口的安全 help 检查，拦截 Torch、OpenCV 和设备 SDK 提前导入。
- `configs/` 与包内默认配置：仅现行命令注释改名，参数不变。
- `README.md`：当前能力、数据计数、学习结果边界、源码结构和安装入口。
- 新 `docs/README.md`、`docs/MIGRATION.md`、`docs/AGIBOT.md`：分类导航、持久化与续训边界、G2／X2 文档入口。
- 当前使用文档更新导入与命令；涉及旧训练目录的操作说明增加源码迁移限制。
- `THIRD_PARTY.md`：只调整本项目展示名，保留来源 URL 和署名。

协调者负责的治理文件不由本实施者修改。

## Important Decisions

- 全部既有 `cnsbridge.*.vN` 与 `flydrones.*.vN` 持久化格式保留，不给旧 JSON／NPZ／checkpoint 改名或改摘要。
- `learning/gradient_probe_evidence.py` 当前工作区路径改为 `src/malecns_lab`，但 `git show {R1_COMMIT}:src/cnsbridge/{name}` 保持旧提交的路径。
- `brain/trainable.py` 科学来源描述里的 “explicit CNSBridge event order” 保留为原数值协议归属，非运行入口。
- `docs/MALECNS_AUDIT.md` 和 LICENSE 与 HEAD 完全相同；旧 Spec、Decision 和实验资产未批量改写。
- 现有 hash 准入无放宽。原始训练 run 与本次源码身份不同；未来参数迁移需要独立协议。
- 不移动 `/Users/yang/project/FlyDrones` checkout 根目录，不创建空前端／机器人目录。
- 安装前核对 `.venv` 的旧 `cnsbridge` 确为此 checkout 的 editable；`.venv-train` 未安装旧包。两个环境均以 `--no-deps --no-build-isolation -e .` 安装新包，仅 `.venv` 卸载旧包；没有更新依赖或系统 Python。

## Validation

- Risk Level: **S3**，理由为跨包导入、打包、进程入口和来源身份迁移；神经数值逻辑未改。
- 全部测试仅做一次收集：
  - `.venv-train/bin/python -m pytest --collect-only -q`
  - **761 tests collected，1.44 s**；未运行完整测试套件或完整图计算。
- 定向执行以下节点：

```bash
.venv-train/bin/python -m pytest -q \
  tests/test_business_layout.py \
  tests/test_cli.py::test_cli_help_lists_offline_tools \
  tests/test_cli.py::test_cli_rejects_removed_legacy_commands \
  tests/test_adjoint_benchmark.py::test_parent_import_cli_and_source_closure \
  tests/test_mps_benchmark.py::test_lightweight_entry_and_complete_new_source_closure \
  tests/test_gpu_recovery.py::test_light_parent_and_job_budget_guards \
  tests/test_gradient_probe.py::test_parent_imports_lightweight_and_direct_worker_rejected \
  'tests/test_training_checkpoint.py::test_bad_archives_rejected_atomically[source]' \
  'tests/test_training_checkpoint.py::test_bad_archives_rejected_atomically[runtime]' \
  tests/test_policy_checkpoint.py::test_production_loader_rejects_numerical_role_without_graph_load \
  tests/test_policy_checkpoint.py::test_new_and_old_formats_reject_each_other
```

  - 首次 **31 passed，1 error，4.20 s**。唯一 error 发生在 `test_gradient_probe.py` 的既有 autouse fixture：同进程中重复调用 `torch.set_num_interop_threads(1)`；未进入该测试断言，不是包名或导入失败。未修改这个与迁移无关的 fixture。
  - 仅将该失败节点另起进程执行：
    `.venv-train/bin/python -m pytest -q tests/test_gradient_probe.py::test_parent_imports_lightweight_and_direct_worker_rejected`
  - **1 passed，1.14 s**。其余 31 项未重复；合计 32 个不同定向测试通过。
  - 存档检查只使用已有小型数值测试 fixture；未加载完整参考图。测试产生一条 PyTorch sparse CSR beta 提示。
- 两个环境的安装验证均从临时 cwd 启动、移除 `PYTHONPATH`：新 import、版本、console `--help/--version`、`defaults.yaml`、`default_config()` 和 `source_project_root()` 通过；旧 `cnsbridge` import、分发元数据、bin 入口均不存在。
- 临时目录构建普通 wheel：
  - `.venv/bin/python -m pip wheel --no-deps --no-build-isolation --wheel-dir <temporary-directory> .`
  - **通过**；`malecns_lab-0.1.0-py3-none-any.whl` 包含 144 个成员、`malecns_lab/defaults.yaml`、`__main__.py` 和新 console entrypoint；无旧包、data、logs 或 tests 内容。
- 静态迁移比对：
  - 197 个原跟踪源码／测试／示例／配置文件中的 versioned format 字符串逐项不变。
  - 138 个包内文件与旧源码比对，仅名称／路径迁移；保留既有来源校验。
  - 旧运行名称残留分类后，仅有持久化协议、上述历史 git 路径与数值归属文字；旧科学审计另行原样保留。
  - 预先存在的未跟踪 `specs/spec-008-trainable-snn/scipy-backward-boundary-audit.md` 字节不变，SHA256 为 `bbcd8795d53e03ca4e77138192dea2b49d02a45a65b8046e023fab268f98a075`。
- `git diff --check` 与新 README／导航／迁移／智元文档的本地链接目标检查通过。

## Known Limitations

- 没有运行全图训练、完整性能基准、数据重建、设备连接或实飞；本批不能提供新的控制效果证据。
- 没有重跑全部数值测试；包迁移风险通过全量 collection、定向入口／源码闭包／来源与存档检查覆盖。
- 当前无 WebGL 前端、智元适配器或新模型的实体部署。
- G2 PDF 是 RobotShop 托管的厂商 v2.1.0 手册，最新版与现场版本仍需确认；X2 仿真环境与包尚未部署。
- 历史 hash-bound 实验不能因更名直接续跑；没有实现自动参数或训练状态迁移。
- 已观察到既有两组 PyTorch 测试 fixture 在同一进程配置 interop 线程的冲突；本批将受影响节点单独执行通过，没有扩大范围修复测试调度。

## Remaining Questions

本批实施无未解决的命名迁移问题，待新的独立验收者审查。后续机器人型号／固件、交互任务及 WebGL 设计另立 Spec。
