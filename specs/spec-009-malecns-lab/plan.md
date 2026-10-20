# Plan — MaleCNS Lab 仓库命名与入口整理

状态：Done。[实施报告](implementation.md)与新的[独立验收 PASS](acceptance.md)已处理，无返工项。风险 S3：跨包导入、打包入口、子进程执行和科研来源身份涉及全仓库。不是神经数值算法变更。

实际验证完成一次 761 项测试收集、32 个不同定向测试、两个 venv 的外部 cwd 安装/资源/CLI 检查和普通 wheel 清单检查。一次既有 PyTorch fixture 冲突通过仅将受影响节点单独执行核对，未重跑其余已通过测试。未运行训练或硬件；本批未 commit/push。达到停止条件，不追加验证。

## 分工与顺序

1. 协调者核对当前结构、用户修改、源码身份边界，维护本 Spec、Decision 016、AGENTS/constitution/项目规则和当前 handoff。
2. 一名实施者串行迁移整个 Python 包与所有当前执行入口，更新测试、示例、pyproject 和当前文档；避免多个实施者交叉替换。
3. 实施者在两个已有 venv 内核对旧项目安装归属，再以 `--no-deps --no-build-isolation` 进行本地 editable 安装迁移；不动其他依赖。
4. 实施者执行适量验证并提交 implementation.md；新的独立验收者检查 Spec、实际改动及验证结果，必要时只检查未覆盖的具体风险。
5. 协调者收取报告、更新状态并交付。本批无 GitHub 创建或 push；不把之前训练阶段的提交授权外推到本批。

## 预计文件

- `src/cnsbridge/**` → `src/malecns_lab/**`。
- `pyproject.toml`、`tests/**`、`examples/**` 中的包名、命令及路径引用。
- `README.md`、`THIRD_PARTY.md` 当前项目名、`docs/*.md` 当前入口。
- 新 `docs/README.md`、`docs/MIGRATION.md`、`docs/AGIBOT.md`。
- 协调者负责 `AGENTS.md`、`constitution/*`、`.ai/rules/project.md`、`.ai/decisions/016-malecns-lab-repository.md`、`.ai/prompts/task-handoff.md`、`specs/README.md`、本 Spec 目录。

## 替换边界

先区分 Python 名称/路径/CLI 与 schema 字符串，再迁移。不得全局替换所有 `cnsbridge`。保留 `schema` / `format` / contract 常量、旧运行摘要、上游来源和历史记录。只因更名导致的源码 hash 改变应自然生成新身份，原有准入检查维持。

## 验证计划

- 一次静态残留分类与 diff 检查：旧命名只允许存在于持久化契约、明确的迁移说明、历史证据和上游署名。
- 使用 `.venv-train` 对全部既有测试做一次 collection，检查跨包导入；实际执行入口/路径/资源检查（business_layout、cli）、两个 benchmark 源码依赖闭包检查，以及相关存档/来源校验的最小定向集合。只用小图/临时目录，不运行真实训练/性能基准、完整图或硬件。具体 node ID 与结果记入实施报告；失败时才沿具体原因扩大。
- 两个 venv 的新 import、CLI `--help/--version`、旧项目入口移除及工作区路径检查；安装检查不要借助设置 PYTHONPATH 掩盖失效的 editable 安装。
- 如现有入口测试未覆盖 `train_gpu` 等新 worker 名称，可扩展既有入口测试，只验证 `--help`/安全退出和不提前加载 torch/设备包，不添加镜像实现测试。
- 包装变更可用临时目录构建 wheel 检查包清单与 console entrypoint；不发布、不升级依赖。
- 测试通过后不再全量重复。独立验收以 diff 与结果审阅为主。

审计指出一个必须保留的历史路径：`learning/gradient_probe_evidence.py` 的 `git show {R1_COMMIT}:src/cnsbridge/{name}` 读取旧 commit；同文件当前工作区的 `ROOT / "src/cnsbridge"` 则需迁移。不可把历史路径误当遗漏。

## 回滚与实验资产

源代码迁移可通过版本控制恢复；不修改工作区根路径，不 reset 用户文件。旧数值结果及 checkpoint 留在原路径，原 commit 仍可用于重建原实验软件。新命名不提供旧 hash-bound run 的自动迁移或无条件恢复承诺。
