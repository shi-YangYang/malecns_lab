# 迁移到 malecns_lab

2026-10-20 起，展示名为 **MaleCNS Lab**，分发包、Python 包与 CLI 均为 `malecns_lab`。此次仅调整工程命名与导航，不更改神经方程、连接图、优化器或动作语义。依据见 [Spec 009](../specs/spec-009-malecns-lab/spec.md)。

## 新入口

| 旧入口 | 当前入口 |
|---|---|
| `src/cnsbridge/` | `src/malecns_lab/` |
| `import cnsbridge` | `import malecns_lab` |
| `cnsbridge --help` | `malecns_lab --help` |
| `python -m cnsbridge.learning.train_gpu --help` | `python -m malecns_lab.learning.train_gpu --help` |

不保留旧包、旧 CLI 或转发兼容层。内部模块职责不变。2026-10-20，用户已将本地 checkout 从 `/Users/yang/project/FlyDrones` 改名为 `/Users/yang/project/malecns_lab`，并更新 Codex 中的项目名称和关联目录。当前操作均从新根目录执行；历史记录中的旧绝对路径仍保留原样。

## 已有虚拟环境

目录移动不会自动改写虚拟环境里的绝对路径。`.venv` 与 `.venv-train` 都需要修复指向旧根目录的启动脚本、激活脚本及安装记录，并重新安装本地 editable 包；仅更新 Codex 项目设置不能代替这些步骤。迁移后应重新打开终端，避免沿用已激活环境中的旧路径。

本机两个既有环境的路径修复与 editable 重装已完成，包导入、内置资源、CLI、激活脚本和项目根目录定位均已检查通过，未重新下载或升级依赖。`pyvenv.cfg` 中的 `command` 仍记录原创建命令，仅作为历史保留。

以后若再次迁移目录，先修复环境路径，再在新项目根目录使用该环境自己的 Python 更新本地 editable 安装，不用系统 Python：

```bash
cd /Users/yang/project/malecns_lab
.venv/bin/python -m pip install --no-deps --no-build-isolation --no-index --disable-pip-version-check -e .
.venv/bin/python -m malecns_lab --version
.venv/bin/malecns_lab --help
```

训练环境同样执行：

```bash
cd /Users/yang/project/malecns_lab
.venv-train/bin/python -m pip install --no-deps --no-build-isolation --no-index --disable-pip-version-check -e .
.venv-train/bin/python -m malecns_lab --version
```

这些命令假定环境已经具备项目依赖及构建后端，不安装或升级依赖。新环境请依项目首页和训练文档安装所需 extras。

若环境仍登记旧 `cnsbridge`，先用该环境的 `python -m pip show cnsbridge` 核对 editable project location 确实是本 checkout，再运行 `python -m pip uninstall cnsbridge` 清理旧安装入口。不要卸载其他项目的同名包。仅安装新分发包不会自动删除不同名字的旧包元数据和 CLI。

## 格式保留，源码身份改变

已有 `cnsbridge.*`、`flydrones.*` 的 schema、format 和模型契约标识继续保留。它们是磁盘数据协议，不是旧 Python 导入入口；不要手工替换 checkpoint、JSON 或 NPZ 中的名称。

已有数据、训练结果、模型、日志及登记文件保持原样。代码中的科学来源归属文字、历史 `git show` 路径也保留，例如旧提交的 `src/cnsbridge/...` 必须仍指向当时源码。

**格式可识别，不表示新源码能原位恢复旧训练。** 源码路径及导入文本的变化会改变相关来源摘要。严格 hash 准入仍然生效；不得改写旧 run、伪造摘要、跳过校验或把拒绝恢复当作待绕过的错误。

精确复现旧实验，应在单独 checkout 中使用报告对应的原 commit、原环境及原始实验资产。若以后需要将旧参数迁移到新代码，须另立记录父来源、参数、运行时和优化器／状态处理的迁移协议。此次没有实现自动模型迁移。

## 文档、Git 与本地资产

当前 README、使用文档、示例和测试已采用新入口；旧 Spec、Decision、审计及实验报告保留历史名称。历史报告链接到当时路径时，请结合记录的 commit 阅读。

保留 Git 历史、LICENSE 和第三方署名。`data/`、`logs/`、模型、录像和虚拟环境仍是本地忽略资产，不通过普通源码提交上传。新 GitHub 仓库的创建、远端配置和发布是独立操作，此次没有执行。
