# Spec — MaleCNS Lab 仓库命名与入口整理

状态：Done。2026-10-20 用户要求先整理仓库，并明确选择“项目和 Python 包名一起改成 malecns_lab”；实施完成且新的[独立验收](acceptance.md)为 PASS。

## 背景

项目已从 Tello 单设备实验扩展到完整 MaleCNS 网络研究，下一方向为神经活动展示及智元机器人接入。当前展示名、分发包、Python 导入、CLI 和源码目录仍为 CNSBridge / cnsbridge；README 将目标跟随写成项目的唯一目标，文档缺少分类入口。

## 目标

- 展示名使用 MaleCNS Lab，分发包、Python 包、CLI 统一为 `malecns_lab`。
- 将 `src/cnsbridge/` 迁移为 `src/malecns_lab/`，更新实际导入、子进程入口、路径定位、测试、示例和当前使用文档。
- 整理 README 与文档导航，区分当前能力、历史实验、拟议的 WebGL 展示及 G2/X2 接入。
- 保留完整连接组、模型、训练记录、既有来源和版权，明确改名对来源身份的影响。

## 非目标

本批不实现 WebGL、机器人 SDK、LLM 控制或新训练任务；不删除现有科研模块、数据或历史报告，不启动训练、全图探针或实体设备。不创建 GitHub 仓库、不推送、不重建 Git 历史，不移动当前 checkout 根目录。

## 当前行为与目标行为

当前 `cnsbridge` 命令、`python -m cnsbridge...` 以及 `src/cnsbridge` 路径由新入口替代：

```text
MaleCNS Lab
  src/malecns_lab/brain       连接组与神经计算
  src/malecns_lab/senses      感觉编码
  src/malecns_lab/diagnostics 神经活动记录与分析
  src/malecns_lab/learning    读出、内部学习与实验协议
  src/malecns_lab/control     动作契约与设备数值适配
  src/malecns_lab/drones      已有目标跟随环境
  src/malecns_lab/experiments 已有 Tello 实验
```

内部职责保持，不创建空的 frontend、robot 或 provider 目录。通用项目方向不等于已实现通用控制能力。

## 功能需求

1. `pyproject.toml` 的项目名、console script、package data 使用新包。`malecns_lab --help/--version` 与 `python -m malecns_lab` 可用；不保留旧包或旧 CLI 转发兼容层。
2. 所有可执行 Python 导入、动态导入、`-m` worker、示例源码定位和当前使用命令使用新命名；`source_project_root()` 仍从任意 cwd 正确定位此 checkout。
3. 当前使用文档统一新命名；README 简明介绍完整连接组研究、神经可观测性和设备适配，目标跟随作为现有实验保留。增加 `docs/README.md` 分类导航和迁移说明。
4. 增加简短智元资料入口文档：X2 官方开发站、SDK 获取和仿真指南；G2 厂商 GDK v2.1 PDF（明确第三方托管及版本限制）和官网开发者工具说明。明确仅为接入候选，仓库尚无对应驱动。
5. 既有 `cnsbridge.*`、`flydrones.*` 的 schema/format、模型标识规则等序列化契约不因品牌替换而改变。它们不是 Python 导入兼容层；保留原来的持久化字段与严格校验。
6. 既有完成或未完成实验的源文件摘要和历史登记不得重写。源码改名或导入文本改变会产生新来源身份；不得放松 hash 校验来使旧 run 在新源码下冒充续训。旧 run 的精确恢复须使用原 commit 与原环境，新迁移另立协议。
7. `.venv`、`.venv-train` 若安装的是此 checkout 的旧 editable 包，在不升级依赖的情况下安装新 editable 包、移除旧项目的安装入口。不得卸载其他项目的同名包或修改系统 Python。
8. 当前未跟踪的 `specs/spec-008-trainable-snn/scipy-backward-boundary-audit.md` 原样保留，不纳入本批变更。旧 Spec、Decision、运行报告和 handoff 历史段不作品牌批量替换；本批以新记录明确覆盖当前工程约定。

## 边界条件与技术约束

- 沿用 Python src layout、配置/测试/文档位置，业务目录迁移仅为用户明确授权的包名调整。
- 所有完整脑计算仍使用真实数据；不得生成替代图以演示“迁移成功”。
- 不改网络方程、优化器、输入映射或动作语义，不升级依赖，不安装机器人或前端依赖。
- 已有绝对本地路径、旧数值结果、来源 URL、LICENSE 的上游署名不被机械替换。

## 兼容性要求

旧导入和旧 CLI 有意移除；当前使用文档提供新入口。持久化 schema 保留不代表新源码可原位续跑旧 hash-bound 训练。新旧科学来源隔离须在 README 或迁移说明明确说明。

## 验收标准

- [x] AC1：仅新包/新 CLI 存在，editable 安装和模块入口有效。
- [x] AC2：当前执行路径、worker、示例、测试和使用命令完整迁移，无遗漏旧运行入口。
- [x] AC3：源码工作区定位及离线入口检查通过；验证不接硬件、不加载完整图。
- [x] AC4：持久化格式字符串、数据与历史实验资产保持，来源哈希准入未被放松。
- [x] AC5：README、文档索引和当前治理入口一致；WebGL/G2/X2 明确为待实现方向。
- [x] AC6：版权/来源、已有 Git 历史及用户未跟踪文件保留；本批不推送。

## 待确认问题

命名与本批整理范围已足以实施。机器人具体型号、现场版本、新交互任务和三维展示细节属于后续 Spec，不阻塞本批。
