# Plan — 删除转发层并统一入口

状态：Done。实施完成，定向入口测试 12 passed，独立验收 PASS，无返工项；见 [验收报告](acceptance.md)。

## 范围与顺序

一名实施 Agent 负责 16 个兼容文件删除、对应缓存清理、入口测试更新及 README/docs/tech-stack/项目规则；主 Agent 负责 Spec、交接、Decision 和历史替代说明。真实业务文件仅 check.py 的帮助示例更新。

## 接口与数据流

调用统一到 Python 模块入口：tello 的 check/hover/shadow/record/trial，以及 learning 的 prepare/train。无数据流变化；有意停止旧 tools 与 motor/safety/calibrate 转发导入。现有 cnsbridge CLI 不变。

## 验证

Risk Level: S3（有意移除旧公开导入和命令路径），但不改运行实现，验证集中在入口边界：

- 静态核对 src、tests、examples 与当前文档中无仍在使用的旧导入或命令，历史记录及迁移前基准注释允许保留。
- 仅执行 `.venv/bin/python -m pytest -q tests/test_business_layout.py`，覆盖七个 `-m` 帮助、两种无起飞入口、低层导入、CLI、资产路径及来源身份；使用既有阻断 SDK/OpenCV 的子进程机制，不连接无人机。
- 不重跑完整 pytest、build 或 lint；独立验收审查删除范围与测试证据，执行一次 git diff --check，不重复已通过定向测试。

## 风险与迁移

主要风险是残留文档命令、旧 motor 的缓存目录成为命名空间或新模块入口漏检。删除对应目录缓存、更新现有入口回归并核对当前引用解决这些具体风险。旧调用方需换成文档中的新入口；不再保留转发层。无需数据迁移，也不改历史提交。

## 交付

实施者写 implementation.md；新的独立验收 Agent 写 acceptance.md。PASS 后协调者更新状态并 commit。
