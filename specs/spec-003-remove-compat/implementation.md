# Implementation Report

## Summary

按 Spec 003 删除旧导入与脚本转发层，当前文档和入口检查统一使用真实业务模块。真实运行逻辑不变。

## Files Changed

- 删除 `src/cnsbridge/motor/` 的 7 个转发模块、根 `src/cnsbridge/safety.py` 与 `calibrate.py`、`tools/` 的 7 个转发脚本，共 16 个源码文件。
- 删除上述模块的本地缓存；`motor/` 与 `tools/` 目录整体移除，根包中仅移除对应 `safety`、`calibrate` 缓存。
- 更新 `README.md`、`docs/ARCHITECTURE.md`、`docs/DATASET.md`、`docs/TELLO.md`、`constitution/tech-stack.md`、`.ai/rules/project.md` 中的命令、目录职责与兼容说明。
- 更新 `src/cnsbridge/experiments/tello/check.py` 文档字符串中的启动命令。
- 更新 `tests/test_business_layout.py`：删除 10 个旧别名兼容用例，七个帮助入口与两个无起飞标志入口改为 `python -m`；保留可选依赖阻断、异 cwd、真实资产根、CLI 和特征身份断言，基础导入改为真实控制与学习模块。

## Important Decisions

执行 [Decision 003](../../.ai/decisions/003-remove-compat.md)：停止旧 Python 导入与脚本路径兼容，不增加包装。保留现有五个 `cnsbridge` 子命令、`control/legacy.py`、校准和控制限制实现、序列化格式与实验资产。

## Validation

- Risk Level: S3（有意移除公开入口）；验证按 Plan 限于入口与导入边界。
- Checks Performed:
  - 阅读修改差异，确认业务代码仅有 `check.py` 的帮助文案变化。
  - 静态搜索 `src`、`tests`、`examples`、README、当前 docs、constitution、项目规则与 `pyproject.toml`：唯一剩余旧工具路径是来源身份测试的迁移前基准注释，无仍在使用的旧导入或启动命令。
  - `.venv/bin/python -m pytest -q tests/test_business_layout.py`，执行一次，进程退出码 `0`：`12 passed in 1.13s`。
- Result: 计划中的定向检查通过；`git diff --check` 留给独立验收者，不重复运行完整测试、构建或 lint。

## Known Limitations

旧工具路径及旧 Python 导入按需求失效，调用方应改用新模块入口。未连接设备、下载数据或执行训练；离线入口测试不证明实飞效果。

## Remaining Questions

无。交由新的独立验收 Agent 验收，提交由协调者执行。
