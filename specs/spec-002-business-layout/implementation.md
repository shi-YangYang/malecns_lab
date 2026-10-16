# Implementation Report

## Summary

完成业务实现迁移：动作与旧控制归入 `control/`，新读出和数据流程归入 `learning/`，Tello 实验归入 `experiments/tello/`。旧七条工具路径保留为调用 `main` 的薄入口；旧 Python 导入显式导出同一实现对象。未提交，等待新的独立验收 Agent 审查。

## Files Changed

- `src/cnsbridge/control/`：body、command、adapters、legacy、calibration、safety 真实实现。
- `src/cnsbridge/learning/`：decoder、dataset、prepare、train 真实实现。
- `src/cnsbridge/experiments/tello/`：check、hover、shadow、record、trial，以及从原脚本提取的 policy、worker、session。
- `src/cnsbridge/paths.py`：按实际 `pyproject.toml` 与 `src/cnsbridge` 布局定位源码工程根，不依赖启动 cwd，也不将 site-packages 误认为仓库。
- `motor/*`、`safety.py`、`calibrate.py`：显式兼容导入；`tools/*.py`：七个命令启动入口。
- CLI、离线仿真、示例、原测试：改为新实现路径；试飞配置仍定义于 trial，原 PROJECT patch 对真实实现生效；纯策略测试直接访问 policy，spawn 测试直接使用 worker.BrainProcess。
- `tests/test_business_layout.py`：旧导入对象身份、七个工具异 cwd 帮助、两个飞行入口无起飞标志、原 CLI、低层无可选依赖急加载、源码资产路径及迁移前特征身份。
- README、ARCHITECTURE、DATASET、TELLO、constitution/tech-stack：同步职责、兼容方式、源码资产定位和 editable 安装边界。

## Important Decisions

- `trial_config` 留在 trial，配置根与命令处理仍由该真实模块拥有；未通过 re-export 假装修复 mock。
- policy 只放原纯策略与常量；worker 不访问 SDK；session 共享遥测、输入新鲜度、起飞检查与降落清理。原函数体完整提取，未改变 RC 数值、门控、时限或清理顺序。
- 原 check 顶层 OpenCV/DJITelloPy 导入移到参数处理之后，让 `--help` 无需视频或硬件依赖。实际地面检查流程不变。
- Brain/Retina 四个来源文件保持字节不变。RecordingBrain 的哈希计算、特征顺序、模型/数据序列化标识保持不变。
- 不新增多层业务包装；旧入口只为现有外部调用提供迁移兼容。未变更依赖、版本、算法、飞行默认时长或实验资产。

## Validation

- Risk Level: S3。跨控制、学习、真实包导入、进程与设备监督模块的结构迁移。
- 在修改前，用原 `tools/prepare_motion_dataset.py` 与 `default_config()` 获取独立基准：MiniFly 的 source ID 为 `ground-replay-sha256:17277d1fcd8e9f68ac683ff913a1a7f6f51534060fcf3da6e507d486bf513ef5`，输出特征为 407–424、832–849 号神经元，50 ms 窗口。新回归测试核对此旧基准，不比较两个新实现别名。
- 用 Git 基线 `8546f6c` 内容逐字节核对 `brain/brain.py`、`brain/lif.py`、`senses/retina.py`、`senses/encoder.py`：四个文件不变。
- 对原 `tools/tello_yaw_trial.py` 和 `tools/tello_hover.py` 的所有函数/类与提取后的真实模块进行 AST 比较（忽略源码位置）：全部一致。清理、warmup、RC 循环与门控未重写。
- `.venv/bin/python -m pytest -q`：**144 passed、251 subtests passed，退出码 0，2.40 s**。原 122 个测试保留实质断言；22 个新增参数化迁移用例通过。真实 spawn 回归也已运行，所有设备调用均为假 SDK 或在导入前被阻断。
- pytest 先有一次调用只捕获到运行中输出片段，未保留会话和退出码，结果不能确认为通过。根据“上次结果不明确”的停止规则，重新运行一次获得上述确定结果；没有发现或修复测试失败，也未继续重复测试。
- 唯一 warning 为原 MaleCNS 小型测试夹具缺失神经群组的提示；不是新增实机或完整数据结果。
- `git diff --check` 留给独立验收；未执行构建、lint、实体连接、下载、训练或新的实飞验证。

## Known Limitations

实验工具依赖源码 checkout 的 configs/data/logs；此次不新增 wheel 中的实验资产分发体系。旧入口继续保留，未来删除需另定范围。离线回归只验证迁移契约，不代表实飞认证或新读出有效性。

## Remaining Questions

无阻塞实施问题。等待独立验收；提交由协调 Agent 在验收后完成。
