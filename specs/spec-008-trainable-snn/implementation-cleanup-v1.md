# Implementation Report — V1 重复离线流程清理

## Summary

按 [cleanup-v1](cleanup-v1.md) 删除已被 V2 替代、未被旧设备链依赖的离线流程。地面录制程序、既有设备链、V2 冻结脑/读出和图对照保留；没有修改完整图、录像、运行记录、环境或许可。活跃文档转向 V2，并明确完整图内部训练仍属 Spec 008 实施工作，不冒充已训练结果。

## Files Changed

删除 14 个文件：

- `src/cnsbridge/diagnostics/activity.py`
- `src/cnsbridge/diagnostics/collector.py`
- `src/cnsbridge/diagnostics/reporting.py`
- `src/cnsbridge/diagnostics/stimuli.py`
- `src/cnsbridge/learning/prepare.py`
- `src/cnsbridge/learning/dataset.py`
- `src/cnsbridge/learning/train.py`
- `src/cnsbridge/learning/decoder.py`
- `examples/04_shared_motion_decoder.py`
- `docs/NEURAL_ACTIVITY.md`
- `docs/DATASET.md`
- `tests/test_prepare_motion_dataset.py`
- `tests/test_motion_dataset.py`
- `tests/test_neural_activity.py`

更新：

- `src/cnsbridge/learning/__init__.py`：删除 V1 公共重导出，不建立别名或兼容包装；V2 使用所属模块直接导入。
- `tests/test_business_layout.py`：将活跃学习/诊断入口改成 V2；保留从外部 cwd 启动、无硬件导入、显式起飞开关和设备资产解析检查；删除仅服务 V1 录像特征的检查。
- `tests/test_universal_decoder.py`：删除 V1 拟合/序列化检查，保留全部动作契约、方向、支持轴、限幅和底盘几何检查；原“先训练再共享”的末项改为直接检验同一 SI 目标的双设备数值方向，不暗示迁移成功。
- `README.md`、`docs/ARCHITECTURE.md`、`docs/RESEARCH.md`、`docs/TELLO.md`：现行流程、模块、命令和数据边界转向 V2。ARCHITECTURE 内保留直接 BodyVelocity/设备适配示例。
- `docs/MALECNS_AUDIT.md`：只把即将失效的两条 V1 源码链接改为原审计基线 `747a8aa:路径:行号`，保留历史发现。
- `docs/MODEL_V2.md`：将设计时的 V1 读出引用标明为历史，并指向现有 V2 实现；不更改研究结论。

新的采集说明位置：`docs/TELLO.md#地面录像与会话格式`。保留录制命令、5–60 秒范围、退出方式、新目录要求、视频/CSV/JSON 格式、20 Hz 播放与主机取帧时钟区别、30 秒刺激安排、场景/素材族规则和 V2 标签入口。旧 `stimuli.csv` 不被 V2 读取这一差异明确说明。

## Important Decisions

- 删除依据为调用关系和已存在的 V2 替代实现，不以文件旧或数量多作为理由。
- `learning/__init__.py` 只保留包说明。旧导入直接失效，不做兼容。
- 原地面记录仍使用 `flydrones.ground_recording` version 1；V2 对旧两轴区间标签的显式导入保留，这是有效资产协议而非本次要删的 V1 执行入口。
- 历史治理文档保持原样。协调者负责 constitution 中当前入口更新；本实施者未编辑该目录、P0/P2a 核心、pyproject 或其他并行改动。

## Validation

Risk Level: S2（跨 V1 包导入、入口、相关测试和文档清理）。

实际执行：

- 阅读并检查范围内 diff；`git diff --check` 无错误。
- `.venv/bin/python -m pytest tests/test_business_layout.py tests/test_universal_decoder.py`：**20 passed in 1.33s**。
- 在 `src/ tests/ examples/ docs/ README.md THIRD_PARTY.md` 定向搜索被删模块、命令、文档和演示引用；活动源码无残留导入，当前文档无已删除命令推荐或文件链接。仅保留审计/设计中明确标记的历史读出名称及带提交身份的历史源码引用。
- 对源码/测试/示例的相对和绝对 Python 导入再次定向搜索被删的模块名，无匹配。

未运行完整测试套件、完整脑诊断、训练或硬件；不存在本子批验证所需的此类运行。定向检查通过后没有重复运行。

## Known Limitations

本子批仅完成 V1 重复离线流程清理。旧设备链、旧房间仿真和过滤图构建仍有当前调用者，按后续替代子批处理；不宣称整个清理或可训练 SNN 目标已完成。V1 源码可从 Git 历史查看，现行包不再执行这些入口。

## Remaining Questions

无实施阻塞。等待协调者安排新的独立验收并按用户要求创建独立提交；实施者未 commit。
