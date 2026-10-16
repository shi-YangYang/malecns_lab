# Implementation Report

## Summary

删除 MiniFly 生成器、包导出和运行别名，所有脑处理入口默认使用真实 MaleCNS NPZ。诊断与数据准备不再提供合成网络检查路径；shadow 和全部 trial 模式使用真实图与校准读出。

## Files Changed

- 删除 `src/cnsbridge/brain/synthetic.py`；更新 `brain/__init__.py`、`brain/brain.py` 和包内默认配置。
- `paths.py`、`config.py` 统一模型与读出相对工程根解析；CLI 构建、下载、校准资产路径与读取规则一致。
- `experiments/tello/shadow.py`、`trial.py` 更新模型入口与来源日志；保留原试飞动作权限、各模式控制/相机/安全参数和时长。
- `learning/prepare.py`、`diagnostics/activity.py`、`collector.py`、`reporting.py` 和 `senses/encoder.py` 清理合成来源、来源哈希清单与说明；`configs/tello.yaml` 更新模块入口注释。
- 文档实施者更新 README 与 `docs/{ARCHITECTURE,DATASET,NEURAL_ACTIVITY,RESEARCH,TELLO}.md`，移除当前 MiniFly 工作流，说明真实模型、路径、校准和来源身份迁移。
- 测试实施者更新 brain/CLI/loader/control/business-layout、prepare、diagnostics 和 Tello shadow/trial 相关测试；新增真实图 fixture，缺真实资产时明确 skip，不生成替代网络。协调者维护治理记录并删除上一轮指定演示目录和孤立缓存。

## Important Decisions

- `load_connectome` 只读 NPZ；缺失资产明确报错，带 `synthetic` 真标记的图拒绝加载，无替代网络。
- 默认 CLI/shadow 配置采用 `configs/malecns.yaml`，校准命令默认写入 `readout_malecns.json`；trial 必需资产在 SDK 创建前检查。
- `forward-check` 保持固定 RC 探针属性；其被观察的脑同样改用 MaleCNS，不将固定探针称为神经控制。
- LIF 和 Retina 数学实现未改。Brain 入口和 encoder 注释变更自然改变新特征来源 hash，旧数据身份不伪装兼容；必要时重新提取特征。

## Validation

- Risk Level: S3；公开脑 API 删除、默认来源和多个入口改变。
- Checks Performed: 实施者阅读源 diff 与引用，文档实施者 S0 阅读文档 diff；测试实施者一次执行如下受影响子系统命令，未重复已通过检查：

```bash
.venv/bin/python -m pytest -q -s tests/test_brain.py tests/test_cli.py tests/test_malecns_loader.py tests/test_control.py tests/test_business_layout.py tests/test_prepare_motion_dataset.py tests/test_neural_activity.py tests/test_tello_shadow.py tests/test_tello_yaw_trial.py tests/test_tello_reflex_trial.py tests/test_tello_cruise_trial.py tests/test_tello_forward_trial.py tests/test_tello_full_trial.py
```

- Result: exit 0；99 passed、131 subtests passed、8 warnings，14.65 秒。完整真实图实际加载与 50 ms 步进输出为 166700 neurons、10520377 connections、8775 spikes，缺失群组为 HAL_L/HAL_R。CLI bench、真实 worker spawn、ShadowBrain 和 diagnostics 实际图路径已覆盖。
- diagnostics 的临时 6 帧人造视频只用于真实图处理接口回归，未保存演示，不代表真实拍摄刺激响应。prepare 录像对齐使用明确 CountingReplay 计数 mock，不保留合成飞行网络。
- 未连接硬件、下载或训练。最终验收由新的验收 Agent 负责。

## Known Limitations

当前没有真实地面录像会话。真实连接组加载和短步进检查只能证明软件路径，不证明录像刺激方向、飞行效果或避障有效性。完整模型的运行开销增加，未修改超时门限。

## Remaining Questions

无。
