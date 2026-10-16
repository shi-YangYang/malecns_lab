# Plan — 移除 MiniFly

状态：Done。基线 afff8af，初始工作区干净。一次定向检查 99 passed、131 subtests passed，独立验收 PASS，无返工项。

## 范围与顺序

1. 协调者记录 Ready Spec/Decision/Handoff。
2. 实施 A 修改 src/ 与配置：删除生成器、所有默认和别名路径，维护真实模型来源与现有控制边界。
3. 实施 B 仅更新相关 tests，真实脑集成使用现有全图；源修改就绪后一次运行受影响子系统，不默认全项目。
4. 实施 C 仅更新 README/docs 中直接相关内容。协调者治理及删除已知上次演示资产/缓存。
5. 全新验收者审查变更和实际验证证据，不重复通过检查；PASS 后协调者 commit。

## 验证计划

Risk Level S3：删除公开 API、改全局脑来源和设备入口配置。只运行与此边界有关的 brain/CLI、shadow/trial、prepare/business-layout、diagnostics 检查。保留计数 fixture 和模拟设备，不连接实体。真实模型用已有完整 MaleCNS 短步进；无真实录像，不以新造视频冒充实录结果。MiniFly 特有的硬编码行为断言直接移除，不能套到真实模型。检查失败才定向返工并重跑受影响检查。一次 diff --check 与当前代码引用审查后停止。

## 来源变化与风险

Brain 入口源文件移除别名自然改变训练 provenance hash，应重新提取特征，不能保留旧 MiniFly 身份。运行时间可能增长，既有试飞 timeout 保留，未经实体复测不宣称旧模式有效。输出目录/格式与动作限制不变。

## 分工与交付

实施 A 拥有 src/configs 和 implementation.md；B 拥有 tests 并提供验证报告给 A；C 拥有 README/docs；协调者拥有治理和指定本地资产清理。独立验收者仅 acceptance.md，协调者提交。
