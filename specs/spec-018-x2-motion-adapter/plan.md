# Plan — 最小动作适配器

状态：Completed。最终第四轮实测及新独立验收均PASS，见 implementation.md 与 acceptance.md。

1. 协调者明确行为、验证和环境边界，保持仿真关闭。
2. 同一实施者处理耦合的控制核心、ROS仿真入口与定向测试，报告真实检查；不启动仿真。
3. 协调者阅读实现，准备源文件挂载与日志，在一个短窗口执行GPU仿真集成，立即关闭后分析。
4. 新独立验收者只读复核代码与证据；FAIL交新实施者限定修复，新验收者复核；PASS后更新状态并停止。

集成进展：首次启动因覆盖ROS的PYTHONPATH在import前退出；保留日志，随后使用prepend。首轮接口集成ADD成功但零发布不能切换输入源，未发非零，按超时退出，GPU容器与VM随即关闭。官方随包文档确认该仲裁语义，Spec修正为有界CLAIMING；由新实施者限定返工，保持旧源码快照与失败证据，相关检查通过后允许一次针对复验。

授权文件：src/malecns_lab/control/x2.py、src/malecns_lab/experiments/x2.py、tests/test_x2_motion.py；必要的本Spec报告/索引/交接。禁止无关重构、全局依赖变化、改官方组件或其他用户工作。原始数据与日志只写本轮忽略目录。
