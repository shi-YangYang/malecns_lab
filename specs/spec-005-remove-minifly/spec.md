# Spec — 移除 MiniFly，统一真实连接组

状态：Done。用户明确要求删除 MiniFly，只用真实连接组；实施及独立验收完成，AC1–AC4 全部 PASS，本轮按既有授权提交。

## 背景 / 当前行为

MiniFly 仍存在于生成器、导出、默认配置、CLI、Tello 观察/试飞、诊断和测试文档中。本地 data/malecns_brain.npz 的元数据为 malecns-v1.0，166700 个神经元；没有真实地面录像会话。

## 目标 / 目标行为

删除 MiniFly 实现及可调用入口，所有实际脑处理默认加载 MaleCNS NPZ，缺失或显式合成资产应失败，不提供替代网络。删除上次生成的 MiniFly 演示资产，更新当前文档与直接受影响测试。

## 非目标

不启动硬件、不采集新视频、不重新下载或构建图、不改变动作限制/试飞时长/神经动力学；不把检查图形或纯数学 fixture 称为真实实验，不重写旧验收历史。

## 功能需求

1. 删除 brain/synthetic.py 与 build_minifly 导出；load_connectome 只读 NPZ，拒绝已标记 synthetic 的资产，无别名或 fallback。
2. defaults、CLI、shadow、trial、prepare、diagnostics 使用真实 MaleCNS 路径；模型与需要的校准文件相对工程根解析，缺失明确失败。全部脑驱动 trial 模式使用 MaleCNS，保留各模式动作权限、阈值和时长。
3. 删除当前文档中的 MiniFly 使用说明、默认说明与示例；报告不再推荐该模型。通用解码器数学示例不属于 MiniFly，不顺带删无关模块。
4. 替换或删除依赖 MiniFly 的测试与硬编码来源哈希。纯数据/安全单元检查可使用明确的计数 fixture、mock，但不可换名保留合成飞行网络。真正脑集成检查使用本地完整 MaleCNS；无真实资产时明确 skip，不 fallback。
5. 删除上一轮已知生成的 data/neural_activity_demo/synthetic_32459262ba/ 和删除模块的 pycache；保留用户真实模型/视频/历史飞行日志。
6. 记录接口删除导致来源身份变化：brain.py 与 encoder 注释可能更新，重新提取的特征自然生成新 hash；旧 hash 不伪装兼容，不改旧数据。

## 技术约束 / 兼容性

用户新指令替代 Spec004 的 MiniFly 验证路径和四文件字节不变约束。保留 LIF/Retina 数学实现、旧序列化标识和真实资产。旧 specs/验收保留历史事实并在当前交接标记已被本轮替代。

## 边界条件

相对路径在其他 cwd 也解析到源码工程。真实图缺 HAL 群组可报告缺失，不能套用 MiniFly 的全组存在/行为方向断言。缺真实录像时只报告真实图加载和神经步进、接口检查，不能声称完成真实视频响应验证。

## 验收标准

- [x] AC1：MiniFly 源文件/导出/运行别名已删除，当前源码无可用 MiniFly 路径；显式 synthetic 图不被加载。
- [x] AC2：所有脑入口默认真实 MaleCNS，模型/校准缺失不回退，trial 动作限制不变。
- [x] AC3：当前文档、来源采集与测试更新；历史记录可追溯，已知上次 MiniFly 演示与孤立缓存清除。
- [x] AC4：受影响子系统定向检查通过，包含本地完整 MaleCNS 的实际加载和短神经步进；独立验收后 commit。

## 待确认问题

无。用户删除要求明确；不以再次确认阻塞实施。
