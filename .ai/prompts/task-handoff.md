# 当前交接 — Electron 工作台正式接入已完成

更新：2026-10-23。工作目录 `/Users/yang/project/malecns_lab`。

## 当前状态

用户认可精简原型并要求开始实施。[Spec022](../../specs/spec-022-electron-workbench/spec.md) 的 P1/P2 已完成，[独立验收 AC1–AC6 PASS](../../specs/spec-022-electron-workbench/p1-p2-acceptance.md)。没有未处理返工或待运行验收；停止继续实验。用户随后授权本地 commit，当前提交封存本阶段源码与文档，不推送。

桌面已从原型素材改为同一冻结 MPS 控制脑的真实活动与官方 GPU 仿真，保留用户认可的左右布局，不恢复已删除的大标题、底部播放条及历史小字。默认仅载入真实胞体坐标；点击准备才启动完整脑和自有仿真，官方界面 Reset/Run 后点击开始执行两轮纵向反馈。停止、退出、失败均走有界停车/清理路径。

22个真实参考群组按实际 spike 统计；33个输出神经元中选32个记录真实逐步脉冲。颜色代表群组均值，栅格显示选样脉冲；没有全脑逐步归档、插值或另起CPU脑。每50步（5ms神经时间）采一批，显示跳批与归档丢批分开。

## 已通过的证据

- [正式接入实测](../../specs/spec-022-electron-workbench/actual-integration.md)：完整图开关遥测分别对同一原baseline的4个递归边界和3个完整输出逐元素精确一致；耗时增加0.1666%，峰值RSS增加62.4375MiB，符合冻结15%/128MiB预算。配对仅一次，不作统计性能承诺。
- 原生桌面成功运行 `data/x2_sim/agibot/data/var/desktop-20261023-034602-5d90cd78/`，run_id `e2e5ba357a674e8895c8279ce58264e9`。两轮FORWARD位移0.381064m/0.446197m，新观测来自上轮动作后，逐轮owned_core_STOPPED；最终停车、资源清理和parameters_unchanged均为true。
- 32/32遥测批次完整、零丢批/错误，428612bytes；同一run各身份一致。实际回放第28批53.3Hz和5个spike与原始记录相符。截图 `logs/desktop_workbench/live-running.jpg` 和 `replay.jpg`，汇总 `desktop-integration.json`。回放无仿真视频，左侧明确缺少视频。
- 宿主87.508s/VM80.332s，符合240s/120s预算。完成后只读检查私有Lighter未运行、旧CPU容器exited、默认context仍desktop-linux。完整脑已退出，未动Noria或重启Docker。Electron保留在离线回放，重资源关闭。
- 已完成遥测、runtime、IPC/存档边界的定向检查和必要桌面构建，不再重复。实时summary完整性和SVG填充已修复并独立通过；官方SD仍要求明确成功响应，预算内重试且每次返回落盘，实际第二次成功。历史失败均保留，不冒称停车成功；详情见实测。

## 入口与边界

- 已构建开发入口：`cd apps/desktop && npm start`。首次准备/修改构建按 `apps/desktop/README.md`。Electron 45.0.0官方本地依赖，尚无安装包/签名/跨平台交付。
- 固定完整脑入口 `.venv-train/bin/python -m malecns_lab.experiments.x2_desktop --execute --sim-only --output <新desktop-run>` 由主进程封装；renderer无任意命令或路径权限。普通 `.venv` 不含Torch。
- 私有runtime `data/x2_sim/gpu_probe/runtime-rebuild`，容器 `malecns-x2-gpu-trial`，noVNC 6081；不接管已运行资源，不启动旧CPU仿真，不切换Docker context。用户要求不用仿真时关闭以减少发热。
- 范围仍是里程计虚拟标记、纵向离散动作、停稳后计算。没有训练、真实相机、连续控制、真机、SDK改造、全脑逐步脉冲或仿真视频归档；GUI帧率不是脑计算频率。本次共驻留每帧计算约3.64秒。
- 本地提交仅包含源码、配置、锁文件与文档；模型、资产、依赖安装目录、日志及失败run保留在本地且不入库。下一阶段尚未授权实施；不要覆盖或清理科研资产，不重做已通过验证。

## 保留的工程基线

Spec016–018 官方动作/停车契约与适配器、Spec019冻结推理、Spec020精确等价优化、Spec021两轮纵向反馈均已PASS。Spec021本地提交 `fbc5b2d`，原实测0.41884m/0.42901m后逐轮停稳；完整166700节点、25582938非零边、124177617原始连接计数。旧基线数值与失败记录见各Spec，不以桌面完成扩大科学或真机结论。
