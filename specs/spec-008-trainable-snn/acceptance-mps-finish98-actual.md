# Acceptance — GPU完成剩余98窗的实际证据

## Result

**PASS**。由未参与本批执行的新独立验收Agent检查实际保存证据。`data/neural_training_mps_20261019_r3` 从合法62窗边界完成至 **160/160窗**，本批新增98窗；累计为CPU继承60窗、GPU100窗。25段均终态成功，无跳窗、重复窗或未封存尾部。

通过的是既定训练预算、来源、续接和自身数值证据验收；不是目标跟随成功、CPU/GPU轨迹等价、完整生物还原或实飞验收。阶段提交由协调者完成，Spec008整体研究目标保持未完成。

## Spec Coverage

- **E1，准确续接与完整预算**：读取原CPU60窗父身份及既有GPU61/62窗封存锚点；旧62窗仅检查已保存文件摘要、事件/末档父链和前缀，不重做旧窗口数值实验。本批segments3–27实际为24段×4窗和末段2窗，新增窗号恰为63–160；每段父状态均 `completed / registered_mps_windows_verified`，worker退出0，父/worker各有25个独立PID。最终 `epoch_complete=true`，未开始第二epoch。执行调度session95614的终态exit0已由执行Agent与协调者收取。
- **E2，25段完整状态恢复**：每段将上一末档与该段 `resume.npz` 的18个完整数组逐项比较shape、dtype和值，均精确一致：3个参数、9个Adam字段、6组神经/感觉状态，包括全部18×166700 pending。来源与恢复状态metadata、游标和时钟一致；每窗before参数/Adam与上一档相同，after与新封存档相同。
- **E2，场景边界**：第81窗由forward进入vertical、第121窗由vertical进入yaw。前一场景末态frame80、brain/drive40600；新场景起态frame0、brain/drive600，符合既定重置后预热；两次感觉event counts摘要均等于3377维全零整数数组，phase摘要相同。全部23项神经/感觉状态摘要按schema与身份检查，参数和Adam精确继承，Adam分别80→81、120→121。其余96个新窗口起态摘要与上一末态完全相同，没有场景内重置。这里检查保存摘要和已绑定reset实现，不重放预热来伪称独立重建了起始神经数值。
- **E2，逐窗自身数值**：复用原独立保存数组检查器检查新增98窗字段、输入/标签选择、有限性、gain/rho、读出动作、损失、raw/postclip梯度、Adam和实际参数更新；再用独立NumPy公式复算裁剪乘法、Adam矩/参数及归一化损失。原容差保持，全部通过；没有导入Torch或计算神经前反向。每窗theta确有变化，不能由此推断任务学会控制。
- **E2，完整图与来源**：流式核对完整图文件SHA，run绑定166700节点、25582938非零有向连接、124177617绝对突触计数，图角色为 `full_reference`。实际worker的完整图raw counts摘要前后保持。当前42份MPS来源、历史G29/R13/MPS20源码摘要全部相符；数据集文件/manifest及身份、全部320训练帧与场景顺序相符。图没有在验收进程中分配或执行，完整图实际构造证据来自受监督worker和绑定来源。
- **E2，资源与后端**：保持6GiB申请、额外1GiB预留、`min(6GiB, available−1GiB)`有效RSS上限、600秒/.1秒监督；25段无资源/数值失败。各窗封存后的RSS/MPS样本存在，MPS allocator上限与有效预算一致；终态artifact摘要全部相符。后端为MPS前反向和CPU主参数Adam/感觉事件，不称全部运算在GPU。共享锁在独立验收期间独占成功，没有启动并行模型、操作Noria/Docker或硬件。
- **E3，最终档与报告**：最终checkpoint、prefix及160步Adam、yaw末场景游标通过完整链检查；只读核对实施summary中新增98窗的时间分布、资源极值和各场景训练轨迹动作/标签/损失范围。README、roadmap、Spec索引、handoff与 `docs/MPS_TRAINING.md` 的当前状态差异符合证据；已完成run不再作为默认可续训命令。协调者可在此PASS后标记本子批Done并提交。

## Final Boundary

- run：`b7807f8e1997da44e46e83f39155e21bdce84aba020314a1ed68a145851413b7`。
- checkpoint：`fd3019989892f86bac44faefd5230ffff91891dbde9e813956c0945eb7b1e8b1`。
- path：`data/neural_training_mps_20261019_r3/segments/000027/windows/000160/checkpoint`。
- prefix：`1964b500ead62bceec06bf763b75c3e4992433bebb8b66f0d2fdea75d00c13a0`。
- context：epoch0、scene_index3、`train-sine-yaw`、next_frame80、completed_windows160、optimizer_steps160；brain/drive40600。
- CPU父边界仍为60窗，原父checkpoint、prefix及其文件摘要相符；本批未推进冻结脑或R分支。

## Validation

- Risk Level：本批没有业务源码修改，文档S0；实际执行证据沿用既有S3后端的保存数组与链验收范围。
- 检查命令（项目根目录）：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  data/mps_finish98_review_20261019/review_saved.py
```

- 审查脚本与结果分别保存于 `data/mps_finish98_review_20261019/review_saved.py` 和 `review.json`。session71491最终exit0，第一次实际执行通过；未修改业务源码、门槛或训练产物。
- 脚本只读NPZ/JSON/CSV并使用NumPy检查保存数组。metadata reader传递导入SciPy，但没有执行稀疏/模型运算；`torch`未导入。没有重新训练、重做CPU/GPU配对模型、软件测试、完整测试或build。
- 实施报告的墙钟3046.550231秒（50分46.55秒）为完整调度时间；新98窗compute总和2643.187488秒、均值26.971301秒，前向均值10.896139秒、反向含重算16.051153秒。分段加载、恢复、封存及父检查均另外计入墙钟，不将历史CPU不同窗口当严格配对加速结论。

| 保存采样资源 | 独立核对值 |
|---|---:|
| RSS样本数 | 26752 |
| MPS样本数 | 27106 |
| RSS峰值 | 1887862784 bytes，约1.76GiB |
| MPS current峰值 | 1532662016 bytes，约1.43GiB |
| MPS driver峰值 | 1595228160 bytes，约1.49GiB |
| 最低系统available | 2350710784 bytes，约2.19GiB |
| 有效RSS限额范围 | 2185478144–5865603072 bytes |

以上为0.1秒级采样，不是精确瞬时峰值；RSS和MPS统一内存口径不能相加。

## Issues / Regression Risks

没有阻止本批预算完成验收的问题，但实际训练仍存在明确研究限制：

- 新增forward末18窗，网络vx每帧仍约+0.3，而教师vx全部为负（−0.116004至−0.051239）；不能把存档/参数更新通过解释为跟随控制成功。
- theta raw梯度范数约1.24497e28–4.44219e32，裁剪系数约2.25114e−33–8.03233e−29；最多一窗86项非零梯度在float32裁剪后变零。原自身算术有限且通过，不代表梯度动态良好。
- 各场景统计来自不断更新参数的训练轨迹，非固定最终策略评估；首末损失不可直接解释为收敛。没有未见场景、冻结脑完整对照或内部/拓扑贡献结论。
- 全轨迹含CPU60窗与GPU100窗；旧GPU核级严格数值等价FAIL保留，Decision014接受已披露设备差异，不等于两后端轨迹等价。
- 最终场景起始神经值没有重新仿真；验收依据是保存状态摘要、源实现和参数/Adam连续证据。此限制不影响本批按已验收后端执行既定预算的结论。

## Required Rework

无本批返工项。协调者可完成本子批Done状态、验收链接和中文提交；不额外开始epoch、评估或实飞，不标记Spec008整体目标完成。
