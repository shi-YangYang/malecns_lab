# Implementation Report — 固定窗口MPS递归梯度诊断

## Summary

仅实施 [GPU修复规格](gpu-recovery.md) 的 A 软件部分。新增受监督入口从原GPU run第159窗完整末档恢复，固定第160窗两帧输入，在同一个worker串行运行1000、500、250、100神经步递归信用范围；全部分支仍运行完整1000步前向，不执行optimizer更新。当前尚未执行完整连接组诊断，不能把小电路检查写成巨大梯度原因已定位。

按协调者明确要求，包含策略读出/损失的训练梯度诊断归入 `learning/`；不扩展原 `diagnostics/` 的离线神经观察职责。原42个绑定训练源、旧实验资产、CPU/frozen/R分支均未修改。

## Files Changed

- `src/cnsbridge/learning/gradient_probe.py`：轻量父入口、新输出目录、共享数值锁、6GiB申请/available−1GiB实际预算、600秒/.1秒监督及结果复核。
- `src/cnsbridge/learning/gradient_probe_evidence.py`：固定源run与159/160完整档、已封存全链、原数据选择、46份源SHA（旧42＋新4）、逐位比较及保存证据检查。
- `src/cnsbridge/learning/gradient_probe_worker.py`：父PID/共享锁/单次claim后才导入Torch；一份完整图串行四次完整状态恢复、分支封存、原始图计数前后摘要、MPS资源采样与失败保留。
- `src/cnsbridge/learning/gradient_probe_math.py`：复用原50步checkpoint闭包、逐步DN加法顺序和原MSE＋正则；状态边界detach、伴随观察和固定feature的稳定logit损失诊断。
- `tests/test_gradient_probe.py`：5项相关检查，全部显式小电路或小张量，禁止解释为完整图实验。
- 本实施报告。

## Important Decisions

### 不变的前向与分支数学

每支都从同一159末档恢复全部18数组：参数、Adam状态、神经v/g/refractory/pending、感觉phase/event counts和时钟。恢复复用原严格reader及 `restore` 的实际设备往返检查。CPUAdamBridge只作为原状态的恢复/不变性核验对象存在，没有调用 `step`；本探针不是新全GPU训练器。

四支均采用原两帧总动作MSE＋gain正则。1000步支无中途截断；500/250/100步支仅在相应整步边界detach完整神经状态，数值、感觉相位、时间、全部DN逐步累加均不变。DN运行和始终保留梯度，因此“100步范围”表示递归状态传播的最大分段长度，不是只用最后100步DN信号或丢掉早期帧损失。

实际运行时首先要求1000步分支的动作、全部DN特征、原始theta/读出梯度、所有末态数组及状态摘要与原第160窗逐位一致。短分支对同次1000步基线要求动作、特征、全部末态和读出梯度逐位一致；theta梯度允许因截断而不同。失败将保留值与比较报告并停止，不临时放宽数值门槛。

### 伴随观测

每50步登记v、g及18项延迟pending的输出张量hook；integer refractory明确不可微。hook只读取，不修改或返回替代梯度，不保留完整大梯度向量。范围切断边界观察的是**detach之前的左侧输出**；其梯度仍可含此前DN损失，后续递归信用被切断。未被autograd访问的hook标记 `observed=false`，不冒充测得零值。

稳定范数先在MPS取max并缩放，再对缩放值平方求和；最后以Python双精度标量相乘，避免对1e30级float32原值直接平方溢出。每项记录finite、L2、maxabs、nonzero。该观察增加同步开销，探针计时不能冒充未观测训练的性能基准。

### 固定feature的候选损失

使用原第160窗已保存feature和159末档读出参数，在MPS计算 `mean(softplus(2z)−(1+y)z)`，以autograd检查导数与 `(tanh(z)−y)/元素数` 一致。保存logit、原部署动作、标签、梯度和解析梯度。一个0.1步的**局部logit变量**下降只用于检验纠错方向，不更新任何模型参数，不声称完整网络训练效果；原部署动作仍须与基线逐位一致。

### 资源与来源

生产入口固定 `data/neural_training_mps_20261019_r3`，不允许选择其他窗口、图或horizon；输出必须是全新目录且不能位于原run内部。原完整图身份由既有契约核查，worker构造原MPSPolicy并核对raw counts摘要，所有166700节点/25582938边保持。父进程在退出后再次核对原来源与新46源、分支SHA、恢复/选择数组、资源采样和MPS预算。无新依赖、CPU fallback关闭、不操作Noria/Docker。

## Validation

- Risk Level：**S3**，涉及反向图边界、完整状态及诊断来源，但修改仅新增模块。
- 唯一执行的定向命令：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m pytest tests/test_gradient_probe.py -q -s
```

- session **76019**，已收取exit0，**5 passed in63.55s**。仅原CPU小电路构造产生一次既有PyTorch sparse CSR beta警告。
- 检查覆盖：父模块不导入Torch；无合法监督父的worker在重导入前拒绝；3e30/−4e30向量范数为有限约5e30；四范围1000步显式小电路前向/全部末态/读出梯度精确一致，1000步theta梯度与既有原MPS窗口精确一致；短范围theta梯度不同；反向重算不改变初态或末态；正负饱和logit的梯度有限、非零、方向正确，局部logit下降降低loss而模型参数不变；单ULP差异不会被逐位比较静默接受。
- 小图theta梯度示例（组序沿fixture，不代表真实MaleCNS群）：1000步末项约−0.01302013，500步约+0.003720554，250步约−0.000151484，100步约−0.000158049。只证明fixture确实改变反向路径，不推断完整图因果。
- 通过后没有重复旧MPS核测试、旧98窗验收、全项目suite、build或lint。与另一实施者协调共享锁；当前本实施数值进程已退出。

## Actual Command — 尚未执行

需由协调者在新的独立软件验收后登记一次实际完整图诊断：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.gradient_probe \
  --out data/mps_gradient_probe_20261019_r1
```

软件检查未创建该生产输出目录。worker结果、分支values/report、恢复与输入NPZ、固定feature logit-loss证据、RSS/MPS采样均写入实际新目录。超预算、源变化、逐位不一致或非有限值会返回失败，保留已产生现场。

## Known Limitations and Remaining Questions

未执行真实完整图分支，递归伴随放大仍待实测；没有实施B损失/优化器/感觉GPU迁移或C训练。A保留原CPU感觉编码及原Adam恢复只是为了忠实重放旧协议，不能称为“训练全部GPU”；全GPU新训练方案属于后续已分开的实现。尚待新的独立软件验收和实际结果验收。实施者不作最终验收、不提交。
