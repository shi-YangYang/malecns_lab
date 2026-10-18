# 前向缓存软件返工

新模型backend为 `scipy_cpu_csr_forward_torch_csr_adjoint`，但P1和配对诊断证据校验仍硬编码 `torch_cpu_csr`，导致实际入口拒绝新模型。首次软件验收FAIL报告保留；没有执行P1或配对实验，没有覆盖旧资产。

返工由新的实施Agent仅同步两个校验器和相关证据fixture/回归检查，保留严格backend、图、参数、源码及登记身份绑定；新身份通过、旧/未知身份拒绝。不改动力学或原始实验。验证只覆盖受影响的证据路径，原46项数值检查已通过，不重复执行。完成后新独立Agent验收，PASS后协调者commit并执行C/D。
