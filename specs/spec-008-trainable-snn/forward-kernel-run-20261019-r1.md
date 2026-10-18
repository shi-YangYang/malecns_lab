# 完整图前向内核对照 r1

日期：2026-10-19北京时间。软件提交 `2a3979a` 后按[冻结协议](forward-kernel-benchmark.md)执行一次；监督正常结束，实际独立验收另见 acceptance-forward-kernel-run-r1.md。

```sh
PYTHONPATH=src .venv-train/bin/python examples/fullgraph_matvec_benchmark.py \
  --graph data/malecns_v2/full_reference.npz \
  --out data/forward_kernel_20261019_r1
```

登记UTC `2026-10-18T19:20:12.858663+00:00`；registration ID `7d79e3f5c79afa7ac8790b709121c7972f54f7242a1e4de65bf070642ef8ae53`，protocol ID `484bd9646808852847aab7baf1fca3bc9bd8e04264c0feb57a9bc6c33c11e7a5`。固定166,700节点、25,582,938边、124,177,617突触计数，float32、Torch 4线程，同一固定缩放权重；没有改生产模型或训练。

## 原始结果

| 输入 | Torch五轮中位数 | SciPy五轮中位数 | 全部输出最大绝对差 |
|---|---:|---:|---:|
| 全零 | 74.712417 ms | 15.274583 ms | 0 |
| 恰好1%节点为1 | 74.580583 ms | 15.323750 ms | 0 |
| 稠密uniform(-1,1) | 74.637625 ms | 15.445375 ms | 0 |

全部输出均有限、shape为(5,166700)、float32；此三输入下逐元素精确相等，强于预登记容差。SciPy计时已包含输入NumPy视图与输出Torch包装；矩阵视图仅构造一次，耗时0.001570s。data/indices/indptr都与Torch共享地址和内存，分别float32/int64/int64，顺序及前后摘要相同。原始图计数和磁盘文件摘要未变。

监督总墙钟5.489826s、退出0，采样峰值RSS 1,549,238,272 bytes（约1.443GiB）；启动available 7,050,788,864 bytes，扣除1GiB后预算5,977,047,040 bytes。保留6GiB最大进程上限及600s限制，没有Noria/Docker或设备操作。此为采样峰值，不声称测到全部瞬时峰值。

## 解释与下一步

三种输入均满足预登记候选条件，单次内核中位耗时比约4.8。这个结果支持进一步实施SciPy共享CSR前向、保留原Torch完整伴随反向；不证明整体训练快4.8倍、任意输入均逐位一致、完整阈值动力学等价或可实时控制。

下一步另立生产核心子规格，用独立代数/静默细胞梯度检查，再按原P1和配对背景协议进行完整图对照。三轴学习及AC3–AC6仍未完成。

## 证据位置

本地目录保留registration.json、source/源码快照、model_identity.json、inputs.npz、zero/sparse/dense_outputs.npz、timings.json、worker_result.json、result.json、resources.csv及worker.log。每个输入保存预热与全部五轮输出；worker结果绑定所有输入/输出和计时文件SHA。实际独立验收须重新核对输入、数组、时间、绑定及资源，不只读取production_candidate标志。不重跑本次模型。
