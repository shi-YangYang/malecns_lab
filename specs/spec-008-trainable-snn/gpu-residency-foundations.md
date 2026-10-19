# 全GPU感觉编码与优化器基础模块

状态：Ready，属于[gpu-recovery](gpu-recovery.md)的B1独立基础部分。A诊断仍决定训练数学与反向时间范围；本批只实现用户明确要求的GPU计算落点，不启动正式模型训练。

## 已定接口与独立边界

- `senses/gpu_drive.py`：纯MPS目标掩码与感觉驱动。`GPUDriveState`含MPS float32 phase、MPS int64 event_counts及Python整数step、drive/map身份；值不可原位修改，更新返回新state。构造接收已有SpatialInputMap、SpatialDriveConfig和TargetCamera；初始化相位由原PCG64种子生成后一次转换float32，身份登记新dtype/算法。`mask(frame)`上传uint8后在GPU阈值/池化；`step(mask,state)`返回完整N维GPU float32 increment与新state。全部R1–R6选择保持，不含CPU逐神经步计算。保存/恢复在边界显式下载/上传并校验shape/dtype/身份。
- `learning/gpu_optimizer.py`：以`brain.theta`、`readout.weight`、`readout.bias`命名的MPS参数作为唯一主参数。沿用theta/readout两组、lr0.001/0.0001、beta0.9/0.999、eps1e−8、weight_decay0；明确新GPU优化器身份，优先内置Adam fused=True。构造可零状态或明确提供完整moments/steps；是否继承旧Adam由B集成规格决定，基础模块不得擅自决定。
- `step()`读取现有设备梯度，在GPU校验有限性、稳定缩放范数并按组max_norm=1裁剪，再执行GPU Adam；不接受CPU梯度或保留CPU master。零/极小/极大梯度必须正确处理，不以nan_to_num隐藏异常。所有m/v/step也须在MPS。GPU步骤结束后才允许保存副本，CPU检查结果不得写回模型。
- 新模块不依赖环境真值、不改变旧源码与旧协议；输入/输出形状与命名保持，使后续单一集成者可复用。禁止复制整个训练器或引入额外依赖。

## 数值与验证

风险S3，仅本模块定向验证。GPU感觉允许float32相位事件时刻与原float64不同，必须给出固定画面/跨帧的误差与事件差异，不能声明逐位迁移。GPU mask对旧像素规则应相符，浮点占有率允许atol1e−7；阈值边界应单独覆盖。纯state重算和保存恢复要求精确重现新GPU自身结果。

内置fused Adam先实际验证MPS支持；若失败，报告具体原因再决定替代，不能静默CPU回退。独立CPU float64仅作保存结果参考，moment/参数容差事先为rtol2e−5、atol1e−8；小梯度同时检查nonzero与符号，避免绝对容差掩盖全零。包含至少连续三次更新和一次恢复后更新、非有限拒绝、零及约1e30梯度的稳定裁剪；测试结束按本机能力记录实际误差，不反复扩展无关检查。

基础模块可与A诊断软件并行编写，因为文件不重叠、接口在此明确、两者不修改既有源；所有数值检查仍通过同一workspace_lock串行。A诊断先集成/验收，基础模块随后由新独立Agent验收，最终B集成由单一实施者完成。不能并行运行完整模型或GPU测试。

## 验收

- [ ] F1：编码与相位/计数/增量全在MPS，跨帧与重算一致，新感觉身份及旧float64差异真实记录。
- [ ] F2：参数/梯度/裁剪/Adam/m/v/step全MPS、fallback=0，独立算术和恢复定向验证通过。
- [ ] F3：旧绑定源码不变，实施报告与新独立验收完成；本批不宣称训练或控制效果。
