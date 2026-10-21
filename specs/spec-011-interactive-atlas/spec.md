# Spec 011 — 交互式完整脑实验

状态：Done，2026-10-21 独立验收 PASS。用户要求将回放图谱升级为网页设置刺激、实际运行完整 MaleCNS 并查看新响应。沿用当前固定 LIF 诊断模型，不训练、不连接设备；[三次正式实测](actual-run-20261021.md)已完成。

## 范围与约束

- 网页设置刺激种类、速度倍率、对比度与观察时长，提交独立计算、查看真实进度/耗时、取消并切换已完成结果；保留现有记录。
- 后端复用 `NeuralReplayV2` 和默认 `data/malecns_v2/full_reference.npz`，必须校验完整图身份，保留全部节点/连接；不改已封存的神经、感觉、生成器或训练源码。
- 新生成器位于 visualization；默认速度/对比度与旧 GeneratedSourceV2 逐像素一致。速度改变图像运动，对比度改变像素强度，不直接修改神经率。神经 seed=0、纹理 seed=7 固定并记录。每次从独立初始状态开始，保留 2 秒预热。
- 每有效输入帧 20 Hz，对应100个0.5ms SNN步；实际等待时间单独记录。CPU固定LIF诊断复用现有已审计数值路径，不涉及先前GPU训练。
- 单计算子进程、加载前起监控；保留1GiB可用内存，超时、内存不足、视觉无效或取消均如实结束，不发布为成功。不得停止其他应用或Noria服务。
- 新产物进入忽略目录 `data/atlas_experiments/`，保留参数、完整图/模型/源码身份、精确输入帧和原始输出，旧记录不改写。浏览器颜色仍是群组均值；果蝇仍是静态示意。
- 保持 AGENTS 网页内容规则。技术身份留在文件中，界面只呈现操作、状态与必要限制。

## 本地接口契约

Python仅监听127.0.0.1:8766；Vite 127.0.0.1:8765 将 `/api` 代理过去。`npm run dev` 同时管理两进程，结束时回收自己启动的任务。`npm run dev:viewer` 保留只读页面模式。不添加云服务。

- `GET /api/health`：`{status:"ok", active_job_id:null|string}`，可附加资源及配置。
- `GET /api/experiments`：`{experiments: Job[]}`，按创建时间逆序，启动时将未完成旧任务标记中断。
- `POST /api/experiments`：严格JSON `{stimulus, speed, contrast, duration_s}`；stimulus 为 static/left/right/up/down/expansion，speed 0.25–2，contrast 0.25–1，duration_s 0.1–3 且20Hz整帧。默认 expansion/1/1/0.5。202返回Job；并发409；参数/资源等失败返回适当4xx与 `{error:"中文原因"}`。
- `GET /api/experiments/<id>` 返回Job；`POST /api/experiments/<id>/cancel` 取消并返回Job。
- `GET /api/experiments/<id>/assets/<filename>` 只提供已完成实验白名单展示资产，不允许任意文件/目录读取。
- 变更请求必须有 `X-MaleCNS-Request: 1`，校验 Host 与浏览器 Origin（仅本地8765/8766）；不允许跨站触发计算，无通配CORS。不接受路径、代码、任意命令或模型参数。JSON限长，拒绝未知键、布尔数字和非有限数。

Job必需字段：`id`, `status`（queued/loading/running/exporting/completed/cancelled/failed）, `parameters`, `created_at`（ISO）, `frames_completed`, `total_frames`, `simulation_s`, `elapsed_s`, `error`（null或中文字符串）, `catalog_url`（仅completed提供）。可附加phase、RSS等证据。父进程持有任务终态，worker单写progress/result，避免取消与完成竞态。

## 展示兼容

结果提供现有 `malecns_lab.atlas.v1` catalog/atlas/run，当前模型source_id来自实际engine，不伪装旧目录身份。生成刺激预览与实际输入帧一致，`stimulus_preview.kind` 新增 `generated`，已有 `reconstructed` 继续支持。前端复用同一验证与渲染链路；切换结果不可泄漏旧播放循环或丢失状态。新结果可从实验列表恢复；请求失败保留现有图谱并可重试，不使用假响应。

## 验收

跨Python任务管理、前端接口与启动流程，按S2做子系统检查；本地变更入口的Origin/Host、目录限制和并发取消需定向覆盖，不运行全仓库测试。

1. 后端定向测试：参数、默认刺激相等、进度与终态、取消/失败/并发、资源拒绝、本地请求限制与路径限制。
2. 前端一次构建及浏览器交互检查，含新任务、进度、结果载入、历史/旧记录切换、失败/取消和窄屏布局。
3. 通过实际HTTP服务运行两次相同短完整图实验（0.5秒观察，2秒预热），核对100%完整图、5000神经步、完整帧与结果逐数组可复现；另一次不同方向短实验验证参数确实改变输入与输出。三次顺序执行，实际资源不足则保留失败证据，不缩图或关闭其他服务。
4. 实施后由独立验收者复用结果审查，不重复重算。没有新请求时不自动训练、commit或push。
