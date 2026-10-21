# Web 实施报告 — Spec 010

## Role / Scope

Implementation。工作区 `/Users/yang/project/malecns_lab`，按 Ready Spec 实施独立本地 WebGL 子系统。仅新增 `apps/atlas/` 前端源码、依赖锁及本报告；未修改已有 Python 模型、训练、诊断记录或根目录治理文件。`apps/atlas/public/data/` 由数据实施者导出，本实施者仅只读核对契约。

## Implementation

- `apps/atlas/package.json` / `package-lock.json`：固定 Three.js 0.180.0、@types/three 0.180.0、TypeScript 5.9.3、Vite 7.1.12。开发和预览默认 `127.0.0.1:8765`、`strictPort`；构建为 `tsc --noEmit && vite build`。浏览器仅使用本地打包依赖和系统字体，无 CDN。
- `src/types.ts` / `src/data.ts`：定义并校验 `malecns_lab.atlas.v1` 静态契约；catalog/atlas/run 同源加载。检查坐标数量、位置覆盖、群组索引、帧数量、严格递增的原始时间戳、有效性掩码、Hz 数组、模型来源与刺激身份、帧率、预热时间。PNG 相对各 run URL 解析并检查精确 sprite 尺寸。损坏、缺文件、零帧、空群组、空位置不生成合成替代。
- `src/brain.ts`：真实胞体坐标的 WebGL 点云，按原记录逐帧更新群组平均 Hz 颜色；全部刺激共同线性色标。未记录节点静态背景、无虚构突触线。群组高亮、拖动旋转、滚轮缩放、视角复位与键盘方向键/加减/Home。若索引群组重叠，显示最大组均值，此规则在说明中明示。
- `src/chart.ts`：选中群组的 Hz 阶梯曲线，原帧时间横轴，共同纵轴上限；缺失/无效值断线，显示当前帧游标和预热背景。
- `src/fly.ts`：带复眼、分节腹部、透明翅与六腿的静态三维果蝇示意，可手动旋转。没有任何神经驱动、肢体周期、飞行或设备动作动画。
- `src/main.ts`：六刺激按钮、群组选择、逐帧播放/暂停/前后步进、拖动时间轴、从头回放、0.25×/0.5×/1×速度、同步 PNG 刺激重建、来源说明对话框。默认扩张刺激的首条预热后有效记录；全部刺激切换后同步刷新，播放结束后暂停。可见固定诊断模型、组均值、缺位置、刺激重建、无动作控制记录、设备未接入和 MaleCNS v1.0 / CC BY 4.0 署名。
- `src/style.css` / `src/tokens.css`：暖黑与深蓝画布、青绿活动色及琥珀观察阶段，桌面三列仪表盘与窄屏重排；原生按钮/滑块/select、ARIA 文本和焦点样式。没有外部字体请求。

## Actual Data Integration

只读核对数据实施者产物：166,700 参考节点、25,582,938 有向连接；139,662 有胞体位置、27,038 缺位置；22 群组，六刺激每段 50 帧。`time_s` 原记录为 0.00–2.45 s、20 Hz；每段总仿真覆盖 2.5 s。界面原样显示帧时间，不把最后的帧起点改成 2.5 s。

## Validation

Risk Level: S2。新前端子系统及 JSON/WebGL 交互属于必要构建范围。

- 已阅读新增源码及文件范围，确认没有修改模型、原始记录或数据导出器。
- `npm install --no-audit --no-fund`：完成，23 个本地依赖包，生成锁文件。
- 一次 `npm run build`：PASS；TypeScript 类型检查与 Vite 生产打包通过（12 modules，Vite 打包 460 ms）。主 JS 为 567.31 kB / gzip 147.90 kB；Vite 的默认 500 kB chunk 建议为非失败提示。
- 未运行全仓库测试、训练、模型诊断或硬件。未安装浏览器测试框架，未由本实施者启动开发服务器；协调者负责一次浏览器交互与截图集成检查。构建通过不等同于已完成浏览器验收。

## Handoff / Remaining Validation

已通知协调者可使用其启动的 `http://127.0.0.1:8765/` 完成浏览器 QA。关键可检查元素：`data-testid="brain-canvas"`、`fly-canvas`、`time-value`、`current-rate`；主要控件具有中文 ARIA 名称。最终验收由独立验收者记录，本报告不预填最终 PASS。无提交或推送。
