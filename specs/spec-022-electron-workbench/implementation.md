# P0 实施报告

日期：2026-10-23。Risk Level：S2（新增独立桌面主进程、渲染界面及素材读取边界）。本报告不代替协调者的实际 UI 验收与独立复核。

## 已实现

- 新增 `apps/desktop`，Electron 45.0.0 + TypeScript 5.9.3 + Three.js 0.180.0，Vite 7.1.12；局部 npm 依赖及锁文件，当前 Node v24.20.0。版本来自 npm 官方 registry；Electron 声明 Node >=22.12.0。
- 原生桌面标题栏与交通灯，默认 1400×880、最小 1100×720。左侧为已有 X2 截图的 CSS 视口裁剪，右侧是实际胞体点云、群组平均发放率曲线，底部统一回放控件。明确“原型预览”“历史截图”“诊断回放”“独立素材，尚未同步”。
- 支持拖动/键盘调整分栏、旋转/缩放/复位脑图、选择记录与群组、播放/暂停/回到开头/拖动进度/调速。全部播放行为仅作用于已有记录，不产生机器人命令。
- 直接复用 atlas 的 `data.ts` 和类型校验，未修改 atlas。脑图继承原有真实坐标及群组映射；当前 166700 个节点中，139662 个有胞体坐标、27038 个没有，不补造坐标。显示群组均值，不生成逐神经元脉冲。
- 脑图改为事件驱动的单次渲染，不自动旋转、不启用阻尼常驻循环，像素倍率上限 1.5。回放计时最多 25 次/秒，只显示记录中已有帧；默认暂停，窗口隐藏自动暂停，返回不自动播放。
- 没有 shell/进程/实验 IPC，没有控制脑、完整脑、仿真、ROS、Docker、训练或设备启动入口。

## 素材与进程边界

准备脚本只复制 atlas catalog、其 atlas/runs/stimulus_preview 引用及固定 Spec021 截图，共 15 份文件，目标 `apps/desktop/public/prototype/`。相对路径必须留在 atlas 数据目录，解析 realpath 防止链接逃逸；原 JSON 不改写，原 provenance、来源身份和许可保留。源数据缺失即失败，不生成回退素材。

Electron 使用受限 `workbench://app` 协议，只响应 index、编译 JS/CSS 和素材清单中的明确路径；实路径必须位于 dist。无任意仓库文件服务、无 remote page、无 preload。`nodeIntegration: false`、`contextIsolation: true`、`sandbox: true`，CSP 限制同源；权限、弹窗、导航、重定向、下载和 webview 均拒绝；不发出外部网络请求。

根 `.gitignore` 仅新增 desktop 依赖和复制素材目录忽略项；dist 沿用已有通用忽略规则。原始科研资产、截图、依赖和构建产物未纳入 Git。

## 实际检查

1. `npm install --no-audit --no-fund` 完成，新增 36 个局部依赖包。npm 提示 esbuild 安装脚本尚未在 allowScripts 中登记；后续实际构建成功，不修改全局 npm 配置。
2. 一次 `npm run build` 通过：准备 15 份素材，`tsc --noEmit` 通过，Vite 10 个模块构建成功；输出 HTML 0.73 kB、CSS 9.94 kB、JS 513.19 kB（gzip 131.38 kB）。没有运行 Python、旧 atlas 或全仓测试。
3. `git diff --check` 通过；`git check-ignore` 确认复制 atlas、dist 和 node_modules 均被忽略。新源码清单中没有科研资产。
4. 实际 Electron UI 由协调者验证并提供独立证据，本实施者未启动 UI，未把编译通过视为 UI 通过。

## 运行

工作目录：`/Users/yang/project/malecns_lab/apps/desktop`。

```sh
npm ci
npm run build
npm start
```

当前素材已准备、构建已生成，Electron 45.0.0 macOS arm64 二进制也已安装，直接 `npm start` 即可。官方安装器首次直连下载出现 `TypeError: fetch failed`；仅对安装命令沿用系统代理 `127.0.0.1:7897` 后能够下载，单连接约 90 KB/s。后续将同一个官方 GitHub Release ZIP 分段下载合并，完整 120706020 bytes，经 Node crypto 校验后使用 Electron 的官方解压依赖安装，未修改系统代理/全局 npm 或启动窗口。

完整 ZIP SHA256 与 Electron npm 包自带 `checksums.json` 一致：`896c3a50b148e89a158dca08a6996a022ddf39b23aac871e107f5b4087f8caa9`；安装目录的版本文件为 `45.0.0`。本地临时分段文件清理，二进制保留在忽略的 node_modules 内。

界面浏览器预览为 `npm run preview`，地址 `http://127.0.0.1:8775`；只服务 dist，无整个仓库暴露。该预览不替代 Electron 安全与桌面行为验收。

## 限制与遗留

P0 的 X2 截图与 CPU LIF 诊断回放是独立历史素材，截图不随回放更新。右侧没有连接 Spec021 的训练后 MPS 控制脑，不是实时控制或逐神经元脉冲，不能据此宣称已完成同步桌面实验。P1/P2 仍未实施。本轮没有提交、推送或启动任何重资源。
