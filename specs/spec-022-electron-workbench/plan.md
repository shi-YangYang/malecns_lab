# Plan — Electron 工作台

状态：P0 和 P1/P2 均已完成并独立 PASS。用户于2026-10-23认可原型并授权正式实施；正式接入按[冻结协议](implementation-contract.md)完成，[实测](actual-integration.md)与[验收](p1-p2-acceptance.md)保留。以下原型步骤为已完成历史。

## 修改顺序与分工

1. 协调者创建 Spec/Plan，冻结此次仅交付界面原型的范围，并更新索引。
2. 一名实施者负责 `apps/desktop/**`、相关 `.gitignore` 和实施报告。使用真实 atlas 资产及 Spec021 截图，构建有实际交互的 Electron 桌面原型；不改 Python、已有 atlas 数值或运行脚本。
3. 协调者打开实际桌面窗口，检查左右构图、回放、群组选择、分栏和窗口适配，保存原型截图。只在具体 UI 失败时限定修复。
4. 新独立验收者阅读 diff 和已有构建/UI 证据，核查数据标识、边界和资产保护；不重复通过的检查。协调者更新阶段状态与当前交接，向用户交付可查看原型。

## 文件与接口

- 新增 `apps/desktop/` 的 Electron 入口、局部 package-lock、TypeScript 界面、样式与素材准备脚本；素材目标目录忽略。
- 新增 `specs/spec-022-electron-workbench/{spec,plan,implementation,acceptance}.md`。
- 协调者维护 `specs/README.md`、`.ai/prompts/task-handoff.md`、`constitution/tech-stack.md` 的当前入口；不改写旧验收记录。
- P0 只读离线素材，无控制脑、ROS、Docker 或完整脑 IPC。原图谱数据结构沿用 `malecns_lab.atlas.v1`。

## 验证与停止

- Risk Level：S2。新增独立桌面主进程、渲染界面与本地素材边界；不变更现有数值/控制基础设施。仅运行新桌面子系统的必要构建/类型检查，随后一次 UI 冒烟；不跑 Python、atlas 或全仓测试。
- UI 核查播放/暂停、选择群组、拖动时间、脑图复位、分栏与最小窗口布局；同时查看明显加载失败。验收复核已有证据，不机械重复。
- 原型验收通过即停止，不自动推进真实遥测/仿真接入，不启动训练或 VM。
- 当前本轮未收到新的 commit/push 请求，不进行提交或推送。

## 正式接入结果

P1 真实spike旁路、22群组与32神经元批量记录和回放完成，完整图开关遥测数值精确一致且耗时/RSS符合预算。P2官方noVNC、有限IPC与生命周期完成；原生桌面两轮实际位移、逐轮停稳与资源关闭已核对。相关显示完整性与SD重试问题经限定返工和新的独立复核通过。资源已关闭，不追加训练、仿真或检查；本轮无commit/push。
