# 上游代码、数据与引用

## 上游代码

MaleCNS Lab 基于 [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones) 的代码发展。连接组加载、LIF 仿真、视觉编码、旧运动读出与部分设备工具继承并改编自该项目；当前整理聚焦共享动作解码、Tello 实验和离线数据训练。

上游使用 MIT 许可。根目录 [LICENSE](LICENSE) 保留原文：

> Copyright (c) 2026 SpikeCalls and FlyDrones contributors

再分发包含上游代码的副本或实质部分时，需保留上述版权声明和完整许可文本。

## MaleCNS 数据

MaleCNS v1.0 由 HHMI Janelia FlyEM、University of Cambridge、MRC Laboratory of Molecular Biology 和 Google Research 等团队提供，数据采用 CC-BY 4.0。

- [项目与作者信息](https://male-cns.janelia.org/)
- [下载与版本来源](https://male-cns.janelia.org/download/)
- [CC-BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)

源码包不包含原始连接组数据。本地下载、过滤后的连接图和由其产生的实验结果应记录数据版本、来源及处理条件。发布相关结果、视频或派生数据时，按数据许可保留适当署名、来源与改动说明，并引用数据对应论文。

## 模型参考

LIF 参数参考 Shiu P. K. 等人的 *A Drosophila computational brain model reveals sensorimotor processing*（Nature，2024）及其公开实现：

- [论文](https://www.nature.com/articles/s41586-024-07763-9)
- [参考实现](https://github.com/philshiu/Drosophila_brain_model)

神经群组、视觉编码与旧运动读出的参考见 [研究文档](docs/research/boundaries.md)。论文中的生物学发现与本项目的输入映射、读出、阈值和设备行为应分别说明。

## 单独安装的依赖

依赖通过 Python 包管理器安装，各自保留原许可；本项目的 MIT 许可不替代它们的许可。当前依赖范围包括 numpy、scipy、PyYAML，以及可选的 OpenCV、pyarrow、pandas、DJITelloPy、pytest 和 Ruff。具体安装声明以 [pyproject.toml](pyproject.toml) 为准。

原浏览器演示及其随附 three.js 资源已从当前项目移除。新的 `apps/atlas/` 通过 npm 安装 Three.js（MIT）、Vite（MIT）和 TypeScript（Apache-2.0）等依赖，以 `package.json` 与锁文件为准；依赖保留各自许可。新果蝇示意模型由程序化几何构建，不使用原演示资产。

脑图所用胞体坐标来自 MaleCNS 官方注释；派生展示数据保留 CC-BY 4.0 署名、来源和居中缩放说明。神经色彩来自项目 LIF 模型的记录，不是原数据集中测得的活体神经活动。
