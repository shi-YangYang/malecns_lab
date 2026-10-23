# M2 官方预设消息兼容接入

2026-10-23。状态：Ready，工程实施与独立验收后才允许构建及一次自然挥手探针。范围 S3：隔离厂商消息版本、跨进程请求与执行归属；不改官方SDK/MC/Sim源码、不整套降级、不开放M4挥手。

## 依据与边界

第七轮同名SetMcPresetMotion服务可发现，但请求没有ACK。静态诊断确认MC请求为6字段，当前SDK为7字段，多出的input_source插在header之后。官方索引的v1.0.0原包提供匹配的请求、响应及递归依赖，却没有现行GetMcPresetMotionState/McCommonState；这两个新增类型仍须使用v1.1.0。具体源/类型证据以m2-official-message-compatibility.md为准。

原包地址来自 https://x2-aimdk.agibot.com/sdk-downloads.json ，旧包156945205 bytes，实测SHA256为5bbcf724d54fb28f153db0d272f9acb7906bb1d2cac7dd7ccdc699a5c7eeab35。散列为本地来源记录，不称厂商签名。保留原目录和七轮失败/成功证据。

## 最小实施

- v1.0.0未改源aimdk_msgs单独构建到忽略的新工作目录；现有v1.1.0 install与MC lib保持原样。构建脚本绑定官方源文件清单/hash、固定输出目录、编译器/ROS环境及最终安装产物hash，不取第三方消息定义或手删input_source。
- 新独立Python进程仅加载旧版消息与ROS Humble，负责本次单一SetMcPresetMotion请求。父进程继续加载新版状态/查询/速度消息；禁止在同一进程混载两个同名aimdk_msgs。子进程环境从明确白名单构造，不继承新版PYTHONPATH/AMENT/CMAKE/LD_LIBRARY路径；验证实际导入位置和类型支持库身份。
- 父子采用有界JSON行IPC，带本次run/probe nonce、序号和明确ready/request/response/error语义；协议拒绝额外键、重复、错nonce、超长/无效输入。子进程不接受任意shell、路径、服务名或动作，固定motion1002、area2、interrupt=false、立即执行时间戳和空ani_path；不重发。保留实际序列化请求、原始响应/task_id与两端本地时钟，不能跨时钟相减。
- 父进程的状态/关节/odom与拒绝保护循环不被子进程等待阻塞。保留原服务等待和2秒ACK限额，不用延时掩盖兼容问题；实际请求只允许一次。子进程异常、协议不符或ACK超时仍记UNKNOWN，终止进程不等于机器人动作取消。
- 原SetMcPresetMotion新版调用路径不再用于这个匹配MC的wave probe；没有已验收兼容产物/manifest则在任何动作前明确失败，不能悄悄回退。非wave技能无需启动兼容子进程，既有行为保持。
- 新host/runner档案绑定旧包源、新build产物、实际导入类型/来源、MC与新版库hash和helper来源。实际运行前核对请求/响应及递归类型匹配；保持GetMcPresetMotionState和MC实际播放/关节证据要求。ACK不是完成，取消资格始终false。
- helper正常/异常/超时均有界回收，父进程清理GPU仿真和私有VM的路径不依赖IPC成功；不动Noria、不改默认Docker context、不启动旧CPU仿真。

## 检查、构建与一次实测

实施仅涉及新的preset兼容边界、单技能host/runner与部署脚本及相关测试，不碰训练模块、核心脑、M4或桌面。一次最小CPU检查覆盖环境隔离、严格IPC/单次请求、真实runner回调贯通、ACK/异常/清理和默认非wave不启子进程。用隔离假ROS类型做协议检查不能称真实DDS兼容。新的独立验收者审查后，协调者才运行构建。

构建使用已有本项目Linux环境，单次最多900秒、最多2个编译任务；与完整脑训练串行，重新核对系统至少1GiB预留，不安装/升级无关依赖，不启动MC/Sim。结束关闭本次拥有的资源。若依赖缺失或构建失败，保留结果并针对具体问题决定，不能自动扩大环境。

构建完成先只读检查实际产物/导入来源及消息布局。新自然挥手实验单独登记，仅WAVE_RIGHT，沿已核验的GPU生命周期、180秒UI/360秒总预算、原12秒动作观察和30秒清理预留，官方Reset/Run有新鲜画面后才continue。一次发送，保留真实ACK、task_id、query/MC/关节变化及停稳，禁止自动重试或把未确认取消写成功。前进/左右转向已通过，不为本改动机械重跑。

此方案只解决已识别的消息版本缺口；自然挥手和受控取消分别验收。即使自然挥手通过，M4挥手能力仍关闭，P2完整门槛及P5不能由此自动通过。
