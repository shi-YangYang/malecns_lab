# M2 第七轮挥手 ACK 超时诊断

2026-10-23，只读诊断；S0 报告，无业务代码修改、测试、仿真或完整脑启动。

**已确认当前 MC 与客户端的 `SetMcPresetMotion_Request` 字段布局不一致。先解决官方消息版本匹配，不能靠延长 ACK 超时或重发动作解决。** 本轮仍为挥手 UNKNOWN；未获得任务 ID、PLAYING 或实际挥手证据，取消资格仍关闭。

## 本轮实际证据

Run：`data/x2_sim/agibot/data/var/skills-20261023-170921-07ed4474`。以下行号均指其原始文件，未修改数据。

| 检查 | 事实与结论边界 |
|---|---|
| 服务注册与名字 | `mc.log:768–771` 注册 `/aimdk_5Fmsgs/srv/SetMcPresetMotion` 和 `GetMcPresetMotionState`；与运行快照 `transport_src/malecns_lab/experiments/x2_skills.py:264–265` 一致。无 namespace 拼错证据。 |
| 调用前条件 | `probe-02-wave_right/events.jsonl:57` 的输入源 ENABLE 返回 code=0；12 次状态查询均有响应，最后一次在第570行，task_id=0、state=1。挥手全段132条 MC 状态均为 STAND_DEFAULT、player IDLE、motion 空串；并非仍在 LOCOMOTION。 |
| 请求是否发出 | 第596行记录一条 motion=1002、area=2、interrupt=false、ani_path空串、play_timestamp=0、header.stamp=0 的请求。源码128行的 `service_is_ready()` 已通过，138行 `call_async` 返回 future；否则会进入 `preset_transport_error`，本轮没有此错误。因此客户端已调用发送 API；日志不是服务端收到请求的证明。 |
| 响应回调 | 发出到2秒期限间有100次 tick（最大相邻间隔0.021372秒）、20条MC、462条odom、695条关节反馈；持续执行 `spin_once` 和 future.done 检查，无 callback_error/exception。没有整个 ROS 循环阻塞两秒的证据；也不能仅凭其他回调工作排除该服务的 DDS 问题。 |
| 超时与动作 | 请求单调时刻166.084024，第1873行168.086659记录 UNKNOWN/preset_ack_timeout；没有 preset_ack。MC 日志只有服务注册，没有对应 preset 请求/拒绝/播放日志；该日志缺失本身不能证明网络包未送达。关节最大偏移0.000974647rad，未达到挥手阈值。 |
| 超时后归属 | 本轮没有 task_id，执行器按契约不查询未知归属任务，也不重发动作。future.cancel 只取消本地等待，不证明 MC 动作取消；等待到原12秒期限后仍保留 UNKNOWN。 |
| 资源 | host_summary 为 resources_stopped=true、errors=[]，VM约179.540秒；默认Docker context前后都是desktop-linux。诊断没有重启资源。 |

## 已确认的消息布局不一致

对本地 ELF 文件做静态节表、重定位和 ROS introspection 元数据读取；没有加载或执行 Linux 动态库。两个库同名服务的请求分别为：

| 来源 | 字段数 / C++结构大小 | 字段顺序 |
|---|---|---|
| MC随包 `mc/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so` | 6 / 64字节 | header, area, motion, interrupt, ani_path, play_timestamp |
| 当前客户端 `aimdk/install/aimdk_msgs/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so` | 7 / 104字节 | header, **input_source**, area, motion, interrupt, ani_path, play_timestamp |

这里结构大小只用于静态识别，不冒称 CDR 网络字节长度。关键差异是字段数及序列：客户端在 header 后插入服务器类型中不存在的 input_source，默认值或留空也不会移除序列化字段。

可复核身份（路径均以 `data/x2_sim/packages/` 开头）：

- MC库 SHA256：`935d4226013617a7aee2cdb525e31b6c46e44d608bf88f99eb61467b746cb931`。导出 `rosidl_typesupport_introspection_cpp__get_message_type_support_handle__aimdk_msgs__srv__SetMcPresetMotion_Request` 地址 `0x60460`，handle `0xacf00`，MessageMembers `0xa42a0`，member array `0xacfc0`。
- 客户端库 SHA256：`e68a8633e93f28689997160bb5d5804c195373dbc5f90a755d6d301ca609000b`。同导出地址 `0x42780`，handle `0x6b550`，MessageMembers `0x63c60`，member array `0x6b620`。
- 两库 MessageMember 的112字节步长及名字指针由静态元数据读取；客户端结果与生成源码 `aimdk/install/aimdk_msgs/include/aimdk_msgs/aimdk_msgs/srv/detail/set_mc_preset_motion__type_support.cpp:40–144` 的7成员一致。
- `mc/bin/em_run.sh:14–21` 把自身 `../lib` 放在 LD_LIBRARY_PATH 最前；本轮 host_source 使用该官方 MC launcher。客户端执行前 source `aimdk/install/local_setup.bash`（run/host_source.py:238）。运行时 `/proc/<pid>/maps` 没有归档，进程已退出，本次不补启资源假装取得动态加载证据。

这是已确认的**本地版本兼容缺口**，也是本次 ACK 缺失最有依据的解释。尚无线上包捕获或服务端反序列化诊断，不能断言具体丢弃发生在 DDS 类型匹配、反序列化还是服务分发的哪一层。现有名字发现成功与其他服务成功不能证明该请求类型兼容。

## 零时间戳不是已证实缺陷

本地官方文档 `aimdk/docs/cn/_build/html/zh-cn/dev/Interface/control_mod/preset_motion.html:654–660` 明确 header.stamp=0立即播放、play_timestamp预留；`changelog.html:732–740` 也如此。当前 `.srv:19–20` 注释却把立即播放语义写给 play_timestamp；官方 Python示例99–102行和C++示例则把header.stamp设为当前时间。文档与源码注释存在差异，**不能据此认定本轮零请求头错误，更不能将改成 now() 当成 ACK 修复**。

官方文档947行及示例提到服务通信/快速重试，但本技能动作不具备已验证幂等与取消契约，不能照搬八次重发来掩盖消息版本不匹配。

## 最小后续处理

1. **先取得与当前 MC 六字段请求相匹配的官方消息来源。** `changelog.html:695` 明确SDK v1.1.0新增 input_source；但MC包同时提供并已实际回应新增的状态查询服务，不能据版本名直接断言“整套降到SDK v1.0即可”。本地SDK是发布包，没有独立 `.git`；对其执行 git 会回到本项目根目录，不能冒认SDK历史。MC随包只有库，未发现该服务的 `.srv/.idl` 源文件。因此本次未找到可准确指定的官方tag/commit或完整匹配消息源。
2. 向官方确认 `mc-x86-v1.0.0-20260522` 对应的 AimDK 消息提交/包，尤其六字段 `SetMcPresetMotion` 与现有 `GetMcPresetMotionState` 的同时兼容来源。取得后在**新的隔离构建目录/运行环境**使用匹配的官方消息包，并登记来源和hash；保留当前SDK/MC/Sim及全部失败run。不要手改SDK主目录、替换MC动态库或只给对象删input_source后宣称兼容。
3. 实施时先增加/执行针对这一服务的消息契约比对，使已知6/7字段不匹配在资源启动前明确失败；没有匹配消息来源时，不再运行同样的挥手请求。匹配后才安排独立、有限的一次自然完成探针，并保留实际ACK→task_id→PLAYING→关节变化→完成链。取消能力仍需单独官方确认及验证。

本次只新增此报告，不修改SDK、MC、Sim、传输源码或超时；不重跑左右转向，不打开训练/锁定测试数据。
