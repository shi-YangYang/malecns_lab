# M2 官方消息兼容来源调查

2026-10-23。仅下载到新的忽略目录并静态读取官方包、源码与MC类型元数据；未编译、部署、启动VM/仿真或修改现有SDK/MC。S0报告，检查范围限于挥手兼容问题。

**找到官方未改源的六字段 `SetMcPresetMotion`：AimDK `v1.0.0-ga424add`。它的请求/响应及递归消息布局与本地MC匹配；但该包没有 `GetMcPresetMotionState`、`McCommonState`，不能整套替换现有SDK。最小方案是旧版独立进程只发送一次Set，新版主进程保留状态查询和控制。** 这是静态兼容方案，尚无新ACK或动作实测。

## 官方包与保留路径

官方[下载索引](https://x2-aimdk.agibot.com/sdk-downloads.json)提供[SDK v1.0.0包](https://x2-aimdk.agibot.com/downloads/aimdk-sdk-v1_0_0-aimdk-aarch64-a424add7-artifacts.zip)，索引上游为 `https://file.agibot.com/ci_cd/artifacts/lingxi/sdk/aimdk/v1.0.1-r9/aimdk-aarch64-a424add7-artifacts.zip`。包内 `version` 实际内容为 `aimdk v1.0.0-ga424add`；不将下载标签、上游构建路径和源码tag混为同一个版本号。

- 新目录：`data/x2_sim/compatibility-v1_0_0-20261023/`；已确认被Git忽略，未覆盖任何旧文件。
- ZIP：`aimdk-sdk-v1_0_0-aimdk-aarch64-a424add7-artifacts.zip`，**156945205 bytes**，SHA256 `5bbcf724d54fb28f153db0d272f9acb7906bb1d2cac7dd7ccdc699a5c7eeab35`。
- `source.json`保存URL、大小、哈希、时间及范围；本次索引原文 `sdk-downloads.json` SHA256 `54091d44ca111177d414ab46a8f5080cd97999c443a0dfa7e7843275300f31fb`。
- ZIP内完整消息包根：`aimdk-aarch64-a424add7-artifacts/src/aimdk_msgs/`。本次仅把84个官方消息/服务源文件原样提取到新目录 `official-source/src/aimdk_msgs/interface/`；**该检查目录没有提取完整CMake工程，不是已构建环境**。
- 已有MC包仍为 `v1.0.0-20260522`，原ZIP SHA256 `b3bc2417aad656b6a7f19b2274f07d5cb8c7f60b1317482e0122244e57d812d0`；现有SDK1.1原ZIP SHA256 `dc80fe5d2907881ed77beb230680867efac6bd28b458f83d2ff4e9479abe3615`。来源见 `data/x2_sim/manifest.json`，均未改动。

## 静态比对结果

用MC实际随包 `libaimdk_msgs__rosidl_typesupport_introspection_cpp.so` 的ELF元数据，逐项比对名称/顺序、原始类型ID、数组标记/长度/上界，并沿官方源文件的命名消息依赖递归检查。请求/响应根各自计入下表数量；同一依赖可能出现在多行。

| 当前使用接口 | v1.0.0源对MC | v1.1.0源对MC | 保留方式 |
|---|---|---|---|
| SetMcPresetMotion请求 | 4个类型全部布局匹配；根为6字段 | 5个类型中仅根不匹配：多input_source | 独立旧版Set进程 |
| SetMcPresetMotion响应 | 4个类型全部匹配 | 4个类型全部匹配 | 旧版进程返回原始响应字段 |
| GetMcPresetMotionState请求/响应 | 缺失整个服务 | 请求3/响应4个类型全部匹配 | 现有新版主进程 |
| McCommonState | 缺失整个消息 | 18个类型全部匹配 | 现有新版主进程 |
| SetMcInputSource请求/响应 | 请求5/响应4个类型全部匹配 | 请求5/响应4个类型全部匹配 | 现有新版主进程 |
| McLocomotionVelocity | 2个类型全部匹配 | 2个类型全部匹配 | 现有新版主进程 |
| JointStateArray | 4个类型全部匹配 | 4个类型全部匹配 | 现有新版主进程 |
| McInputAction | 布局匹配 | 布局匹配 | 现有新版主进程 |

本范围v1.1共36个不同类型，35个布局匹配，唯一差异为Set请求。`McControlArea`、`McPresetMotion`、`CommonState`、`McInputAction` 的旧新版源文件及常量完全相同。旧Set请求递归为 `RequestHeader`、`McControlArea`、`McPresetMotion`；响应递归为 `CommonTaskResponse`、`ResponseHeader`、`CommonState`。两边的外部依赖为ROS标准 `builtin_interfaces/Time`；当前 `nav_msgs/Odometry` 属于既有ROS Humble环境，不属于本次SDK变动。

此检查不执行共享库，不取运行时嵌套类型指针，不验证DDS包的实际序列化/发现/分发。它证明当前包的静态布局匹配范围，不冒称RPC或动作成功。MC库SHA256 `935d4226013617a7aee2cdb525e31b6c46e44d608bf88f99eb61467b746cb931`；此前ACK失败证据见[m2-wave-ack-diagnosis.md](m2-wave-ack-diagnosis.md)。

关键源文件SHA256（旧版位于新目录 `official-source/src/aimdk_msgs/interface/`）：

| 官方源 | SHA256 |
|---|---|
| 旧版robot/mc/motion/srv/SetMcPresetMotion.srv | `29df5508145be62d1c094d6335bf98d547374a52d3e5f7786c857f75038ae585` |
| robot/common/msg/RequestHeader.msg | `48a9467857fa698ba1de18236f9f80741406e5d4bfedfffd36be6fa50fb10664` |
| robot/mc/motion/msg/McControlArea.msg | `7ecef6b3c48d43b19c9f5da8ae28613e42890a1bd579d91fcdd9214192e7ac8d` |
| robot/mc/motion/msg/McPresetMotion.msg | `02bc6caf0c47ecf8a55ebbea599aeff36047be56d975258c58ca19cecbb7f4d2` |
| robot/common/msg/CommonTaskResponse.msg | `6e80c5a0ff60087b8fb0f6bf8747eb01504da7e71ae431944e2056c98ce171a2` |
| robot/common/msg/ResponseHeader.msg | `d5d78dddcee2ab78e32c4ad81cac5d69b078a99fc637ba4a08fb9a03ba52d87f` |
| robot/common/msg/CommonState.msg | `2dfa028420c9906499bd489dc3761f36503f33920302ae14b112c9bf457489d7` |
| 现有新版GetMcPresetMotionState.srv | `2482d329c1988a9c97cc7c3f90a2c76af4f9659cd252e6120ce446cd44b941ff` |

新目录的 `compatibility-inspection.json` 保存逐字段结果/递归闭包/源hash/MC元数据地址，SHA256 `8af884b74726ee9f2fbaa699566ab9ef65cdec1837fb9cc19d78642a93f340a8`；`selected-sources.json` 保存方案使用的38个不同源文件、41个带版本的类型身份，文件SHA256 `3113d897dc55dd928e606d27f1b327c564592e07ea13f60a4bb706c44abd3b74`。所有84源文件另有 `official-source-sha256.json`，SHA256 `1b018b76641c8f28f1dc33a96d5b883916d87592fd1d301a1c514f5914175608`。只读分析脚本 `inspect-compatibility.py` SHA256 `14b137e93db4c8b313e369a9d6fa3e6a4929b997a2893d3fa064cbacc41b704c` 一并保留。

## 隔离构建所需条件（尚未执行）

官方旧消息包 `CMakeLists.txt` SHA256 `f8cce59abbacf83231f9481c0470fd9dd3f25ff56912a3000af88e05c36753f3`，`package.xml` SHA256 `47ab858b108fb5ee9e8889638ce98ec8e2d9af5757360f3c3712b534633a62a5`。只需构建 `aimdk_msgs`，无需编译例程或MC。其声明依赖为 `ament_cmake`、`rosidl_default_generators`、`std_msgs`、`action_msgs`、`geometry_msgs`、`sensor_msgs`、`nav_msgs`；基础工具为CMake/C++编译器/colcon/Python。现有 `deploy/x2/Dockerfile` 的ROS Humble desktop-full基镜像与colcon/CMake/build-essential配置提供对应构建基础；未启动容器再次核验安装状态，也不新增依赖。

包内有同名C++/C/Python ROS支持库：`prebuilt_aarch64/lib/libaimdk_msgs__rosidl_typesupport_{cpp,introspection_cpp,fastrtps_cpp}.so`等，**只有aarch64预编译，不能装入当前linux/amd64进程**。官方 `cmake/aimdk_msgs_prebuilt.cmake` 按架构选择包；新建x86工作副本没有 `prebuilt_x86_64` 时会源码构建。其install阶段还会回写工作副本中的预编译目录，因此保留ZIP和只读源样本，另建完整可构建工作副本，build/install/log分别隔离；不在现有 `packages/aimdk` 内构建。

## 最小双进程接口与环境边界

1. **新版主进程**保持现有输入源、速度、odom、MC状态、关节订阅，以及 `GetMcPresetMotionState`。它继续负责预检、一次在途、执行归属、期限、UNKNOWN和保护；不得将服务ACK当成挥手完成。
2. **旧版兼容进程**仅加载原样官方v1.0.0 `aimdk_msgs`，只创建固定Set服务客户端。启动后回报消息/源码身份与ready；只接受由主进程发来的单个固定 `motion=1002, area=2, interrupt=false` 请求，以及关联run/execution/token，不接受任意服务、ROS名、路径或脚本。使用有界stdin/stdout JSON消息或同等受限IPC；发送一次后仅回报原始响应code/task_id/state及发送/接收时刻，随后退出。不得自行重试、猜取消请求或替主进程判断完成。
3. **ROS可见性共用，消息装载隔离。** 两进程与MC使用同一既有ROS_DOMAIN_ID=47、ROS_LOCALHOST_ONLY=1、rmw_fastrtps_cpp和网络命名空间；服务绝对名保持不变。旧进程用新prefix，主进程保持现有1.1prefix；不得在一个Python进程同时import两套同名 `aimdk_msgs`，不得用MC目录的同名`.so`覆盖任一客户端库。
4. **干净启动旧环境。** 当前 `/etc/profile.d/x2.sh` 和root的bashrc会自动source新版SDK，因此旧兼容进程/构建不能继承登录shell配置。应使用不读profile/rc的独立shell，清理继承的 `AMENT_PREFIX_PATH/CMAKE_PREFIX_PATH/COLCON_PREFIX_PATH/PYTHONPATH/LD_LIBRARY_PATH/ROS_PACKAGE_PATH` 等叠加路径，再只source `/opt/ros/humble/setup.bash` 和旧包隔离prefix；必要基础系统路径/ROS通信配置显式保留。验收实际 `aimdk_msgs.__file__` 和已加载库路径，确认没有混载。
5. **未知结果保持未知。** IPC断开、旧进程异常、超时或ACK身份不匹配均保留UNKNOWN；本地杀进程/取消future不等于取消MC动作。主进程收到有效ACK后用新版查询该task_id，联合PLAYING及实际关节变化判定；不能把丢ACK当作可再次发送的授权。旧进程的发送、ACK、stderr、退出状态与来源hash归入同一新run。

下一步由单独实施契约冻结上述环境、IPC、预算和证据，再交新实施者实施并独立验收。先做不启动MC的身份/边界检查，再在既有授权与资源预算内登记一次单挥手自然完成探针；左右转向不重做。双进程方案仅解决消息兼容，**不解决尚缺的可靠挥手取消契约**。
