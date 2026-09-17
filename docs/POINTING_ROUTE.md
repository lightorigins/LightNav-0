# 路线与 Pointing 辅助流程

本文描述当前 Isaac Sim + Go2 流程中的路线辅助功能。它把 Isaac Sim 中的世界坐标路线
转换成 LightNav 能理解的图像 pointing，并继续使用 LightNav 输出的 waypoint 驱动 Go2。

## 作用

路线辅助不是另一个底盘控制器，也不会替代 LightNav 的动作预测。它只负责：

1. 保存一组世界坐标路径点；
2. 根据机器人和相机的实时位姿，把当前路径点投影到 Go2 相机画面；
3. 将投影结果编码为 `prompt_pointing`，随 `next` 请求发送给 LightNav；
4. 在路径点满足距离和连续确认帧条件后推进到下一个点；
5. 所有点完成后由 bridge 锁定零速度；`pair` 模式还会产生内部 `APOS_STOP` 状态。

最终运动仍然来自模型返回的 `(H, 3)` waypoint，控制器通常只执行第一条 waypoint，下一帧
重新规划。

```text
世界坐标路线 JSON
        |
        v
实时机器人位姿 + 相机位姿
        |
        v
世界坐标点 -> 相机坐标 -> 图像像素 -> APOS/OPOS token
        |
        v
WebSocket next + prompt_pointing
        |
        v
LightNav 输出 RVQ action tokens
        |
        v
H x 3 waypoint -> Go2 policy / cmd_vel
```

## 相关代码

- `tools/isaac_pointing_picker.py`：在 Isaac Sim 中交互式选择路线点；
- `src/lightnav/pointing_route.py`：相机投影、图像范围/中心判断、连续帧确认和停止哨兵；
- `src/lightnav/world_route.py`：读取路线 JSON，并提供世界坐标路线跟随的 CPU 数学辅助；
- `/home/ubuntu/scene/data/people/scripts/lightnav_isaac_bridge.py`：在 Isaac Sim 中取得实时
  Go2 位姿和相机位姿，生成 `prompt_pointing`，发送请求并显示 HUD。

## 1. 生成路线文件

在 Isaac Sim 环境中运行 picker，选择场景和路线点：

```bash
cd /home/ubuntu/xc/LightNav-0
/home/ubuntu/miniconda3/envs/isaacsim6/bin/python \
    tools/isaac_pointing_picker.py --scene traffic
```

如果只需要记录机器人起点和最终目标，不需要中间 APOS 点，可使用：

```bash
/home/ubuntu/miniconda3/envs/isaacsim6/bin/python \
    tools/isaac_pointing_picker.py --scene traffic --start-and-opos
```

该模式第一次点击记录 OPOS，第二次点击记录起点；第三次及之后的点击会被忽略。

新路线 JSON 只保存有序的 `points` 坐标数组，不再写入 `opos`、`apos` 或 `role` 字段：

```json
{"points": [[far_x, far_y, far_z], [via_x, via_y, via_z], [start_x, start_y, start_z]]}
```

约定是最后一个点为起点，其余点的 pointing 语义由运行模式决定：`pair` 将中间点作为
APOS、首点作为最终 OPOS，`multi_opos` 则将起点之外的所有点作为 OPOS。读取器仍兼容
旧的 `start`/`opos`/`apos` 格式和早期带 `role` 的 `points` 格式。

默认输出到：

```text
/home/ubuntu/scene/pointing_routes/traffic_route.json
```

路线文件的点击顺序是从远端目标方向回到起点，因此运行时会反向使用起点之外的点。
`pair` 把这些点依次作为 APOS，最后回到固定的最终 OPOS；`multi_opos` 把它们全部作为
依次切换的 OPOS。

加载路线启动 Go2 时，控制器会把最后一点的 XY 作为机器人初始位置（Z 继续使用场景配置），
并自动设置初始 yaw，使机器人朝向当前模式下的第一个执行目标。对 `multi_opos` 而言，
这就是最靠近起点、最先切换到的 OPOS。

## 2. 启动 LightNav 服务

路线辅助仍然需要模型服务，因为路线点是作为模型输入 pointing 提供的：

```bash
cd /home/ubuntu/xc/LightNav-0
conda activate "$PWD/.venv"

PORT=8050 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
    --task vln \
    --model_path checkpoints/LightNav-0 \
    --backend vllm_local
```

rm -f /tmp/lightnav.ready


env -u PYTHONPATH -u LD_LIBRARY_PATH \
  VLN_VLLM_ENFORCE_EAGER=1 \
  VLN_VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  VLN_KV_CACHE_GIB=2 \
  PORT=8050 \
  CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/lightnav-serve \
  --task vln \
  --model_path checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu_memory_utilization 0.5 \
  --max_batch_size 1 \
  --ready_file /tmp/lightnav.ready


服务端加载 checkpoint 的 `eval_config.json` 和 `action_tokenizer/`，接收图像与指令后生成
pointing/action token，并把 action token 解码为 waypoint。详细协议见
[`PROTOCOL.md`](PROTOCOL.md)。

## 3. 启动带路线的 Isaac 场景

当前启动器要求 `--lightnav-route` 与 `--lightnav-server` 同时提供：

```bash
cd /home/ubuntu/scene
./start_crowd_composition.sh \
    --scene three \
    --lightnav-server ws://127.0.0.1:8050 \
    --lightnav-route /home/ubuntu/scene/pointing_routes/three_route.json \
    --lightnav-instruction "navigate through the traffic scene" \
    --no-people
```

CROWD_LIGHTNAV_POINTING_MODE=multi_opos \
./start_crowd_composition.sh \
  --scene traffic \
  --lightnav-server ws://127.0.0.1:8050 \
  --lightnav-route pointing_routes/traffic_route.json \
  --lightnav-instruction "walk on the crosswalk" \
  --no-people

`--no-people` 只关闭行人，不会关闭 Go2。LightNav 模式会自动启用一个 Go2，并使用其第一
视角相机采集图像。


仅使用最终 OPOS：

```bash
CROWD_LIGHTNAV_POINTING_MODE=opos_only ./start_crowd_composition.sh \
    --scene traffic \
    --lightnav-server ws://127.0.0.1:8050 \
    --lightnav-route /home/ubuntu/scene/pointing_routes/traffic_route.json \
    --lightnav-instruction "walk on the crosswalk" \
    --no-people
```

多 OPOS（最后一点是起点，其余点按靠近起点到远端的顺序依次作为 OPOS）：

```bash
CROWD_LIGHTNAV_POINTING_MODE=multi_opos ./start_crowd_composition.sh \
    --scene traffic \
    --lightnav-server ws://127.0.0.1:8050 \
    --lightnav-route /home/ubuntu/scene/pointing_routes/traffic_route.json \
    --lightnav-instruction "walk on the crosswalk" \
    --no-people
```

```bash
CROWD_LIGHTNAV_POINTING_MODE=multi_opos ./start_crowd_composition.sh \
    --scene three \
    --lightnav-server ws://127.0.0.1:8050 \
    --lightnav-route /home/ubuntu/scene/pointing_routes/three_route.json \
    --lightnav-instruction "navigate through the scene" \
    --no-people
```

## 4. 每帧数据流

桥接器按约 5 Hz 采集 Go2 相机帧，并为每一帧执行以下操作：

1. 从 Isaac Sim 获取实时机器人世界坐标和旋转；
2. 根据相机的静态外参重建动态相机位姿；
3. 将当前路线目标投影到图像；
4. 在 `pair` 中把当前目标编码成 `apos_id`、最终目标编码成 `opos_id`；
5. 在 `opos_only` 和 `multi_opos` 中只把当前目标编码成 `opos_id`；
6. 发送：

   ```json
   {
     "action": "next",
     "data": {
       "seq": 123,
       "image": "<base64 JPEG>",
       "instruction": "navigate through the traffic scene",
       "prompt_pointing": {
         "apos_id": 812,
         "opos_id": 1044
       }
     }
   }
   ```

`prompt_pointing` 是语义 pointing id，不是 tokenizer 的内部词表 id。服务端会校验范围，
然后把它追加到本次模型输入中。

如果路线只提供最终 OPOS，也可以只发送 `{"opos_id": <int>}`。这会启用兼容性的两阶段
推理：模型先生成 APOS，再把生成的 APOS 与外部 OPOS 一起用于动作 token 推理。当前 Isaac
桥接器可通过 `CROWD_LIGHTNAV_POINTING_MODE=opos_only` 使用该模式；此时外部路线中的 APOS
不参与距离计算、路线推进或停止判断，`distance_m` 始终是机器人到最终 OPOS 的距离。
仅 OPOS 模式适用于带 `<apos_*>` 网格 pointing token 的 checkpoint。

`CROWD_LIGHTNAV_POINTING_MODE` 当前支持三种值：

- `pair`（默认）：最后一点是起点，中间点作为依次切换的 APOS，第一个点是固定的最终
  OPOS；每帧发送 `apos_id` 和 `opos_id`；
- `opos_only`：只使用第一个点作为最终 OPOS，忽略中间点，每帧只发送 `opos_id`；
- `multi_opos`：最后一点是起点，其余所有点都作为 OPOS。执行时从靠近起点的点开始，
  按与 `pair` 模式 APOS 相同的距离门限和确认帧规则逐个切换，每帧只发送当前
  `opos_id`。

`opos_only` 和 `multi_opos` 的每一帧都执行两阶段推理：第一阶段让模型生成一个 APOS；
第二阶段把该 APOS 与 bridge 提供的当前 OPOS 追加到同一图像/指令上下文，然后约束模型
只生成 action token。切换 OPOS 后，下一帧的第二阶段会使用新的 OPOS。

## 5. 路径点推进规则

当前实现中的 `AposRoute` 只有在当前点满足条件时才推进：

- 库默认要求点处于图像中央区域；当前 Isaac bridge 设置 `require_visible=False`，因此
  bridge 的路线推进不受点是否出现在图像内影响；
- 机器人到点距离默认不超过 3 m，可通过 `CROWD_ROUTE_REACH_M` 调整；
切换点的距离由环境变量 CROWD_ROUTE_REACH_M 设置：
- 实际场景桥接代码：[lightnav_isaac_bridge.py (line 129)](/home/ubuntu/scene/data/people/scripts/lightnav_isaac_bridge.py:129)

- `CROWD_ROUTE_CONFIRM_FRAMES` 默认是 1，可提高以抑制抖动；
- 在 `pair` 模式下，点不满足距离条件时不推进路线，也不会错误地把点强行夹到图像边缘；
- 在 `opos_only` 模式下，距离条件只针对最终 OPOS；
- 在 `multi_opos` 模式下，距离条件针对当前 OPOS；达到阈值后立即切换到下一个 OPOS；
- 为防止模型生成的 APOS 绕过当前 OPOS，`multi_opos` 还支持“已越过”切换：当前距离相对
  历史最近值回升至少 `0.5 m`、下一 OPOS 已经更近、且机器人已越过当前点朝下一点方向
  的垂直平面时，按相同的确认帧数切换到下一 OPOS；
- 回升量可通过 `CROWD_ROUTE_PASS_HYSTERESIS_M` 调整。最终 OPOS 没有下一点，因此不会被
  越过逻辑跳过，仍必须进入正常到达半径。

路线推进日志中的 `reason=reached` 表示进入了正常到达半径，`reason=passed` 表示触发了
上述越过保护。调试状态同时记录当前距离、历史最近距离和下一目标距离。

路线完成后在成对 pointing 模式返回 `APOS_STOP`；在 `opos_only` 模式下不会把 APOS
哨兵发送给模型，且路线只跟踪最终 OPOS。`multi_opos` 同样不发送 APOS 哨兵，在最后一个
OPOS 到达后完成路线。路线完成由 bridge 自身的 OPOS 到达判断负责停车。

当前 bridge 还会在路线控制器确认机器人进入最终 `opos` 的到达半径后设置
`route_done`，并持续发布零速度，避免后续模型响应重新驱动车辆。切换 instruction
时会清除该停止锁存并重置路线。

## 6. waypoint 到 Go2 控制

LightNav 返回的每一行 waypoint 是机器人坐标系下的：

```text
[forward_m, lateral_m, yaw_rad]
```

其中 lateral 正方向为左，yaw 正方向为逆时针。桥接器取第一条非零 waypoint，转换成 Go2
策略输入或 `/cmd_vel`。后续 waypoint 是短期前视轨迹，不会一次性全部执行。

调试画面会显示：

- 蓝色 waypoint 轨迹；
- APOS 和 OPOS 的输入/输出 pointing 标记；
- 当前推理延迟、waypoint 数量和 WP0；
- 当前路线点索引和距离信息。

## 7. 安全联调顺序

第一次联调必须保持运动关闭：

```bash
cd /home/ubuntu/jingzhang_new
GO2_ROS_DOMAIN_ID=42 ./Nav2/start_lightnav_bridge.sh --ros-args \
    -p instruction:="navigate through the traffic scene" \
    -p enable_motion:=false
```

先确认：

```bash
ros2 topic hz /realsense/color/image_raw
ros2 topic echo /lightnav/status
ros2 topic info /cmd_vel --verbose
```

并确认日志中有非零 `latency`、`wp0` 和 waypoint 数量，再开启运动：

```bash
ros2 param set /lightnav_isaac_bridge enable_motion true
```

桥接器还会在以下情况发布零速度：

- WebSocket 断开或服务端返回错误；
- 控制命令超过 `command_timeout_s` 未刷新；
- LightNav 返回 `stop`；
- 最终停止条件满足。

当前 ROS 桥接默认要求连续 3 帧 `stop=true`，并且每帧都有 `apos_state=stop`，才把停止
状态升级为终止状态。确认不足时会保持零速度并继续推理，状态显示为
`stop_pending`。

## 8. 常见误区

- `prompt_pointing` 不是让服务端直接执行地图路线；它只是给模型的空间输入。
- 不能只看到 Isaac Sim 页面就认为推理成功，必须检查 `latency`、waypoint 和
  `/lightnav/status`。
- 路线坐标使用 Isaac Sim 世界坐标；pointing id 使用图像网格坐标，两者不能直接混用。
- 路线模式仍需先启动 `lightnav-serve`；以当前 `start_crowd_composition.sh` 的校验逻辑为准。
- 不要同时启动其他会发布 `/cmd_vel` 的导航或控制器。
