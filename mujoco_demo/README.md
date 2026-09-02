# mujoco_demo

A small vision-language-navigation simulation (Python package `vln_mujoco`):
it runs
in the fixed MolmoSpaces ProcTHOR 10K validation `val_2` multi-room scene, using
the ceiling MJCF variant, MuJoCo RGB camera rendering, and either the bundled
TurtleBot or an external MicroDuck model and walking policy. A single web page
provides VLN instructions, live view, WASD driving, emergency stop, reset, and
state readout.

## Highlights

- One scene: only the MolmoSpaces resources actually referenced by
  `val_2_ceiling` are bundled — no full dataset download required;
- Two robot modes: a differential-drive TurtleBot defined in-project, or an
  optional dynamically simulated MicroDuck driven by an ONNX walking policy;
- One process: Python runs MuJoCo, the web page, and the VLN WebSocket client
  together;
- TurtleBot commands are integrated kinematically; MicroDuck runs gravity,
  contacts, 14 position actuators, and policy inference at 50 Hz;
- Two views: the web page switches live between the robot's first-person RGB
  and a rear-elevated third-person follow camera;
- On-change state sync: web state is pushed only when the robot, VLN, control
  ownership, or configuration changes;
- VLN semantic feedback: the first-person view shows APOS/OPOS markers, and
  `stop=true` ends the task and releases control automatically;
- MPC control: the same CasADi/IPOPT unicycle MPC, parameters, and
  capture-time pose alignment as `robot_deploy`'s `vln_mpc`;
- No ROS 2 or Node.js; the default TurtleBot mode needs no locomotion policy;
- The web interaction and VLN server protocol are compatible with
  `robot_deploy`'s `vln_web` and `vln_client`.

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+:

```bash
./run.sh
```

Open <http://127.0.0.1:8088>. Without a VLN server configured you can still
click **Take control** on the page and drive with WASD. A default server
address can also be set at startup:

```bash
./run.sh --vln-server ws://127.0.0.1:8050
./run.sh --host 0.0.0.0 --port 8088
```

### MicroDuck (optional)

MicroDuck support deliberately keeps the robot MJCF, meshes, and walking policy
outside this repository. Obtain those files from their owners under terms that
permit your use, then start the demo with the optional ONNX Runtime dependency:

```bash
MUJOCO_GL=egl uv run --extra microduck vln-mujoco \
  --host 0.0.0.0 \
  --vln-server ws://127.0.0.1:8050 \
  --robot microduck \
  --robot-model /path/to/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml \
  --walking-policy /path/to/alpha_walking.onnx
```

The MJCF must provide `trunk_base`, `trunk_base_freejoint`, `head_camera`, the
`imu_ang_vel` three-axis sensor, and 14 joint actuators. The ONNX policy must
accept one `[1, 61]` float32 observation, return one `[1, 14]` float32 action,
and provide `joint_names`, `default_joint_pos`, `action_scale`,
`observation_names`, and `command_names` metadata. Actuator order must match
`joint_names`; incompatible files fail at startup instead of running with an
incorrect joint mapping.

The head camera is used for LightNav input, while MuJoCo trunk pose and velocity
provide MPC feedback. Commands are clipped to the policy range (`±0.30 m/s`,
`±1.50 rad/s`); small non-zero linear commands enter the stable walking range,
while commands inside the stop deadband remain zero.

Pressing space or clicking `STOP` on the page zeroes the velocity immediately.
Manual commands also auto-zero when not refreshed for 350 ms.

## Driving it with LightNav-0

Start `lightnav-serve` on a GPU host (see
[docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)):

```bash
PORT=8050 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
    --task vln --model_path checkpoints/LightNav-0 --backend vllm_local
```

Then paste `ws://<gpu-host>:8050` into the console's **VLN Server WebSocket**
field, type an instruction, and press **Start VLN**. The selected simulated
robot streams its first-person frames to the server and tracks the returned
waypoints with the MPC — the same client protocol and controller the real robots use in
[`robot_deploy/`](../robot_deploy/README.md).

## Layout

```text
mujoco_demo/
├── vln_mujoco/
│   ├── assets/        # a single MolmoSpaces scene, nothing more
│   ├── web/           # static single page, no build step
│   ├── robots/        # backend protocol + TurtleBot and MicroDuck adapters
│   ├── model.py       # scene loading and MuJoCo compilation
│   ├── simulation.py  # shared runtime, cameras, watchdog, frame capture
│   ├── mpc.py         # CasADi/IPOPT kinematic MPC
│   ├── vln_client.py  # VLN WebSocket client
│   └── server.py      # HTTP/WebSocket and control ownership
├── scripts/
├── tests/
└── run.sh
```

The simulation runtime is independent of the bundled TurtleBot embodiment.
Robot implementations satisfy the `RobotBackend` protocol in
`vln_mujoco/robots/base.py`: they provide a MuJoCo model and data, consume the
shared planar velocity command, report pose and velocity, and select the two
render cameras. The runtime continues to own threading, command timeout,
rendering, frame timestamps, and the server-facing snapshot. A new embodiment
therefore does not need to modify the VLN client, MPC, web server, or render
loop.

Body-frame waypoints returned by VLN are first transformed into the world frame
using the robot pose at image-capture time, then tracked by the MPC in the
current robot-local frame. The control rate, horizon, model, cost, constraints,
IPOPT settings, and parameters all match `robot_deploy`'s `vln_mpc`:

| Parameter | Value |
| --- | --- |
| control rate / `horizon` | `10 Hz` / `5` |
| MPC / waypoint `dt` | `0.1 s` / `0.1 s` |
| `track_v_max` / `objnav_v_max` | `1.5 m/s` / `0.8 m/s` |
| `w_max` | `3.0 rad/s` |
| `a_max_v` / `a_max_w` | `2.0 m/s²` / `5.0 rad/s²` |
| `q_x` / `q_y` / `q_yaw` | `10.0` / `10.0` / `1.0` |
| `r_v` / `r_w` | `0.1` / `0.1` |
| IPOPT | `max_iter=100`, `acceptable_tol=1e-8`, `acceptable_obj_change_tol=1e-6` |

## Why a single MolmoSpaces scene is enough

The official MolmoSpaces resource manager installs per scene archive, but
ProcTHOR scene XMLs also reference shared THOR meshes/textures. With the default
MolmoSpaces initialisation flow, the shared THOR object source may be unpacked
in full, so "selecting one scene" does not necessarily mean only that scene's
files end up on disk.

This project runs a pruning script once before release: it parses every `file=`
reference in the fixed scene XML and copies only the meshes/textures inside that
closure, preserving the original directory layout. The current asset closure is
62.4 MiB, so after cloning there is no need for the MolmoSpaces repo or its
downloader.

Developers can regenerate the assets from an existing MolmoSpaces checkout:

```bash
python3 scripts/vendor_molmospaces_scene.py \
  ~/Desktop/molmospaces/assets/scenes/procthor-10k-val/val_2_ceiling.xml
```

The upstream revision, file list, and checksums are recorded in
`vln_mujoco/assets/manifest.json`.

## Attribution and licensing

Developed and maintained by [Light Origins](https://www.lightorigins.com/en).
Copyright 2026 Light Origins.

The project source code is licensed under the
[Apache License 2.0](../LICENSE). The bundled MolmoSpaces ProcTHOR `val_2` scene
and THOR assets are licensed under CC BY 4.0; attribution, modification notes,
and license links for dependencies and assets are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

TurtleBot is a trademark of Open Robotics (Open Source Robotics Foundation).
The robot in this demo is an original simplified differential-drive geometry
created for this project, inspired by the TurtleBot form factor; it uses no
official meshes or design files, and this project is not affiliated with or
endorsed by Open Robotics or ROBOTIS.

## Known limits

- The TurtleBot is a lightweight geometry drawn for this project, not the
  official high-fidelity TurtleBot3 mesh;
- MicroDuck mode requires external MJCF assets and a compatible ONNX policy;
- The spawn pose is fixed at `(x=6.5, y=13.8, yaw=0)` and environment objects
  are frozen;
- RGB only for now, at a camera resolution of `480 × 270`;
- The MPC only tracks the local waypoints provided by VLN — no navigation map,
  global planning, collision response, or ROS interface.
