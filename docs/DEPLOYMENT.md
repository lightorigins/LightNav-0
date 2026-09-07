# Real-robot deployment

Deployment is a two-process setup: **the model runs on a GPU host behind
`lightnav-serve`**, and **the robot runs a thin WebSocket client** that streams camera frames
and executes the returned waypoints. The robot side needs no GPU and no torch — only a
WebSocket library and a JPEG encoder.

This document covers the server side and the client protocol. A complete ROS 2 reference
implementation of the robot side — camera, client, MPC waypoint tracking, web control
panel, and robot adapters (Unitree Go2, LimX TRON1) — lives in
[`robot_deploy/`](../robot_deploy/README.md), and
[`mujoco_demo/`](../mujoco_demo/README.md) exercises the same protocol and MPC
against a simulated TurtleBot, no robot required.

The GPU host does not have to be remote: [JETSON_THOR.md](JETSON_THOR.md) documents
serving the model **onboard** from a robot's own NVIDIA Jetson Thor (SM 11.0 kernel
pitfalls, unified-memory budgeting, ViT-cache sizing, a measured 4+ Hz closed loop),
with a matching launcher in [`scripts/serve_thor.sh`](../scripts/serve_thor.sh).

```
 robot (any language)                                GPU host
 ┌──────────────────────────┐    ws://host:8050     ┌────────────────────────────┐
 │ camera ─▶ JPEG ─▶ base64 │ ───── next ─────────▶ │ lightnav-serve             │
 │ instruction text         │ ◀── waypoints/stop ── │  one engine per GPU        │
 │ waypoint[0] ─▶ velocity  │                       │  micro-batched sessions    │
 └──────────────────────────┘                       └────────────────────────────┘
```

## Start the server on the GPU host

```bash
# tracking prompt (follow a person / object). A released checkpoint ships its own decoder,
# so --model_path is the only asset argument.
PORT=8050 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
    --task tracking \
    --model_path checkpoints/LightNav-0 \
    --backend vllm_local --gpu_memory_utilization 0.85

# navigation prompt (instruction / object-goal navigation)
PORT=8051 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
    --task vln \
    --model_path checkpoints/LightNav-0 \
    --backend vllm_local

# a checkpoint that ships NO decoder: pass the RVQ bundle explicitly
PORT=8052 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
    --task vln \
    --model_path /path/to/hf_ckpt \
    --action_tokenizer_bundle /path/to/action_tokenizer --horizon 10 \
    --backend vllm_local
```

The server runs one synthetic warm-up inference before binding the port, so an open port
(or the `--ready_file`) means the engine is actually ready. Several robots can share one
server: each WebSocket connection is an independent session (own frame history, own ViT
cache) and concurrent requests are micro-batched onto the shared engine. For several GPUs
use `scripts/start_servers.sh` (one process per GPU) or the Docker image (see *Docker image* below) and put a
plain TCP load balancer in front — sessions are connection-scoped, so any balancer that
keeps a connection on one backend works.

## Robot client loop

Per episode (a new instruction / a new goal):

1. `login` once per connection, then `reset` — clears the server-side frame history.
2. Every control tick: capture the RGB frame, JPEG-encode it, send `next` with a
   monotonically increasing `seq`, the frame and the **current instruction** (the
   instruction travels on every `next`, so it can be changed mid-episode without a reset).
3. Read the reply: `actions.actions` is the `(H, 3)` chunk, `stop` the arrival flag,
   `visible` the target-visibility flag (tracking checkpoints), `pointing` the target pixel
   (pointing checkpoints).
4. Execute the **first waypoint** as a velocity command for one control period and repeat —
   the model replans every frame, so the remaining rows are only a look-ahead.

Frame rate: send frames at the checkpoint's training rate (`video_fps` in
`eval_config.json`, typically 4 Hz) or faster; every frame sent is appended to the history
window, so a client that only wants to warm the history (no prediction) sends `next` with an
empty `instruction` and gets `{"rc": 0, "msg": "image received"}` back. Send frames at the
camera's native resolution; the server resizes to the checkpoint's `video_size`.

Waypoint → velocity. Each row is a robot-local displacement for one *planning step*; the
trajectory vocabularies cap a step at roughly 0.25 m of translation and 30° of yaw. The
rows carry **no time base of their own**: choose `dt` = your control period and command

```
v_forward = forward_m / dt          w_yaw = yaw_rad / dt          (lateral_m for holonomic bases)
```

clipped to the platform's limits. The reference clients (`lightnav-ws-client`, the EVT-Bench
agent) sidestep the choice by normalising the first waypoint by the per-step maxima and
clipping to `[-1, 1]`: `vx = clip(fwd / 0.375)`, `vy = clip(lat / 0.25)`,
`vyaw = clip(yaw / (π/20))`, then scaling by the platform's top speeds.
`lightnav.velocity.first_waypoint_to_velocity_cmd` implements the Habitat-style
normalised mapping if you want to reuse it. (The visualisation HUD's velocity readout
assumes 0.1 s per step; that is a display convention, see
[VISUALIZATION.md](VISUALIZATION.md).) Treat `stop=true` (the model emitted the
stop action) as "goal reached": halt and end the episode. Errors are non-fatal (`rc` 400/500
with a message); reuse the previous command or stop, and keep the connection.

Minimal Python client (any language works — the protocol is plain JSON):

```python
import base64, io, json, math
import numpy as np
from PIL import Image
from websockets.sync.client import connect          # websockets>=12

SERVER = "ws://gpu-host:8050"
V_MAX, W_MAX = 0.5, 1.0                              # your platform's top speeds (m/s, rad/s)

def jpeg_b64(rgb):                                    # rgb: HWC uint8 RGB numpy array
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

with connect(SERVER, max_size=64 * 1024 * 1024) as ws:
    ws.send(json.dumps({"action": "login", "data": {"clientId": "robot-01"}}))
    assert json.loads(ws.recv())["data"]["rc"] == 0
    ws.send(json.dumps({"action": "reset", "data": {}}))          # new episode
    assert json.loads(ws.recv())["data"]["rc"] == 0

    seq = 0
    while not done:
        frame = camera.read_rgb()                                 # your camera
        ws.send(json.dumps({"action": "next", "data": {
            "seq": seq, "image": jpeg_b64(frame),
            "instruction": "follow the person in the red shirt",
        }}))
        seq += 1
        data = json.loads(ws.recv())["data"]
        if data["rc"] != 0 or "actions" not in data:
            continue                                              # keep last command
        if data["stop"]:
            robot.stop(); break
        fwd, lat, yaw = data["actions"]["actions"][0]             # first waypoint, robot frame
        vx = max(-1.0, min(1.0, fwd / 0.375))                     # normalised by per-step maxima
        vyaw = max(-1.0, min(1.0, yaw / (math.pi / 20)))
        robot.set_velocity(vx * V_MAX, vyaw * W_MAX)              # for one control period
```

`lightnav-ws-client` is the same loop driven from an mp4 or a frame directory and is the
quickest way to check a server from the robot's network:

```bash
lightnav-ws-client --server ws://gpu-host:8050 --video clip.mp4 --fps 4 \
    --instruction "follow the person in the red shirt"
```

## Operational notes

- **Latency.** With `vllm_local` on a single modern GPU a step is ~60–150 ms including the
  ViT (only frames new to the window go through the vision tower); the `hf` backend is
  several times slower. `latency_ms` in every reply is the server-side wall time including
  queueing.
- **Connection hygiene.** Sessions live and die with the connection; reconnecting starts an
  empty history (send `reset` and re-stream). Keep one outstanding request per connection.
- **Message size.** Configure the client's WebSocket library for ≥ 64 MiB frames (the server
  uses `max_size=64*1024*1024`); the default 1 MiB limit of many libraries rejects large images.
- **Pointing checkpoints** return the target pixel in the *client's* frame size
  (`pointing.frame_size`), so the robot can overlay or track it without knowing the model's
  input resolution ([PROTOCOL.md](PROTOCOL.md)).
- **Multiple robots per GPU.** Raise `--max_batch_size` (default 8) to the number of
  concurrent sessions; the LLM decode is batched, the ViT is per-session.

## Recording and visualisation

To see what the model predicted on the robot's own frames, record on the server and
render afterwards:

```bash
# GPU host: record every connection's episodes (the client's JPEG frames + one record per prediction)
lightnav-serve ... --record_dir output/episodes --cam_hfov_deg 112 --cam_height 0.45
#   env equivalents: RECORD_DIR, CAM_HFOV_DEG, CAM_HEIGHT (also read by scripts/start_servers.sh and docker)

# afterwards, anywhere with the video extra installed:
lightnav-render output/episodes                     # -> <episode dir>/traj_pointing.mp4
lightnav-render output/episodes --height 1080 --fps 15 --forward-offset 0 --overwrite
```

Recordings land in `output/episodes/run_<timestamp>/<clientId>/episode_NNN/` (`manifest.json`,
`image_*.jpg`, `actions.json`); the server never encodes video in the serving loop. The
rendered video shows the predicted chunk as a ground-plane ribbon (projected with the
client camera's FOV and height — set `--cam_hfov_deg` / `--cam_height` for your robot),
the pointing pixels as mint (`apos`) / magenta (`opos`) discs, and a HUD with the
instruction, GO/STOP, step, step rate and first-waypoint velocities. By default the video
plays at wall-clock pace (`realtime` timebase). Details, the record schema and the eval
client's `--save_video`: [VISUALIZATION.md](VISUALIZATION.md).

---


## Docker image (server)

```bash
docker build -t lightnav0:latest .
docker run --gpus all -p 8050:8050 -v /path/to/models:/models \
    -e MODEL_PATH=/models/LightNav-0 -e TASK=tracking \
    lightnav0:latest
# a checkpoint without a shipped decoder additionally needs
#   -e ACTION_TOKENIZER_BUNDLE=/models/action_tokenizer
```

`docker/entrypoint.sh` turns `MODEL_PATH`, `ACTION_TOKENIZER_BUNDLE` (optional),
`TASK`, `BACKEND`, `GPU_MEM_UTIL`, `MAX_NEW_TOKENS`, `HOST`, `PORT`,
`MAX_BATCH_SIZE`, `MAX_WAIT_MS`, `NUM_HISTORY_FRAMES`, `POOL_SPATIAL` and the recording knobs
(`RECORD_DIR`, `RECORD_FPS`, `RECORD_TIMELINE`, `RECORD_IMAGES`, `CAM_HFOV_DEG`, `CAM_HEIGHT`,
`TRAJ_FORWARD_OFFSET`, `WAYPOINT_DT_S`; see *Recording and visualisation* above) into
`lightnav-serve` flags; extra `docker run` arguments are appended verbatim.


## Python API

```python
from lightnav.tracking import build_tracking_agent

agent = build_tracking_agent(
    model_path="checkpoints/LightNav-0",     # decoder + processing params from the checkpoint
    backend="vllm_local",                    # or "hf"
)
# A checkpoint that ships no decoder needs one passed in:
#   build_tracking_agent(model_path=..., action_tokenizer_bundle=..., horizon=10)

agent.reset(instruction="follow the person in the red shirt")
for frame in rgb_frames:                     # HWC uint8 RGB numpy arrays
    agent.observe(frame)
waypoints, raw_text, latency_ms = agent.predict_waypoints(agent.instruction)
# waypoints: (H, 3) float32 -- [forward_m, lateral_m(+=left), yaw_rad(+=ccw)]
```

`observe()` appends a frame to the history (resized to the checkpoint's `video_size` and
normalised internally); `predict_waypoints()` runs one step on the current buffer. SlowFast
checkpoints keep the full episode; others keep a ring buffer of `num_history_frames`.
`predict_waypoints(instruction, task_type="vlnce_traj")` selects the VLN prompt instead of
the tracking one.

Lower level: `lightnav.inference.InferenceConfig` + `build_engine(config, task_type)`
return the `VLNInferenceEngine` and `ModelBundle`;
`engine.generate_from_frames(video_tensor, instruction, frame_ids=..., task_type=...)`
returns the raw token text; `lightnav.velocity.first_waypoint_to_velocity_cmd` maps a
waypoint to a normalised `{linear_velocity, angular_velocity}` command;
`lightnav.habitat.run_habitat_eval(HabitatEvalConfig(...))` runs a Habitat evaluation
programmatically.

---


## Wire protocol summary

Every message is one JSON text frame; every request gets exactly one response. Full shapes,
error codes and the pointing payload: [PROTOCOL.md](PROTOCOL.md).

```
client -> {"action": "login", "data": {"clientId": "<id>"}}       # clientId optional
server <- {"action": "login", "data": {"rc": 0, "msg": "ok"}}

client -> {"action": "reset", "data": {}}                       # new episode; clears the frame buffer
server <- {"action": "reset", "data": {"rc": 0, "msg": "ok"}}

client -> {"action": "next", "data": {"seq": 12, "image": "<base64 JPEG>", "instruction": "..."}}
server <- {"action": "next", "data": {
  "rc": 0, "seq": 12,
  "actions": {"step": 64, "actions": [[0.4, 0.0, 0.0], [0.8, 0.1, 0.05], ...]},   # (H, 3)
  "stop": false, "visible": true, "latency_ms": 123.4,
  "timings_ms": {"batch_size": 1, "vit_ms": 40.1, "llm_ms": 60.2, ...},
  "raw_text": "<tpos_17><traj_3877>",
  "pointing": {...}                       # only for checkpoints that emit <apos_*>/<opos_*>
}}
```

- `actions.actions` is the predicted `(H, 3)` chunk in robot-local metres / radians;
  `actions.step` is the number of frames in the history buffer.
- An empty / `null` `instruction` only buffers the frame: `{"rc": 0, "seq": N, "msg": "image received"}`.
- `stop=true`: the model emitted the stop action. `visible`: decoded from the grounding
  token, `null` for checkpoints without one. `raw_text`: the decoded output (≤ 256 chars).
- Errors are non-fatal: malformed requests get `rc: 400`, a failed inference `rc: 500`; the
  connection stays open.

---
