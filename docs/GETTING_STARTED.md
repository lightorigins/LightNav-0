# Getting Started

Everything needed to install LightNav-0, run a first prediction, reproduce the benchmark
numbers and drive a real robot. The [README](../README.md) covers what the model is and how
it scores; this page covers how to use it.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Simulation demo](#simulation-demo)
- [Evaluation](#evaluation)
- [Real-robot deployment](#real-robot-deployment)
- [Visualisation](#visualisation)
- [Reference documentation](#reference-documentation)

## Installation

### 1. Inference environment (Python 3.11, CUDA GPU)

```bash
git clone https://github.com/lightorigins/LightNav-0.git && cd LightNav-0
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,video,habitat]"      # vLLM backend + video/visualisation + Habitat client
```

`pip install -e .` alone gives the `hf` backend and the WebSocket server; add the `test`
extra for the CPU test suite (`make test`). A Docker image is provided
(`docker build -t lightnav0 .`, see [DEPLOYMENT.md](DEPLOYMENT.md)).

**Blackwell GPUs (including RTX 5090 `sm_120` and B300/B30Z `sm_103`):** PyPI's
`torch 2.10.0` is a cu12.8 build. On these GPUs, the cu12.8 stack can fail in the vLLM
initialisation path (or the `hf` vision tower) with a native CUDA crash. In particular,
the `hf` backend on `sm_103` can report `nvrtc: error: invalid value for --gpu-architecture`.
Install the cu12.9 wheels on these GPUs:

```bash
pip install --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.10.0+cu129" "torchvision==0.25.0+cu129"
```

The `+cu129` suffix matters: pip treats a bare `torch==2.10.0` as already satisfied by the
cu12.8 build and would leave it in place.

### 2. Checkpoint and action decoder

```bash
hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0
```

In general a checkpoint is a Hugging Face directory (`config.json`, `model*.safetensors`,
`tokenizer*`, `processor_config.json`) with an `eval_config.json` and an RVQ action-tokenizer
bundle (`action_tokenizer/`) next to it. See [CONFIGURATION.md](CONFIGURATION.md).

### 3. Habitat environment (VLN-CE / ObjectNav evaluation only; Python 3.9, separate conda env)

```bash
conda env create -f habitat_server/environment.yml && conda activate habitat
pip install --no-deps "habitat-lab==0.3.20231024" && pip install --force-reinstall "numpy>=1.20,<1.24"
pip install -e habitat_server
```

Datasets (R2R / RxR VLN-CE, HM3D / MP3D ObjectNav v1, HM3D-OVON) and scenes (MP3D, HM3D) go
under `data/`, see [HABITAT_SERVER.md](HABITAT_SERVER.md).

### 4. EVT-Bench (tracking evaluation only)

Install [EVT-Bench](https://github.com/wsakobe/TrackVLA) following its README, then add our
client:

```bash
cp evt_bench/trackvla_client_agent.py ~/EVT-Bench/
(cd ~/EVT-Bench && git apply /path/to/LightNav-0/evt_bench/run_py.patch && pip install websocket-client)
```

See [EVAL_EVT_BENCH.md](EVAL_EVT_BENCH.md).

## Quick Start

### Offline prediction on a video clip

A released checkpoint ships its own action decoder, so `--model_path` is the only asset
argument needed:

```bash
lightnav-predict --model_path checkpoints/LightNav-0 \
    --backend vllm_local --video clip.mp4 --fps 4 \
    --instruction "follow the person in the red shirt"
```

### Serve the model and drive it with the reference client

```bash
PORT=8050 CUDA_VISIBLE_DEVICES=0 lightnav-serve --task tracking \
    --model_path checkpoints/LightNav-0 --backend vllm_local

lightnav-ws-client --server ws://localhost:8050 --video clip.mp4 --fps 4 \
    --instruction "follow the person in the red shirt"
```

`--task vln` selects the navigation prompt. A checkpoint that ships **no** decoder needs it
passed explicitly: `--action_tokenizer_bundle /path/to/action_tokenizer` (see
[CONFIGURATION.md](CONFIGURATION.md)).

### Python API

```python
from lightnav.tracking import build_tracking_agent

agent = build_tracking_agent("checkpoints/LightNav-0")   # decoder read from the checkpoint
agent.reset(instruction="follow the person in the red shirt")
for frame in rgb_frames:            # HWC uint8 RGB
    agent.observe(frame)
waypoints, raw_text, latency_ms = agent.predict_waypoints(agent.instruction)   # (H, 3)
```

## Simulation demo

[`mujoco_demo/`](../mujoco_demo/) is a self-contained MuJoCo TurtleBot in a bundled ProcTHOR
scene — no ROS, no Habitat, no GPU on the client side:

```bash
cd mujoco_demo && ./run.sh        # needs uv; then open http://127.0.0.1:8088
```

Point the web console at your `lightnav-serve` address and type an instruction; it drives with
the same MPC and client protocol as the real robots in
[`robot_deploy/`](../robot_deploy/README.md).

![MuJoCo demo: the simulated robot navigates to the trashcan from a language instruction](assets/mujoco_demo.gif)

## Evaluation

| Benchmark | Split | Episodes | Success radius |
|---|---|---|---|
| VLN-CE R2R | `val_unseen` | 1,839 | 3.0 m |
| VLN-CE RxR (en-US + en-IN) | `val_unseen` | 3,669 | 3.0 m |
| ObjectNav HM3D v1 | `val` | 2,000 | 0.1 m |
| ObjectNav MP3D v1 | `val` | 2,195 | 0.1 m |
| ObjectNav HM3D-OVON | `val_seen` / `val_seen_synonyms` / `val_unseen` | 3,000 each | 0.25 m |
| EVT-Bench (DT / STT) | `val`, 30 shards | 1,405 each | SR / TR / CR |

### Habitat

One env server + one eval client per GPU, sharded and merged automatically:

```bash
MODEL_PATH=/path/to/hf_ckpt bash scripts/eval_habitat.sh                                  # R2R, all GPUs
MODEL_PATH=/path/to/hf_ckpt HABITAT_CONFIG=habitat_server/configs/vlnce_rxr.yaml \
    LANGUAGES="en-US en-IN" GPU_IDS="0 1" bash scripts/eval_habitat.sh                     # RxR
MODEL_PATH=/path/to/hf_ckpt TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_hm3d_v1.yaml \
    SPLIT=val bash scripts/eval_habitat.sh                                                 # HM3D v1
MODEL_PATH=/path/to/hf_ckpt TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_mp3d.yaml \
    SPLIT=val bash scripts/eval_habitat.sh                                                 # MP3D v1
MODEL_PATH=/path/to/hf_ckpt TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_ovon.yaml \
    SPLIT=val_unseen SUCCESS_DISTANCE=0.25 bash scripts/eval_habitat.sh                    # HM3D-OVON (also val_seen / val_seen_synonyms)
```

Results land in `output/habitat_<task>_<timestamp>/summary.json` (SR / OS / SPL / NDTW / NE
for VLN-CE; SR / SPL for ObjectNav). Add `CLIENT_ARGS="--save_video"` for per-episode overlay
videos. Single-process usage, flags and outputs: [EVAL_HABITAT.md](EVAL_HABITAT.md).

### EVT-Bench

Start tracking servers, run the 30 shards, aggregate with `analyze_results.py`:

```bash
MODEL_PATH=checkpoints/LightNav-0 \
    NUM_GPUS=2 SERVERS_PER_GPU=4 EVT_BENCH_REPO=$HOME/EVT-Bench TASK_VARIANTS="dt stt" \
    bash scripts/eval_evt_bench.sh
```

Keep `CHUNKS=30` (the shard count is part of the benchmark definition). Details, the
jaw-camera FOV option and the metric definitions: [EVAL_EVT_BENCH.md](EVAL_EVT_BENCH.md).

## Real-robot deployment

The model runs on a GPU host behind `lightnav-serve`; the robot runs a thin WebSocket client
(any language) that streams JPEG frames + the instruction and executes the first returned
waypoint each control period:

```python
import base64, json
from websockets.sync.client import connect

with connect("ws://gpu-host:8050", max_size=64 * 1024 * 1024) as ws:
    ws.send(json.dumps({"action": "login", "data": {"clientId": "robot-01"}})); ws.recv()
    ws.send(json.dumps({"action": "reset", "data": {}})); ws.recv()            # new episode
    for seq, jpeg in enumerate(camera_jpegs):
        ws.send(json.dumps({"action": "next", "data": {"seq": seq,
                            "image": base64.b64encode(jpeg).decode(),
                            "instruction": "follow the person in the red shirt"}}))
        data = json.loads(ws.recv())["data"]
        if data["rc"] == 0 and "actions" in data:
            fwd, lat, yaw = data["actions"]["actions"][0]                     # robot-local, +lat = left
            if data["stop"]: break
```

Waypoint rows carry no time base of their own: command `v = fwd / dt`, `w = yaw / dt` with
`dt` = your control period, or normalise by the per-step maxima as the reference clients do.
Cameras that are not 16:9 can be served without squeezing via `--aspect_mode keep`
([CONFIGURATION.md](CONFIGURATION.md)). Several robots can share one server (sessions are
micro-batched). Server flags, the full protocol and recording: [DEPLOYMENT.md](DEPLOYMENT.md),
[PROTOCOL.md](PROTOCOL.md).

Don't want to write the robot side yourself? [`robot_deploy/`](../robot_deploy/) is a complete
ROS 2 on-robot stack — camera driver, this WebSocket client, an MPC waypoint tracker, and a
web control panel — with adapters for the Unitree Go2 and LimX TRON 1, and a
[bring-your-own-robot](../robot_deploy/README.md#bring-your-own-robot) adapter interface.

## Visualisation

Every prediction can be rendered on the robot's own frames as a ground-plane trajectory
ribbon, pointing markers and a telemetry HUD:

```bash
lightnav-serve ... --record_dir output/episodes --cam_hfov_deg 112 --cam_height 0.45   # record on the server
lightnav-render output/episodes                                                          # -> traj_pointing.mp4
CLIENT_ARGS="--save_video" MODEL_PATH=... bash scripts/eval_habitat.sh                   # per-episode eval videos
```

See [VISUALIZATION.md](VISUALIZATION.md).

## Reference documentation

| Document | Content |
|---|---|
| [CONFIGURATION.md](CONFIGURATION.md) | Checkpoint layout, action decoders, `eval_config.json`, every server / CLI parameter |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Real-robot deployment, client loop, velocity mapping, Docker, Python API |
| [PROTOCOL.md](PROTOCOL.md) | JSON-over-WebSocket wire protocol |
| [EVAL_HABITAT.md](EVAL_HABITAT.md) | VLN-CE / ObjectNav evaluation client, multi-GPU sharding, output schema |
| [HABITAT_SERVER.md](HABITAT_SERVER.md) | Habitat conda environment, datasets, env server CLI |
| [EVAL_EVT_BENCH.md](EVAL_EVT_BENCH.md) | EVT-Bench setup, sharding, metrics |
| [VISUALIZATION.md](VISUALIZATION.md) | Overlay rendering, recording layout, `lightnav-render` |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Lint / test commands, GPU smoke test (`scripts/smoke_gpu.sh`) |
| [../robot_deploy/README.md](../robot_deploy/README.md) | ROS 2 on-robot stack: fresh-machine setup, per-robot launch, web panel, adapter interface |
| [../mujoco_demo/README.md](../mujoco_demo/README.md) | Self-contained MuJoCo simulation demo: bundled scene, web console, same MPC/protocol as robot_deploy |
