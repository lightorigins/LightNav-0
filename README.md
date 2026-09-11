<h1 align="center">LightNav-0</h1>

<h3 align="center">Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation</h3>

<p align="center"><b>Light Origins Team</b></p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2608.30935-b31b1b.svg)](https://arxiv.org/abs/2608.30935)
[![Project Page](https://img.shields.io/badge/Project%20Page-9c403d?style=flat)](https://www.lightorigins.com/en/blog/lightnav-0)
[![Model](https://img.shields.io/badge/🤗%20Model-LightNav--0-yellow.svg)](https://huggingface.co/LightOriginsHQ/LightNav-0)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/zwZuD9JG)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat&logo=wechat&logoColor=white)](#community)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

</div>

<div align="center">

![LightNav-0 driving four different robots through an unseen park from language instructions, with the predicted trajectory overlaid on each robot's own camera](docs/assets/hero_cross_embodiment.gif)

*Humanoid, quadruped, wheeled and aerial robots in an unseen park, each following a target
named in language. No teleoperation, fully autonomous.*

</div>

## 🏡 About

<div align="center">
  <img src="docs/assets/teaser.png" alt="LightNav-0 overview: a simulation-based data engine, three-stage model training, zero-shot deployment onto four robot embodiments, and success-rate comparisons on ten public benchmarks" width="95%"/>
</div>

<br>

**LightNav-0** is a compact generalist embodied navigation model that elicits the spatial
intelligence of a pretrained vision-language model (Qwen3-VL) and aligns it with navigation,
without task-specific prediction heads. Diverse tasks share one token interface: dual-channel
pointing expresses task-, scene- and embodiment-agnostic spatial intent, and a residual
vector-quantized action tokenizer maps that intent to precise, embodiment-specific
trajectories — so instruction following, open-vocabulary object navigation and visual tracking
live in a single model that transfers zero-shot across robot embodiments and scenes.

## 🧠 Method

<div align="center">
  <img src="docs/assets/pipeline.png" alt="LightNav-0 architecture: a pretrained VLM backbone consumes a compressed egocentric RGB history and a language instruction, then emits dual-channel pointing tokens followed by three RVQ action tokens that decode to ten SE(2) waypoints" width="95%"/>
</div>

<br>

LightNav-0 is instantiated from **Qwen3-VL-4B-Instruct** and adds no navigation-specific
modules — no waypoint predictor, no task-specific action head, no per-embodiment expert. Only
the vocabulary is extended, with indexed pointing tokens and RVQ action tokens, so both the
spatial reasoning trace and the action codes are decoded through the backbone's original
autoregressive LM head.

At each decision step the model consumes a timestamped egocentric RGB history and a
natural-language instruction, interleaved in a single causal sequence, and emits:

1. **Dual-channel pointing** — an *affordance* point (a feasible local direction or free-space
   waypoint) and an *object* point (the task goal), each as one image-grid token. This is an
   explicit spatial reasoning trace that grounds the plan in pixels before any action is
   generated.

2. **Three RVQ action tokens**, which decode to 10 future SE(2) waypoints — a common geometric
   interface handed to each embodiment's own low-level controller.

Task semantics come entirely from the instruction; there is no task-identification token, and
the same backbone, token interface and objective serve every navigation task.

### Temporally Aware History Compression

Navigation needs both recent geometric detail and long-horizon context, but encoding every
frame at native resolution makes the visual-token count grow without bound. LightNav-0
compresses history by recency, following the shape of the Ebbinghaus forgetting curve: the
sampling rate decays exponentially with frame age while the spatial pooling stride grows
exponentially, so distant observations contribute fewer and coarser tokens and the current
observation keeps the finest detail. Timestamp tokens preserve ordering after pooling. The
compressor runs after the vision transformer under configurable pixel budgets of 256K, 576K
and 1M, bounding context length without collapsing the whole history into one fixed-resolution
summary.

### RVQ Action Tokenizer

<div align="center">
  <img src="docs/assets/rvq.png" alt="Hierarchical residual vector-quantized action tokenizer: a coarse codebook plus two residual codebooks quantize a ten-step SE(2) trajectory, and the composed codewords decode back into a trajectory" width="95%"/>
</div>

<br>

A 10-step SE(2) trajectory is quantized by a coarse 256-entry codebook and two residual
256-entry codebooks, resolving roughly 0.9 m, 7 cm and 4 cm respectively. Any non-empty token
prefix already decodes into an executable coarse trajectory, and each further residual level
refines geometric precision — so the same three tokens express both the gross motion and the
centimetre-scale shape of the path.

## 🏆 Benchmarks

One shared checkpoint, no per-benchmark fine-tuning. Every LightNav-0 number below comes from a
single forward RGB stream — no depth, odometry or panoramic rig. Baselines are the strongest
monocular entries; full tables, including NE / nDTW / CR and the panoramic comparisons, are in
the paper.

### Instruction Following (VLN-CE)

Val-unseen splits of R2R and the longer-horizon RxR.

| Model | R2R SR (%) | R2R SPL (%) | RxR SR (%) | RxR SPL (%) |
| :--- | :---: | :---: | :---: | :---: |
| NaVILA | 54.0 | 49.0 | 49.3 | 44.0 |
| StreamVLN | 56.9 | 51.9 | 52.9 | 46.0 |
| DualVLN | 64.3 | 58.5 | 61.4 | 51.8 |
| CorrectNav | 65.1 | 62.3 | 69.3 | 63.3 |
| Qwen-RobotNav-8B | 65.7 | 59.6 | 73.4 | 63.5 |
| **LightNav-0** | **68.5** | **62.8** | **73.6** | **64.5** |

### Object-Goal and Open-Vocabulary Navigation

Success rate on the six ObjectNav settings. HM3D-OVON tests category names never seen in
training, as synonyms and as entirely unseen classes.

| Model | MP3D | HM3D v1 | HM3D v2 | OVON Seen | OVON Syn. | OVON Unseen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VLFM | 36.4 | 52.5 | 63.6 | 35.2 | 32.4 | 35.2 |
| SG-Nav | 40.2 | 54.0 | 49.6 | — | — | — |
| CogNav | 46.6 | 72.5 | — | — | — | — |
| Uni-NaVid | — | 73.7 | — | 41.3 | 43.9 | 39.5 |
| MTU3D | — | — | — | 55.0 | 45.0 | 40.8 |
| **LightNav-0** | **53.3** | **74.5** | **77.2** | **55.3** | **54.6** | **47.0** |

### Embodied Visual Tracking (EVT-Bench)

STT is single-target tracking; DT adds distractors that look like the target.

| Model | STT SR (%) | STT TR (%) | DT SR (%) | DT TR (%) |
| :--- | :---: | :---: | :---: | :---: |
| Uni-NaVid | 53.3 | 67.2 | 31.9 | 50.1 |
| TrackVLA | 85.1 | 78.6 | 57.6 | 63.2 |
| VLingNav | 88.4 | 81.2 | 67.6 | 73.5 |
| ReferTrack | 89.4 | **92.5** | 73.3 | **81.8** |
| **LightNav-0** | **91.7** | 87.7 | **82.6** | 80.1 |

On DT, LightNav-0 also passes every panoramic and multi-camera system in the paper, including
CoMaTrack at 74.2 SR.

### INSIGHT-Bench

Our deployment-oriented benchmark: 1,097 episodes across 210 indoor and outdoor scenes, with
every policy driven through one shared 120° forward RGB interface and a 300-action budget.

| Model | SR (%) | SPL (%) | NE (m) |
| :--- | :---: | :---: | :---: |
| StreamVLN | 11.6 | 10.8 | 6.56 |
| Uni-NaVid | 24.3 | 22.1 | 4.91 |
| NaVid | 26.9 | 23.0 | 4.25 |
| JanusVLN | 27.4 | 24.0 | 4.89 |
| **LightNav-0** | **43.7** | **41.5** | **3.88** |

Episodes and evaluation code are released separately at
[lightorigins/light-insight-bench](https://github.com/lightorigins/light-insight-bench/tree/main);
this repository ships the VLN-CE / ObjectNav and EVT-Bench harnesses.

### Scaling Analysis

How R2R and RxR val-unseen respond to backbone size, training-data volume and training-environment
coverage.

<div align="center">
  <img src="docs/assets/scaling.png" alt="Three line charts on continuous VLN: success rate and SPL against backbone size, fraction of training data, and fraction of training environments, for R2R and RxR" width="95%"/>
</div>

<br>

Three different behaviours. **Model scaling** saturates: 2B → 4B lifts R2R SR/SPL by 8.6/7.4
points, but 8B is mixed and mostly slightly worse. **Data scaling** is monotonic yet
diminishing — the last doubling, from half the corpus to all of it, buys only 0.8 R2R SR.
**Environment scaling** is the one axis that keeps paying: going from 1/8 of the training
environments to all of them adds 16.7/16.2 points on R2R and 21.1/19.1 on RxR, ahead of what
data scaling delivers over the matched range. Scene diversity, not parameters or sheer hours,
is the reliable lever.

<details>
<summary><b>Embodied reasoning (LightNav-ER)</b></summary>

<br>

The Stage-I embodied-reasoning checkpoint used to initialise LightNav-0, evaluated before any
navigation alignment. A 4B model that outscores an 8B spatially-specialised one on the
complete-set average.

| Model | Params | Point-Bench | RefSpatial | RoboSpatial POI | RoboSpatial VQA | Where2Place | CV-Bench | ERQA | EmbSpatial | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL | 4B | 58.2 | 45.5 | **64.8** | 69.7 | 64.0 | 85.6 | 39.5 | 77.6 | 63.1 |
| Qwen3.5-4B | 4B | 60.4 | 54.6 | 47.9 | 59.7 | 61.3 | 85.0 | 40.8 | 76.8 | 60.8 |
| Molmo2-ER | 8B | **77.3** | 52.5 | 32.0 | **73.4** | 54.0 | 87.8 | **46.8** | 78.8 | 62.8 |
| **LightNav-ER** | 4B | 64.5 | **57.4** | 56.5 | 71.9 | **76.6** | **88.4** | 43.8 | **79.8** | **67.4** |

</details>

## ⚡ Quick Start

```bash
git clone https://github.com/lightorigins/LightNav-0.git && cd LightNav-0
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,video]"
hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0
```

Predict on a video clip — a released checkpoint ships its own action decoder, so
`--model_path` is the only asset argument needed:

```bash
lightnav-predict --model_path checkpoints/LightNav-0 \
    --backend vllm_local --video clip.mp4 --fps 4 \
    --instruction "follow the person in the red shirt"
```

Or serve it and stream frames over WebSocket:

```bash
PORT=8050 lightnav-serve --task tracking --model_path checkpoints/LightNav-0 --backend vllm_local
lightnav-ws-client --server ws://localhost:8050 --video clip.mp4 --fps 4 \
    --instruction "follow the person in the red shirt"
```

Habitat evaluation, EVT-Bench, the Python API, Docker and the Blackwell `sm_103` workaround:
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

## 🕹️ Try It in Simulation

[`mujoco_demo/`](mujoco_demo/) is a self-contained MuJoCo TurtleBot in a bundled ProcTHOR
scene — no ROS, no Habitat, no GPU on the client side:

```bash
cd mujoco_demo && ./run.sh        # needs uv; then open http://127.0.0.1:8088
```

Point the web console at your `lightnav-serve` address and type an instruction; it drives with
the same MPC and client protocol as the real robots in
[`robot_deploy/`](robot_deploy/README.md):

![MuJoCo demo: the simulated robot navigates to the trashcan from a language instruction](docs/assets/mujoco_demo.gif)

The same runtime also carries an optional **MicroDuck** biped: Pollen Robotics' MJCF and ONNX
walking policy sit under the same MPC, so LightNav's waypoints become gait commands. The two
external files to fetch are listed in
[`mujoco_demo/README.md`](mujoco_demo/README.md#microduck-optional):

![MuJoCo demo at 2x speed: the MicroDuck biped follows "turn around, go to the TV" from its head camera, with LightNav's pointing markers and waypoint trail overlaid and a third-person picture-in-picture](docs/assets/mujoco_microduck.gif)

## 🤖 Real-Robot Deployment

The model runs on a GPU host behind `lightnav-serve`; the robot runs a thin WebSocket client
(any language) that streams JPEG frames plus the instruction and executes the first returned
waypoint each control period. Several robots can share one server — sessions are
micro-batched.

Don't want to write the robot side yourself? [`robot_deploy/`](robot_deploy/) is a complete
ROS 2 on-robot stack — camera driver, WebSocket client, MPC waypoint tracker and a web control
panel — with adapters for the Unitree Go2 and LimX TRON 1, and a
[bring-your-own-robot](robot_deploy/README.md#bring-your-own-robot) adapter interface.

The client loop, velocity mapping and wire protocol are in
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#real-robot-deployment),
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) and [docs/PROTOCOL.md](docs/PROTOCOL.md).

## ✍️ Prompt Guide

What makes a good navigation instruction: one **action verb** (`Go to` / `Walk to` / `Head to`
/ `Walk towards` / `Approach` — any works), an optional **direction**, an **unambiguous object
phrase**, and an optional `and stop`:

```
[action verb] + [direction (optional)] + [disambiguated object phrase] + [and stop (optional)]
```

"Disambiguated" means there is no doubt *which* object is meant. Pick **one** of the four
strategies below per instruction — don't stack them. Ranked by reliability (every example is a
verified real instruction):

**① Direction + object — most reliable, use first.**

```
Turn left and walk to the red lamppost
Go to the front-left TV
Go to the desk on your right and stop.
Turn right, then walk to the chair and stop.
```

The direction may precede the action (`Turn left and go to X`) or follow the object
(`the desk on your right`) — both work. Indoors prefer `front-left` / `front-right` /
`in front`; outdoors prefer `turn left` / `turn right`. Directions are relative to the
**robot**, not the room.

**② Relational anchor (`next to` / `on` / `behind`) — second choice.**

```
Walk towards the trash can next to the green lawn
Walk to the vase on the dining table ahead.
Go to the table behind you
Head to the plant behind you on the right.
```

`A next to B` / `A on B` / `A behind you` all work — `behind you` is especially effective,
since it gives both a direction (turn around) and disambiguation at once. Choose a **large,
salient** anchor B (lawn / trees / dining table / walkway), not another small object.

**③ Extremes (`leftmost` / `nearest`) — usable.**

```
Go to the leftmost TV in front
Walk to the rightmost curtain.
Turn left and walk to the nearest grey pointed stone bollard on the park lawn
```

`leftmost` / `rightmost` outperform `nearest` / `farthest`: the former are directly visible,
the latter require depth estimation.

**④ Ordinals (`first` / `second` / `third`) — weakest, use sparingly.**

```
Walk to the first wooden park bench on the right
Turn left and walk to the second stone bench from the left.
Go to the first chair on the right side of the dining table
```

An ordinal **must** come with a counting direction (`from the left` / `on the right`),
otherwise where to start counting is ambiguous. Avoid anything beyond `third`; to single out
one object, prefer an extreme (`leftmost`) or a relation (`next to the door`) over an ordinal.

## 🔗 Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{lightnav0,
  title  = {LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation},
  author = {Light Origins Team},
  year   = {2026},
  eprint = {2608.30935},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2608.30935}
}
```

## 🙏 Acknowledgements

Built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [vLLM](https://github.com/vllm-project/vllm),
[Habitat](https://github.com/facebookresearch/habitat-lab), [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
and [EVT-Bench / TrackVLA](https://github.com/wsakobe/TrackVLA). Third-party code and licences are
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 📄 License

This project is released under the [Apache License 2.0](LICENSE). EVT-Bench itself is
CC BY-NC-SA 4.0 and is not redistributed here.

<a id="community"></a>

## 💬 Community

Questions, deployment notes and release news — join us on
[Discord](https://discord.gg/zwZuD9JG), or scan to join the WeChat group:

<div align="center">
  <img src="docs/assets/wechat_group.png" alt="WeChat QR code for the LightOrigins discussion group" width="280"/>
</div>
