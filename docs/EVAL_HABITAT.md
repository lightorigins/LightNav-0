# Habitat evaluation: VLN-CE (R2R / RxR) and ObjectNav (HM3D v1/v2 / MP3D / HM3D-OVON)

Evaluation is split across two processes:

* **Habitat env server** (`habitat_server/`, its own conda env, no torch): renders the
  simulator, exposes one environment over a ZMQ REP socket and computes the metrics.
  See [HABITAT_SERVER.md](HABITAT_SERVER.md) for the conda recipe, datasets and CLI.
* **Eval client** (`lightnav-eval-habitat`, the `lightnav` env, GPU): loads the
  checkpoint, connects to the server, runs episodes and writes the summary.

## Benchmarks

| Task | Server config | Split | Episodes | Camera | Success radius |
|---|---|---|---|---|---|
| R2R (VLN-CE) | `habitat_server/configs/vlnce_r2r.yaml` | `val_unseen` | 1,839 | 480x270, hfov 120, height 0.88 m | 3.0 m |
| RxR (VLN-CE) | `habitat_server/configs/vlnce_rxr.yaml` | `val_unseen` | 3,669 with `--languages en-US en-IN` (11,006 total) | same | 3.0 m |
| HM3D ObjectNav v1 | `habitat_server/configs/objectnav_hm3d_v1.yaml` | `val` | 2,000 (20 scenes x 6 categories) | same | 0.1 m (to a viewpoint) |
| HM3D ObjectNav v2 | `habitat_server/configs/objectnav_hm3d_v2.yaml` | `val` | 1,000 (HM3D v0.2 scenes) | same | 0.1 m; **needs `--navmesh-cell-height 0.05`** |
| MP3D ObjectNav v1 | `habitat_server/configs/objectnav_mp3d.yaml` | `val` | 2,195 (21 categories) | same | 0.1 m (to a viewpoint) |
| HM3D-OVON | `habitat_server/configs/objectnav_ovon.yaml` | `val_seen`, `val_seen_synonyms`, `val_unseen` | 3,000 each (36 scenes, open-vocabulary) | same | 0.25 m |

RxR language splits: hi-IN 3,669 / te-IN 3,668 / en-IN 2,446 / en-US 1,223 episodes.
English numbers are reported on `--languages en-US en-IN`.

All tasks use the `velocity_control` action with a 500-step cap. Decoding is greedy.

## 1. Start the env server

In the habitat conda env (one server per GPU / port):

```bash
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve \
    --task vlnce --config habitat_server/configs/vlnce_r2r.yaml \
    --split val_unseen --port 5555 --ready-file /tmp/habitat_5555.ready
```

ObjectNav variants:

```bash
# HM3D ObjectNav v1 (success distance defaults to 0.1 m)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve \
    --task objectnav --config habitat_server/configs/objectnav_hm3d_v1.yaml \
    --split val --port 5555

# HM3D ObjectNav v2: the re-baked navmesh is REQUIRED (see "Navmesh alignment" below)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve \
    --task objectnav --config habitat_server/configs/objectnav_hm3d_v2.yaml \
    --split val --navmesh-cell-height 0.05 --port 5555

# MP3D ObjectNav v1 (same defaults as HM3D v1)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve \
    --task objectnav --config habitat_server/configs/objectnav_mp3d.yaml \
    --split val --port 5555

# HM3D-OVON: success distance 0.25 m; splits: val_seen, val_seen_synonyms, val_unseen
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve \
    --task objectnav --config habitat_server/configs/objectnav_ovon.yaml \
    --split val_unseen --success-distance 0.25 --port 5555
```

Notes:

* `HABITAT_SIM_GPU_ID` selects the render GPU. Do **not** derive `CUDA_VISIBLE_DEVICES`
  from it; habitat-sim matches the CUDA and EGL devices by UUID and needs to see all GPUs.
* The server touches `--ready-file` only after the simulator is up; wait for it before
  starting the client.
* Sensor resolution comes from the yaml (480x270). Only pass `--image-height/--image-width`
  if you deliberately want to change it; the reference numbers use the yaml values.

## 2. Run the client

In the `lightnav` env:

```bash
lightnav-eval-habitat \
    --model_path /path/to/checkpoint \
    --server tcp://localhost:5555 \
    --backend vllm_local \
    --episodes -1 \
    --output_dir output/r2r
```

Add `--languages en-US en-IN` for RxR. `--episodes -1` (default) runs the whole split:
the client stops when the server's episode iterator cycles back to an already seen
`(scene_id, episode_id)`. `--episodes N` stops after `N` admitted episodes.

### Client flags

| flag | default | meaning |
|------|---------|---------|
| `--model_path DIR` | required | checkpoint directory (its `eval_config.json` supplies the processing params) |
| `--server ADDR` | `tcp://localhost:5555` | env server ZMQ address |
| `--backend {hf,vllm_local}` | `vllm_local` | in-process vLLM (recommended) or plain transformers |
| `--episodes N` | `-1` | `<= 0` = full split (stop at the first repeated episode key), else stop after N admitted episodes |
| `--max_steps N` | `500` | client-side per-episode step cap (the server truncates at its own `--max-steps`) |
| `--no_force_stop` | off | by default the LAST action of the budget is an explicit STOP (zero velocity), so an episode that runs out of steps ends by `agent_stop` and Success / SPL are measured at the final pose; pass this to let the env truncate instead (`success=0` for every timed-out episode). Recorded per episode as `forced_stop` |
| `--output_dir DIR` | `output/habitat_eval` | where `results.jsonl` and `summary.json` go |
| `--languages L [L ...]` | none | RxR only: keep episodes whose `info["language"]` is listed; skipped episodes are not counted |
| `--action_tokenizer_bundle DIR` | env `ACTION_TOKENIZER_BUNDLE` | RVQ decoder (see below) |
| `--gpu_memory_utilization F` | `0.65` | vLLM GPU memory fraction |
| `--max_num_seqs N` | `1` | vLLM batch width (one env per client, so 1) |
| `--num_history_frames N` | from `eval_config.json` | override the history window (normally leave unset) |
| `--max_new_tokens N` | `64` | per-step decode cap; raised automatically to grounding prefix + action tokens when smaller |
| `--zmq_timeout_ms N` | `600000` | receive timeout per attempt (Habitat scene loads are slow) |
| `--verbose` | off | one line per step with the raw model text and the chosen waypoint |
| `--save_video` | off | write `<output_dir>/videos/<episode_id>.mp4` per episode (predicted trajectory, pointing markers, HUD over the agent's frames; one frame per step). Needs `pip install -e ".[video]"` |
| `--video_fps N` | `10` | playback frame rate of the saved videos |
| `--hfov_deg F` | `120.0` | agent camera horizontal FOV for the trajectory overlay (the shipped yamls) |
| `--cam_height F` | `0.88` | agent camera height in metres for the trajectory overlay (the shipped yamls) |
| `--waypoint_dt_s F` | `0.1` | seconds per waypoint row assumed by the HUD velocity readout |
| `--record_dir DIR` | off | also record the raw episodes (JPEG frames + per-step records) for `lightnav-render` |

Decoding is greedy (temperature 0); the client never touches the sampling knobs. The
visualisation flags are described in [VISUALIZATION.md](VISUALIZATION.md); none of them
changes the evaluation.

### Action decoder

The policy needs to know how to turn the model's trajectory tokens into waypoints.
It is resolved in this order:

1. the explicit flag `--action_tokenizer_bundle <dir>` (RVQ bundle, `<act_l*>` tokens);
2. the `eval_config.json` snapshot next to the checkpoint
   (`tasks.*.action_tokenizer.bundle_path`), when that path exists;
3. the sibling directory `<model_path>/action_tokenizer/` (an RVQ bundle with `manifest.json`).

The released checkpoints ship their bundle, so no decoder flag is needed. A wrong
`--action_tokenizer_bundle` is reported with an explicit message, but note *when*: the
decoder is loaded after the inference engine, so it surfaces only once the weights are up
(tens of seconds), not at argument-parse time.

### Navmesh alignment (HM3D ObjectNav v2)

HM3D ships one `<scene>.basis.navmesh` per scene, baked with the habitat defaults
(`cell_height=0.20`). The ObjectNav **v2** episodes - start positions and goal view points
alike - were generated on a navmesh baked at `cell_height=0.05`, so on the shipped mesh
every v2 view point sits a constant 0.05 / 0.10 / 0.15 m (per scene) *below* the walkable
surface. `geodesic_distance` keeps that vertical residual (`geo = sqrt(h^2 + offset^2)`), so
`distance_to_goal` acquires a hard floor equal to the offset and the official 0.1 m success
radius ends up measuring data alignment rather than the policy.

`--navmesh-cell-height 0.05` (launcher: `NAVMESH_CELL_HEIGHT=0.05`) fixes this by re-baking
each scene's navmesh once at load time through the public `sim.recompute_navmesh` API, using
the agent radius / height from the yaml. The hook wraps `sim.reconfigure` so a scene's first
episode (and its SPL denominator) is already scored on the re-baked mesh; a failed re-bake
logs a warning and falls back to the shipped navmesh. Effect on the mesh itself is marginal
(navigable area within ~2%, start/goal reachability unchanged, ~0.1-0.15 s per scene).

When it is on, the server log shows:

```
[navmesh] rebake hook installed (cell_height=0.05)
[navmesh] rebaked TEEsavR23oF at cell_height=0.05 (r=0.18, h=0.88): ok=True area=62.7m2 in 88ms
```

Where the other benchmarks stand (vertical offset of stored goals vs the loaded navmesh):
HM3D v1, MP3D and R2R are clean (zero offset) and must be run WITHOUT the flag; RxR's
worst offset (0.03 m) is irrelevant at its 3.0 m radius; HM3D-OVON carries the same
0-0.15 m offsets but its 0.25 m radius absorbs them (measured: SR moves < 1.2 pt,
p > 0.3 on all three splits) - leave it off there too, so numbers stay comparable
with prior OVON evaluations.

### Velocity mapping

Each step the policy decodes `(H, 3)` robot-local waypoints `[forward_m, lateral_m, yaw_rad]`,
takes the first row whose forward or yaw component is non-zero, and sends

```
linear_velocity  = clip(2 * (forward_m / dt - lin_min) / (lin_max - lin_min) - 1, -1, 1)
angular_velocity = clip(2 * (deg(yaw_rad) / dt - ang_min) / (ang_max - ang_min) - 1, -1, 1)
```

where `dt`, `lin_*` and `ang_*` are the env's `velocity_control` settings reported by the
server (`habitat_time_step`, `lin_vel_range`, `ang_vel_range`). A zero waypoint (or an
undecodable output) maps to zero speed, which Habitat treats as STOP and ends the episode.

## 3. Parallel evaluation (one shard per GPU)

`scripts/eval_habitat.sh` does the whole thing on the local GPUs: one env server + one
eval client per GPU (the model and the simulator share the GPU), each on a disjoint shard
of the split, then a merge of the shards into one `summary.json`:

```bash
# R2R on every visible GPU
MODEL_PATH=/path/to/checkpoint bash scripts/eval_habitat.sh
# RxR (English) on GPUs 0 and 1
MODEL_PATH=/path/to/checkpoint HABITAT_CONFIG=habitat_server/configs/vlnce_rxr.yaml \
    LANGUAGES="en-US en-IN" GPU_IDS="0 1" bash scripts/eval_habitat.sh
# HM3D ObjectNav v2 (NAVMESH_CELL_HEIGHT is required, see "Navmesh alignment")
MODEL_PATH=/path/to/checkpoint TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_hm3d_v2.yaml \
    SPLIT=val NAVMESH_CELL_HEIGHT=0.05 bash scripts/eval_habitat.sh
# MP3D ObjectNav v1
MODEL_PATH=/path/to/checkpoint TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_mp3d.yaml \
    SPLIT=val bash scripts/eval_habitat.sh
# HM3D-OVON (SPLIT=val_seen / val_seen_synonyms / val_unseen)
MODEL_PATH=/path/to/checkpoint TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_ovon.yaml \
    SPLIT=val_unseen SUCCESS_DISTANCE=0.25 bash scripts/eval_habitat.sh
```

Knobs (env vars, see the script header): `GPU_IDS` / `NUM_GPUS` (default: all GPUs from
`nvidia-smi`), `TASK`, `HABITAT_CONFIG`, `SPLIT`, `SUCCESS_DISTANCE`, `DATA_PATH`,
`SCENES_DIR`, `LANGUAGES`, `EPISODES` (per shard), `BACKEND`, `GPU_MEM_UTIL` (0.65),
`CLIENT_ARGS` (e.g. `"--save_video"`), `HABITAT_CONDA_ENV`/`HABITAT_PYTHON`,
`INFER_VENV`/`CLIENT_PYTHON`, `OUTPUT_ROOT` (default `output/habitat_<task>_<timestamp>`).
Output: `OUTPUT_ROOT/shard_<i>/` per GPU plus the merged `OUTPUT_ROOT/results.jsonl` and
`summary.json`; logs under `OUTPUT_ROOT/logs/`. Servers and clients are killed when the
script exits. Merging can also be run by hand:

```bash
lightnav-eval-merge output/r2r                 # merges output/r2r/*/results.jsonl -> output/r2r/summary.json
lightnav-eval-merge shard_a shard_b --output merged
```

Every metric in `summary.json` is an unweighted per-episode mean, so the merged value is
exactly the summary of the concatenated episodes; `total_time_sec` is the longest shard's
time (wall clock), and `shards` lists the per-shard episode counts.

The manual equivalent (the server shards the split deterministically: episodes sorted by
scene, cut into `split-num` chunks):

```bash
# GPU i of N: one server + one client pair, distinct port and output dir
HABITAT_SIM_GPU_ID=$i python -m lightnav_habitat.serve --task vlnce \
    --config habitat_server/configs/vlnce_r2r.yaml --split val_unseen \
    --split-id $i --split-num $N --port $((5555 + i)) --ready-file /tmp/hab_$i.ready

CUDA_VISIBLE_DEVICES=$i lightnav-eval-habitat --model_path /path/to/checkpoint \
    --server tcp://localhost:$((5555 + i)) --episodes -1 --output_dir output/r2r/shard_$i
```

Run one pair per GPU, then `lightnav-eval-merge <parent-dir>`.

## 4. Output files

`<output_dir>/results.jsonl` gets one JSON line per finished episode (fsync'ed as it is
written, so a killed run keeps what it completed). The file is append-only: use a fresh
`--output_dir` per run, otherwise lines from earlier runs stay in it (`summary.json` only
covers the current run):

```json
{"episode_id": "episode_000", "habitat_episode_id": "1234", "scene_id": "...",
 "rollout_idx": 0, "success": 1.0, "oracle_success": true, "spl": 0.81, "ndtw": 0.77,
 "soft_spl": 0.0, "object_category": "", "steps": 42, "final_distance": 1.9,
 "min_distance": 1.9, "instruction": "Walk past the ...",
 "termination_reason": "agent_stop", "termination_details": {...}}
```

`min_distance` is the minimum `distance_to_goal` over the executed steps. `oracle_success`
uses the server's `oracle_success` metric when present, otherwise `min_distance < 3.0 m`
(VLN-CE) / `< 0.1 m` (ObjectNav).

With `--save_video` each record also carries `"video": "videos/episode_000.mp4"` (relative
to `output_dir`) and `<output_dir>/videos/` holds one mp4 per episode: the frame the policy
acted on at every step with the predicted trajectory ribbon, the pointing markers and a HUD
(instruction, GO/STOP, step, first-waypoint velocities), plus the terminal observation.
`--record_dir DIR` additionally writes the raw episodes (frames + records) under
`DIR/run_<timestamp>/eval/episode_NNN/`, which `lightnav-render DIR` renders with other
settings later. See [VISUALIZATION.md](VISUALIZATION.md).

`<output_dir>/summary.json` is written at the end. VLN-CE (no `object_category`):

```json
{
  "num_episodes": 1839,
  "metrics": {"SR_success_rate_pct": 0.0, "OS_oracle_success_pct": 0.0, "SPL_pct": 0.0,
              "NDTW_pct": 0.0, "NE_navigation_error_m": 0.0},
  "table_format": "SR / OS / SPL / NDTW / NE",
  "avg_steps": 0.0,
  "total_time_sec": 0.0,
  "episodes": [{"episode_id", "habitat_episode_id", "scene_id", "rollout_idx", "success",
                "oracle_success", "spl", "ndtw", "steps", "final_distance", "min_distance",
                "instruction"}],
  "model": "/path/to/checkpoint",
  "backend": "vllm_local"
}
```

ObjectNav (any result has a non-empty `object_category`):

```json
{
  "num_episodes": 2000,
  "metrics": {"SR_success_rate_pct": 0.0, "SPL_pct": 0.0, "SoftSPL_pct": 0.0,
              "NE_navigation_error_m": 0.0},
  "table_format": "SR / SPL / SoftSPL / NE",
  "per_category_sr": {"chair": 0.0, "...": 0.0},
  "avg_steps": 0.0,
  "total_time_sec": 0.0,
  "episodes": [{"episode_id", "habitat_episode_id", "scene_id", "rollout_idx", "success",
                "spl", "soft_spl", "object_category", "steps", "final_distance",
                "min_distance"}],
  "model": "/path/to/checkpoint",
  "backend": "vllm_local"
}
```

Percentages are per-episode means x 100 rounded to 2 decimals; `NE` is the mean final
`distance_to_goal` in metres.

## Python API

```python
from lightnav.habitat import HabitatEvalConfig, run_habitat_eval

results = run_habitat_eval(HabitatEvalConfig(
    model_path="/path/to/checkpoint",
    server="tcp://localhost:5555",
    episodes=-1,
    output_dir="output/r2r",
))
```

`run_habitat_eval` also accepts `env_factory`, `policy_factory` and `engine_factory`
keyword arguments to swap in test doubles.

## Troubleshooting

* `ConnectionError: Timeout waiting for response ... 'reset'`: the server is still loading
  a scene or is not running. The client waits `--zmq_timeout_ms` (600 s) per attempt and
  retries the receive twice without re-sending.
* `KeyError: Habitat env did not expose velocity-control config keys`: the server is not
  the one from `habitat_server/` (it must report `habitat_time_step`, `lin_vel_range`,
  `ang_vel_range` in `info`).
* `FileNotFoundError: Could not resolve the action decoder`: pass
  `--action_tokenizer_bundle` explicitly (see above).
* Every episode ends after one step with `agent_stop`: the model output is not decodable
  with the chosen decoder (wrong bundle); run with `--verbose` to see the
  raw text.
