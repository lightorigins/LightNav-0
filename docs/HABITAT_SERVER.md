# Habitat environment server

`habitat_server/` is a small, self-contained Python 3.9 package (`lightnav_habitat`) that runs
inside a habitat-sim / habitat-lab conda environment and exposes **one** Habitat benchmark
environment over ZeroMQ. The model side (`lightnav-eval-habitat`, see
[EVAL_HABITAT.md](EVAL_HABITAT.md)) runs in the `lightnav` environment, connects to the
server, and drives it with `velocity_control` commands. Neither process imports the other's
dependencies: the server never imports torch or `lightnav`, the client never imports
habitat.

Supported benchmarks:

| `--task`    | config                              | split (yaml) | episodes | success radius |
|-------------|-------------------------------------|--------------|----------|----------------|
| `vlnce`     | `configs/vlnce_r2r.yaml`            | val_unseen   | 1,839    | 3.0 m          |
| `vlnce`     | `configs/vlnce_rxr.yaml`            | val_unseen   | 3,669 (en-US + en-IN) | 3.0 m          |
| `objectnav` | `configs/objectnav_hm3d_v1.yaml`    | val          | 2,000    | 0.1 m (to a viewpoint) |
| `objectnav` | `configs/objectnav_hm3d_v2.yaml`    | val          | 1,000    | 0.1 m; needs `--navmesh-cell-height 0.05` |
| `objectnav` | `configs/objectnav_mp3d.yaml`       | val          | 2,195    | 0.1 m (to a viewpoint) |
| `objectnav` | `configs/objectnav_ovon.yaml`       | val_seen / val_seen_synonyms / val_unseen | 3,000 each | 0.25 m (`--success-distance 0.25`) |

All six configs render 480x270 RGB at 120 deg horizontal FOV from a camera 0.88 m above
the floor, with `allow_sliding: true` and a 500-step episode limit.

## 1. Install

The only combination we have verified is **python 3.9 + habitat-sim 0.3.1 (headless,
bullet) + habitat-lab 0.3.20231024 + numpy < 1.24**. Newer numpy breaks
`numpy-quaternion`, and habitat-lab's own requirements would pull numpy 2.x, so habitat-lab
is installed with `--no-deps` and numpy is pinned last.

```bash
# environment.yml pulls python/cmake from the `defaults` channel; recent conda versions
# require accepting its terms once: conda tos accept --override-channels \
#   --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r
conda env create -f habitat_server/environment.yml
conda activate habitat
pip install --no-deps "habitat-lab==0.3.20231024"
pip install --force-reinstall "numpy>=1.20,<1.24"
pip install -e habitat_server            # pyzmq, pyyaml, fastdtw, numpy<1.24

python -c "import habitat, habitat_sim; import lightnav_habitat.serve; print('ok')"
```

`fastdtw` (or `dtw-python`) is required for the VLN-CE NDTW measure.

### Headless rendering (EGL)

habitat-sim renders through EGL without a display. On a machine or container with NVIDIA
drivers the following usually has to be in place:

```bash
export NVIDIA_DRIVER_CAPABILITIES=all
# Point glvnd at the NVIDIA EGL vendor library (create the file if it does not exist):
#   /usr/share/glvnd/egl_vendor.d/10_nvidia.json
#   {"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0   # if habitat-sim fails to create a GL context
```

System packages typically needed: `libgl1 libglx-mesa0 libegl1 libopengl0 libglvnd0
libglib2.0-0 libsm6 libxext6 libxrender1`. (On Ubuntu 24.04 the older names
`libgl1-mesa-glx` and `libegl1-mesa` no longer resolve — the first four above replace them.
Without `libopengl0`, `import habitat_sim` fails with
`libOpenGL.so.0: cannot open shared object file`, which looks like a habitat-sim build
problem but is a missing apt package.)

### GPU selection

The render device is taken from `HABITAT_SIM_GPU_ID` (default `0`) and written to
`habitat.simulator.habitat_sim_v0.gpu_device_id`. **Do not** restrict
`CUDA_VISIBLE_DEVICES` to the same index: habitat-sim matches the CUDA and EGL devices by
UUID and needs to see every GPU. If you want strict isolation, set
`CUDA_VISIBLE_DEVICES=<gpu>` together with `HABITAT_SIM_GPU_ID=0`.

## 2. Data layout

The shipped yamls use example paths relative to the working directory. Either edit them or
pass `--data-path` / `--scenes-dir` (both override the yaml).

```
data/
  scene_datasets/
    mp3d/<scene>/<scene>.glb                       # VLN-CE (Matterport3D)
    hm3d/val/<id>/<id>.basis.glb                   # ObjectNav (HM3D v0.2 val)
    mp3d/<id>/<id>.glb                             # ObjectNav (MP3D)
  datasets/
    R2R_VLNCE_v1-3_preprocessed/val_unseen/
      val_unseen.json.gz                           # episodes
      val_unseen_gt.json.gz                        # NDTW ground truth (same dir, _gt suffix)
    RxR_VLNCE_v0/val_unseen/
      val_unseen_guide.json.gz
      val_unseen_guide_gt.json.gz
    objectnav/hm3d/v1/val/
      val.json.gz                                  # stub (category maps, no episodes)
      content/<scene>.json.gz                      # one shard per scene
    objectnav/hm3d/v2/val/
      val.json.gz                                  # v2 stub; run with --navmesh-cell-height 0.05
      content/<scene>.json.gz
    objectnav/mp3d/v1/val/
      val.json.gz                                  # 21-category stub
      content/<scene>.json.gz
    ovon/hm3d/val_unseen/
      val_unseen.json.gz                           # stub (may have empty category maps)
      content/<scene>.json.gz
    ovon/hm3d/val_seen*/                           # val_seen / val_seen_synonyms: same layout
```

Notes:

* Episode files reference scenes as `data/scene_datasets/...`; the loaders strip that prefix
  and join the remainder with `scenes_dir`.
* `--data-path ROOT` rewrites `dataset.data_path` to `ROOT/{split}/{split}.json.gz`. It cannot
  express RxR's `{split}_guide.json.gz`; edit `vlnce_rxr.yaml` for RxR instead.
* `--task vlnce` always writes `dataset.split` (default `val_unseen`, the benchmark split);
  `--task objectnav` keeps the yaml split (`val` for HM3D / MP3D v1, `val_unseen` for OVON) unless
  `--split` is given.
* ObjectNav splits need the `<split>/<split>.json.gz` stub next to `content/` (habitat loads the
  stub, then every shard). HM3D-OVON episodes carry fields habitat-lab 0.3.x rejects and ship
  empty category maps; `lightnav_habitat.objectnav_extensions` re-registers a tolerant
  `ObjectNav-v1` loader that handles both (and is behaviour-identical for HM3D / MP3D v1).
* RxR: `vlnce_rxr.yaml` has no `languages` entry, so `VLNCEEnv` injects
  `habitat.dataset.languages = [en-US, en-IN]` (English-only, 3,669 episodes). Add a
  `languages:` list under `habitat.dataset` to change it; the client can additionally
  filter with `--languages`.

## 3. Running the server

```bash
conda activate habitat
cd /path/to/LightNav-0

# VLN-CE R2R val_unseen
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve --task vlnce \
    --config habitat_server/configs/vlnce_r2r.yaml --port 5555 --ready-file /tmp/hab5555.ready

# VLN-CE RxR val_unseen (English guide annotations)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve --task vlnce \
    --config habitat_server/configs/vlnce_rxr.yaml --port 5555

# HM3D ObjectNav v1 val (success 0.1 m = env default)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve --task objectnav \
    --config habitat_server/configs/objectnav_hm3d_v1.yaml --port 5555

# HM3D ObjectNav v2 val: ALWAYS re-bake the navmesh (v2 episodes were generated at
# cell_height=0.05; the shipped .basis.navmesh is 0.20 and floors distance_to_goal)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve --task objectnav \
    --config habitat_server/configs/objectnav_hm3d_v2.yaml --navmesh-cell-height 0.05 --port 5555

# MP3D ObjectNav v1 val (success 0.1 m = env default)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve --task objectnav \
    --config habitat_server/configs/objectnav_mp3d.yaml --port 5555

# HM3D-OVON val_unseen (success 0.25 m)
HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve --task objectnav \
    --config habitat_server/configs/objectnav_ovon.yaml --split val_unseen \
    --success-distance 0.25 --port 5555
```

`pip install -e habitat_server` also installs the `lightnav-habitat-serve` console script
(same flags). The server binds `tcp://*:PORT`, builds the simulator, touches `--ready-file`
(if given) and then blocks until it receives a `close` command or SIGINT/SIGTERM. Start the
client only after the ready file exists; the first scene load can take tens of seconds.

### Flags

| flag | default | meaning |
|------|---------|---------|
| `--task {vlnce,objectnav}` | `vlnce` | environment class |
| `--config PATH` | required | Habitat Hydra yaml |
| `--port N` | 5555 | ZMQ port |
| `--max-steps N` | 500 | report `truncated` after N steps (keep <= yaml `max_episode_steps`) |
| `--image-height H --image-width W` | yaml | override the RGB/depth sensor size (both or neither) |
| `--split-id I --split-num N` | none | serve shard I of N (see below) |
| `--early-stop-rotation N` | 0 | force STOP after more than N consecutive steps with unchanged `distance_to_goal` (0 = off) |
| `--early-stop-steps N` | 0 | force STOP after more than N steps (0 = off) |
| `--split NAME` | `val_unseen` (vlnce) / yaml (objectnav) | dataset split written to `habitat.dataset.split` |
| `--data-path ROOT` | yaml | `ROOT/{split}/{split}.json.gz` |
| `--scenes-dir DIR` | yaml | scene datasets directory |
| `--success-distance M` | 3.0 / 0.1 | success radius (VLN-CE / ObjectNav); use 0.25 for OVON |
| `--ready-file PATH` | none | touched when the simulator is up |

The published numbers were produced with early stop **disabled** (both flags 0): episodes end
only when the policy stops (a zero-velocity command) or at `--max-steps 500`.

### Image size

The yaml sensor size (480x270) is what the published numbers were rendered at. Pass
`--image-height/--image-width` only if you deliberately want a different resolution; the
policy resizes frames to its own input size anyway, but a different render aspect ratio
changes what the model sees.

## 4. Parallel evaluation

One server serves one environment. To use several GPUs, start one server per GPU with a
disjoint shard and its own port, and one client per server:

```bash
NUM=4
mkdir -p logs
for i in $(seq 0 $((NUM - 1))); do
  HABITAT_SIM_GPU_ID=$i python -m lightnav_habitat.serve --task vlnce \
      --config habitat_server/configs/vlnce_r2r.yaml \
      --port $((5555 + i)) --split-id $i --split-num $NUM \
      --ready-file /tmp/hab$((5555 + i)).ready > logs/habitat_$i.log 2>&1 &
done
```

Sharding sorts the episode list by `scene_id`, cuts it into `split-num` contiguous chunks
(the last chunk takes the remainder) and serves chunk `split-id`. Sorting by scene keeps
scene reloads to a minimum. The episode iterator is deterministic (`shuffle=False`) and
cycles forever; the client detects the wrap-around by watching for a repeated
`(scene_id, episode_id)` pair. Concatenate the clients' `results.jsonl` files to aggregate.

## 5. Protocol summary

Requests and responses are pickled dicts (protocol 4) over a ZMQ REQ/REP pair:

```
{"command": "reset", "data": {"seed": null, "options": null}}
    -> {"status": "success", "obs": {...}, "info": {...}}
{"command": "step", "data": <action>}
    -> {"status": "success", "obs", "reward", "terminated", "truncated", "info"}
{"command": "close"}  -> {"status": "success"}     (server exits)
any failure           -> {"status": "error", "message": "..."}
```

Actions are either an `int` in 0..3 (STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT) or

```python
{"action": "velocity_control",
 "action_args": {"linear_velocity": v_lin, "angular_velocity": v_ang}}   # both in [-1, 1]
```

Habitat de-normalizes `v` as `min + (v + 1) / 2 * (max - min)` over `lin_vel_range` /
`ang_vel_range` and integrates for `time_step` seconds. A command whose de-normalized speeds
are both below `min_abs_lin_speed` / `min_abs_ang_speed` is a STOP (it sets
`is_stop_called`, which the Success measure requires). VLN-CE uses `lin [0, 2.5] m/s`,
`ang [-300, 300] deg/s`, `dt 0.1 s`; ObjectNav uses `lin [0, 0.25]`, `ang [-30, 30]`, `dt 1 s`
(same 0.25 m / 30 deg per-step caps). The client must read these from `info` rather than
hard-code them.

`obs`: `rgb` uint8 (H, W, 3), `depth` float32 (H, W), `instruction` `{"text": str}`,
`goal_distance` float32 (1,), `progress` float32 (1,).

`info` (every step): `steps`, `episode_id` (str), `scene_id`, Habitat metrics
(`distance_to_goal`, `success`, `spl`, `path_length`, `oracle_success`, `steps_taken`, plus
`ndtw` for VLN-CE and `soft_spl` for ObjectNav), `instruction`, `habitat_time_step`,
`lin_vel_range`, `ang_vel_range`, `goal_distance`, `goal_position`; VLN-CE adds
`reference_path` and (RxR) `language`; ObjectNav adds `object_category`, `goal_positions`
and `raw_episode_id`. On the final step `termination_reason` is one of `agent_stop`,
`early_stop_no_progress`, `early_stop_step_limit`, `max_steps_truncated`, `unknown`, with a
free-form `termination_details` dict.

## 6. Troubleshooting

* `AttributeError: _ARRAY_API not found` / quaternion import errors: numpy got upgraded past
  1.23. Re-run `pip install --force-reinstall "numpy>=1.20,<1.24"`.
* `ImportError: NDTW measure requires either 'fastdtw' or 'dtw-python'`: `pip install fastdtw`.
* `KeyError` from NDTW at reset: the `_gt.json.gz` file for the split is missing or does not
  match the episode file.
* `AssertionError` inside habitat on `step`: the client stepped past `max_episode_steps`;
  keep `--max-steps` <= the yaml value (both default to 500).
* Simulator hangs at startup with several servers on one GPU: start them staggered.
