# Serving LightNav-0 on NVIDIA Jetson AGX Thor

[中文版](JETSON_THOR.zh.md)

Field notes from running `lightnav-serve` **onboard** a robot's own Jetson AGX Thor
(SM 11.0, unified memory) instead of a remote GPU host. Everything below was measured
while driving LimX humanoids in daily testing; the robot-side client on the same board
connects to `ws://127.0.0.1:8050`, so a Wi-Fi drop can never stall control.

Companion launcher: [`scripts/serve_thor.sh`](../scripts/serve_thor.sh).

## Install (aarch64)

* Use the NVIDIA Jetson wheel channel for the CUDA stack — plain PyPI has no
  aarch64+CUDA builds. We deploy, for aarch64: `torch==2.10.0+cu130`, `torchvision==0.25.0+cu130`,
  `torchaudio==2.10.0+cu130`, and `vllm==0.19.1` (the fp8 patch below binds vLLM's
  private quant API, so pin the minor — a patch bump may move it).
* cu13 torch wheels need their bundled NVIDIA libs ahead of any system copy, or
  `import torch` can die on an undefined nvJitLink symbol:

  ```bash
  # a colon-joined LD_LIBRARY_PATH cannot hold an unexpanded glob — build it dir by dir
  for d in "$VENV"/lib/python*/site-packages/nvidia/*/lib; do
      [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  done
  ```

  (`serve_thor.sh` does this automatically.)
* Air-gapped robots: pre-download wheels on a workstation into a wheelhouse and install
  with `pip install --no-index --find-links`. Also export `VLLM_NO_USAGE_STATS=1
  DO_NOT_TRACK=1` — vLLM posts telemetry on engine start and the outbound attempt is a
  startup stall when there is no route out.

## fp8 serving (`--quantization fp8_llm_only`)

The engine loads bf16 by default; `--quantization fp8_llm_only` (env `VLLM_QUANT`) quantizes the **LLM half
only** to fp8 on the fly and keeps every `visual.*` linear in bf16
(`lightnav.inference.vllm_utils._patch_fp8_llm_only`). Measured on an AGX Thor,
single session, server-side per-step medians over the same greedy replay:

| Mode | per step | rate |
|---|---|---|
| bf16 (default) | 268 ms | 3.7 Hz |
| `fp8_llm_only` | **178 ms** | **5.6 Hz** |

`serve_thor.sh` turns it on automatically when a Thor is detected
(`VLLM_QUANT= bash scripts/serve_thor.sh` reverts to bf16). Full-model `fp8` is refused by the loader: quantizing the ViT both corrupts perception (on a
348-step tracking replay: 91 visible-flag flips, 121 stop/go flips) and *slows*
the ViT ~2×, since dynamic-quant overhead dominates its small eager-mode GEMMs.
LLM-only measured near-parity with bf16 (97.4% stop agreement, waypoint
displacement p50 4.4 cm).

### SM 11.0 kernel pitfalls

fp8 on Thor needs one guard (set automatically by `serve_thor.sh`): three vLLM
linear-layer kernels are compiled for the SM 10.0 family only and **kill the CUDA
context with a device-side trap** on SM 11.0:

```bash
export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel
```

With those disabled, vLLM falls back to the per-tensor `torch._scaled_mm` (cuBLASLt)
route, which is correct and still delivers the speedup above.

Poisoned compile cache: if the crash message `This kernel only supports sm100f` appears
*after* applying the workaround, a `torch.compile` graph traced before the fix is being
replayed — clear `~/.cache/vllm/torch_compile_cache` and start again.

## Unified-memory quirks

* `nvidia-smi --query-gpu=memory.used` reports `[N/A]` on Jetson — restart scripts that
  wait for VRAM to drain must fall back to a process check (`serve_thor.sh --stop` does).
* GPU and CPU share one LPDDR pool. Budget `--gpu_memory_utilization` against what the
  *rest of the system* needs, not against a fixed VRAM size; we run the navigation model
  at `0.60` and leave the remainder for the OS, the robot stack and optional co-located
  services (below).
* A recording-enabled server flushes episode segments before exiting, so memory is
  released well after the process is signalled — start the next server only after the
  old process is really gone or the engine fails its free-memory check.

## ViT tubelet cache sizing

SlowFast history re-encodes recurring tubelets every step; the LRU in
`lightnav.inference.vit_cache` makes those hits (~34 ms vs ~51 ms per step when cold on
Thor). The default capacity (512 with SlowFast) covers roughly two minutes of session at
4 Hz — beyond that, the mid/span tiers' recurring tubelets are evicted before reuse and
misses climb from ~1.0 to ~3.4 per step. For long robot episodes we run 1024 entries
(steady through ~5 min sessions with the engine budgeted at `gpu_memory_utilization
0.60`; on the same budget 2048 did **not** fit alongside the engine). The launcher
leaves the engine default alone — opt in per run:

```bash
export VLN_VIT_CACHE_ENTRIES=1024
```

A non-integer value exits at startup with an error; values are floored to 32.

## Co-locating other GPU services (optional)

Smaller models fit on the same board next to navigation. We ran a
Chinese-speech → English-instruction pipeline (Qwen3-ASR-0.6B at
`gpu_memory_utilization 0.10` + a Qwen2.5-3B translator at `0.15`, leaving `0.60`
for navigation; ≈0.6 s per utterance while navigation held full rate). The one rule
that matters: **start the small models first**, so the big engine's free-memory
check sees a settled pool.

## Measured throughput

640×360 input, 64-frame SlowFast history, single session, AGX Thor. Server-side
rows are a 12-frame greedy replay with this repo's code launched with `scripts/serve_thor.sh` (2026-09);
the closed-loop row is a months-long production deployment of the same
engine+quantization on LimX humanoids:

| Configuration | per-step median | rate |
|---|---|---|
| bf16, server-side | 268 ms | 3.7 Hz |
| `fp8_llm_only`, server-side | **178 ms** | **5.6 Hz** |
| `fp8_llm_only`, full closed loop (camera → client → server → MPC) | — | 4.5–4.8 Hz |

Per-episode closed-loop rate distribution across 186 field episodes (including a
before/after of a USB-2 camera incident):

![Closed-loop inference-rate distribution across 186 field episodes](assets/closed_loop_rate.png)

The closed-loop rate is what the robot actually gets with the client on the same board;
at ~4.5 Hz the stack tracked people and reached objects at up to 1.5 m/s. One practical
warning from the field: an **undetected camera downgrade dominates everything** — a
USB-2 link silently drops an Orbbec's default profile to 10 fps, halving inference rate
and destabilising turns. Verify the camera enumerates at 5000M (`lsusb -t`) before
debugging the model.
