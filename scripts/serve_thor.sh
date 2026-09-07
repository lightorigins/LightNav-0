#!/usr/bin/env bash
# One-command lightnav-serve launcher for NVIDIA Jetson Thor (SM 11.0).
#
# Wraps the standard `lightnav-serve` entrypoint with the Thor-specific
# environment documented in docs/JETSON_THOR.md: venv NVIDIA libs on
# LD_LIBRARY_PATH, SM 11.0 kernel guard for quantized configs, telemetry off
# for air-gapped robots, capped BLAS/OMP threads, per-port inductor cache,
# and a drain that tolerates Jetson's "[N/A]" memory reporting.
#
# Env knobs reuse scripts/start_servers.sh names where the two overlap
# (MAX_BATCH_SIZE, ACTION_TOKENIZER_BUNDLE, ready files in .servers_ready/).
#
#   bash scripts/serve_thor.sh                    # start, wait for READY
#   TEMPERATURE=0.2 bash scripts/serve_thor.sh    # sampled decoding
#   RECORD=1 bash scripts/serve_thor.sh           # enable episode recording
#   bash scripts/serve_thor.sh --stop             # stop THIS PORT's server
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=${VENV:-$REPO_ROOT/.venv}

# ----------------------------------------------------------------- knobs ----
MODEL_PATH=${MODEL_PATH:-$HOME/models/LightNav-0}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}  # only for ckpts with no decoder
TASK=${TASK:-vln}
PORT=${PORT:-8050}
HOST=${HOST:-0.0.0.0}             # NOTE: no auth of its own -- private subnet only
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-}   # empty -> server default
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}   # unified memory: leave room for the robot stack
TEMPERATURE=${TEMPERATURE:-0.0}
RECORD=${RECORD:-0}
RECORD_DIR=${RECORD_DIR:-$REPO_ROOT/vln_episodes}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
LOG=${LOG:-$LOG_DIR/server_${PORT}.log}
READY_DIR="$REPO_ROOT/.servers_ready"     # same convention as start_servers.sh
READY=${READY:-$READY_DIR/port${PORT}.ready}

# Only THIS port's server. The launched command line contains "--port $PORT",
# so the pattern is port-scoped; the bracket keeps pkill from matching us.
# POSIX ERE (what pgrep/pkill speak) has no \b -- guard the number's end with
# an explicit non-digit-or-EOL group so port 8050 never matches 80500.
PATTERN="lightnav[.]serving[.]ws_server.*--port ${PORT}([^0-9]|\$)"

# SM 11.0 guard: these fp8 kernels are compiled for the SM 10.0 family only and
# trap the CUDA context on Thor. Harmless for the stock bf16 path; required the
# moment a quantized vLLM config is enabled. docs/JETSON_THOR.md has the story.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case "$GPU_NAME" in
    *Thor*)
        export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-MarlinFP8ScaledMMLinearKernel,FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel}
        # fp8 LLM / bf16 ViT: measured 268 -> 178 ms per step (3.7 -> 5.6 Hz)
        # on an AGX Thor with no perception change. `VLLM_QUANT= bash ...`
        # (empty) reverts to bf16. Uses `-` not `:-` so an explicit empty wins.
        export VLLM_QUANT=${VLLM_QUANT-fp8_llm_only}
        echo "[serve_thor] Thor detected: VLLM_QUANT=${VLLM_QUANT:-bf16}  VLLM_DISABLED_KERNELS=$VLLM_DISABLED_KERNELS" >&2 ;;
esac

# ------------------------------------------------------------------ stop ----
drain() {
    pkill -f "$PATTERN" 2>/dev/null || true
    # Jetson iGPUs report memory.used as "[N/A]": fall back to the process
    # check alone -- memory is released when the process dies. A recording
    # server flushes segments first, so wait rather than respawn immediately.
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1) || used=0
        case "$used" in (""|*[!0-9]*) used=0;; esac
        procs=$(pgrep -f "$PATTERN" 2>/dev/null | wc -l) || procs=0
        if [ "${used:-99999}" -lt 500 ] && [ "$procs" -eq 0 ]; then return 0; fi
        sleep 2
    done
    return 1
}

if [ "${1:-}" = "--stop" ]; then
    echo "[serve_thor] stopping port $PORT..."
    drain || echo "[serve_thor] WARN: still draining after timeout" >&2
    rm -f "$READY"
    echo "[serve_thor] stopped."
    exit 0
fi

# ----------------------------------------------------------------- start ----
PY="$VENV/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
[ -d "$MODEL_PATH" ] || { echo "[serve_thor] MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$READY_DIR"
if ! drain; then
    # Never race a still-flushing server for the unified memory pool
    # (docs/JETSON_THOR.md, "Unified-memory quirks").
    echo "[serve_thor] ERROR: previous server did not exit; refusing to start a second one" >&2
    exit 1
fi
rm -f "$READY"

# src layout: `-m lightnav...` must resolve even without `pip install -e .`.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# cu13 torch wheels need their bundled nvidia libs ahead of any system copy.
# Built dir-by-dir: correct when the glob matches nothing and when the repo
# path contains spaces (a colon-joined LD_LIBRARY_PATH cannot hold a literal
# unexpanded glob or a half path).
NV_LIBS=""
for d in "$VENV"/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NV_LIBS="${NV_LIBS:+$NV_LIBS:}$d"
done
if [ -n "$NV_LIBS" ]; then export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"; fi

# Air-gapped robots: don't stall on vLLM's telemetry post.
if [ "${VLLM_USAGE_STATS:-0}" != "1" ]; then
    export VLLM_NO_USAGE_STATS=1
    export DO_NOT_TRACK=1
fi

export VLN_EVAL_TEMPERATURE="$TEMPERATURE"
# VLN_VIT_CACHE_ENTRIES is respected if the caller exported it; the engine's
# own default (512 with SlowFast) is otherwise left alone. See
# docs/JETSON_THOR.md for when 1024 pays off.

# bash 3.2 compatibility under `set -u`: expand arrays with the :+ guard.
QUANT_FLAG=()
[ -n "${VLLM_QUANT:-}" ] && QUANT_FLAG=(--quantization "$VLLM_QUANT")
extra_args=()
[ -n "$ACTION_TOKENIZER_BUNDLE" ] && extra_args+=(--action_tokenizer_bundle "$ACTION_TOKENIZER_BUNDLE" --horizon 10)
[ -n "$MAX_BATCH_SIZE" ] && extra_args+=(--max_batch_size "$MAX_BATCH_SIZE")
[ "$RECORD" = "1" ] && { mkdir -p "$RECORD_DIR"; extra_args+=(--record_dir "$RECORD_DIR"); }

# Small per-step CPU ops are dominated by thread-launch overhead when fanned
# across many cores (see start_servers.sh) -- cap at min(32, nproc).
CORES=$(nproc 2>/dev/null || echo 8)
THREADS=${THREADS:-$(( CORES < 32 ? CORES : 32 ))}

echo "[serve_thor] bind=$HOST:$PORT task=$TASK quant=${VLLM_QUANT:-bf16} gpu_mem_util=$GPU_MEM_UTIL temp=$TEMPERATURE"
echo "[serve_thor] model=$MODEL_PATH  log=$LOG"

TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER:-$(id -un)}_port${PORT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
    nohup "$PY" -u -m lightnav.serving.ws_server \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --backend vllm_local \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --host "$HOST" \
        --port "$PORT" \
        --ready_file "$READY" \
        ${QUANT_FLAG[@]+"${QUANT_FLAG[@]}"} \
        ${extra_args[@]+"${extra_args[@]}"} \
        >"$LOG" 2>&1 &

SERVER_PID=$!
echo "[serve_thor] pid=$SERVER_PID  waiting for READY..."
for _ in $(seq 1 120); do
    [ -f "$READY" ] && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[serve_thor] FAILED before ready -- last log lines:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sleep 5
done
[ -f "$READY" ] || { echo "[serve_thor] TIMEOUT waiting for READY" >&2; tail -20 "$LOG" >&2; exit 1; }

IP=$(ip -4 -o addr show scope global 2>/dev/null | awk 'NR==1{split($4,a,"/"); print a[1]}')
echo "[serve_thor] READY  ->  ws://localhost:$PORT${IP:+  |  ws://$IP:$PORT}"
