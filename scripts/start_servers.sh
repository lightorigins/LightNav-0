#!/usr/bin/env bash
# Start N inference WebSocket servers, K per GPU, on consecutive ports.
#
# Knobs (env vars):
#   NUM_GPUS                 number of GPUs to use (default 1)
#   SERVERS_PER_GPU          servers per GPU (default 1: one shared engine per GPU)
#   BASE_PORT                first port (default 8050)
#   MODEL_PATH               HF checkpoint dir (required)
#   ACTION_TOKENIZER_BUNDLE  RVQ bundle dir (manifest.json + codebooks; optional:
#                            only to override the decoder the checkpoint ships)
#   TASK                     tracking | vln (default tracking)
#   NUM_HISTORY_FRAMES       history window override (optional)
#   BACKEND                  hf | vllm_local (default vllm_local)
#   POOL_SPATIAL             post-ViT spatial pooling override (optional)
#   ASPECT_MODE              stretch | keep (how non-16:9 client frames are fitted; optional)
#   MAX_BATCH_SIZE           micro-batch width per server (optional; server default 8)
#   MAX_NEW_TOKENS           lower bound on the decode cap (optional; server default 8)
#   GPU_MEM_UTIL             vLLM gpu_memory_utilization (default 0.85 / SERVERS_PER_GPU)
#   LOG_DIR                  server log dir (default logs/servers)
#   INFER_VENV               virtualenv to run from (default $REPO_ROOT/.venv, else PATH python)
#   CPU_PIN                  1 (default) pins each server to a core range with taskset; 0 disables
#   RECORD_DIR               record every connection's episodes under RECORD_DIR/port<PORT>/
#                            for `lightnav-render` (optional; off when empty). See docs/VISUALIZATION.md
#   RECORD_FPS               recording manifest fps (optional; server default 10)
#   RECORD_TIMELINE          realtime | per_step (optional; server default realtime)
#   RECORD_IMAGES            0 records the per-step JSON without frames (default 1)
#   CAM_HFOV_DEG             client camera horizontal FOV in degrees, for the overlay (server default 90)
#   CAM_HEIGHT               client camera height in metres, for the overlay (server default 0.5)
#   TRAJ_FORWARD_OFFSET      overlay forward offset in metres (optional; default automatic)
#   WAYPOINT_DT_S            seconds per waypoint row for the HUD readout (optional; server default 0.1)
#
# Example:
#   NUM_GPUS=2 \
#   MODEL_PATH=/path/to/hf_ckpt \
#   bash scripts/start_servers.sh
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
SERVERS_PER_GPU=${SERVERS_PER_GPU:-1}
BASE_PORT=${BASE_PORT:-8050}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH must be set to the HF checkpoint dir}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}
TASK=${TASK:-tracking}
NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES:-}
BACKEND=${BACKEND:-vllm_local}
POOL_SPATIAL=${POOL_SPATIAL:-}
ASPECT_MODE=${ASPECT_MODE:-}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
LOG_DIR=${LOG_DIR:-logs/servers}
RECORD_DIR=${RECORD_DIR:-}
RECORD_FPS=${RECORD_FPS:-}
RECORD_TIMELINE=${RECORD_TIMELINE:-}
RECORD_IMAGES=${RECORD_IMAGES:-1}
CAM_HFOV_DEG=${CAM_HFOV_DEG:-}
CAM_HEIGHT=${CAM_HEIGHT:-}
TRAJ_FORWARD_OFFSET=${TRAJ_FORWARD_OFFSET:-}
WAYPOINT_DT_S=${WAYPOINT_DT_S:-}

# vLLM gpu_memory_utilization: each instance grabs this fraction of free GPU
# memory at init. With multiple servers per GPU it must shrink accordingly,
# otherwise the 2nd+ instance fails with "No available memory for the cache
# blocks". Heuristic: leave ~15% headroom for ViT activations / cudaMalloc.
if [ -z "${GPU_MEM_UTIL:-}" ]; then
    if [ "$BACKEND" = "vllm_local" ]; then
        # 0.85 / SERVERS_PER_GPU, floor 0.10
        GPU_MEM_UTIL=$(awk -v n="$SERVERS_PER_GPU" 'BEGIN { v = 0.85 / n; if (v < 0.10) v = 0.10; printf "%.2f", v }')
    else
        GPU_MEM_UTIL=0.85
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_MODULE="lightnav.serving.ws_server"
PID_FILE="$REPO_ROOT/.servers.pids"
READY_DIR="$REPO_ROOT/.servers_ready"
# Do not inherit ROS/Gazebo's Python 3.12 modules into the Python 3.11
# inference process.  Importing a ROS extension from that path can crash the
# interpreter before lightnav has a chance to report a useful error.
export PYTHONPATH="$REPO_ROOT/src"

# Pick the Python interpreter. Override with INFER_VENV=/path/to/venv; defaults
# to $REPO_ROOT/.venv, else PATH `python`.
INFER_VENV="${INFER_VENV:-$REPO_ROOT/.venv}"
if [ -x "$INFER_VENV/bin/python" ]; then
    PY="$INFER_VENV/bin/python"
    # cu12.9 torch wheels need nvJitLink 12.9 symbols; prepend the venv's bundled
    # nvidia libs over any older system libnvJitLink.so.12. Without this
    # `import torch` can fail with an undefined nvJitLink symbol.
    NV_LIBS=$(echo "$INFER_VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr ' ' ':')
    # ROS prepends Gazebo/OGRE libraries (often with ABI-incompatible CUDA
    # dependencies) through LD_LIBRARY_PATH.  Keep only the venv's bundled
    # CUDA libraries; the system linker still supplies libc and libstdc++.
    [ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS"
else
    echo "[start_servers] WARN: $INFER_VENV not found; using PATH python."
    PY="python"
    # A PATH-provided interpreter must not accidentally load ROS's native
    # libraries either.  Leave LD_LIBRARY_PATH unset and use default linker
    # paths in this fallback mode.
    unset LD_LIBRARY_PATH
fi

mkdir -p "$LOG_DIR" "$READY_DIR"
find "$READY_DIR" -name '*.ready' -delete
: > "$PID_FILE"

extra_args=()
if [ -n "$ACTION_TOKENIZER_BUNDLE" ]; then
    extra_args+=("--action_tokenizer_bundle" "$ACTION_TOKENIZER_BUNDLE")
fi
if [ -n "$NUM_HISTORY_FRAMES" ]; then
    extra_args+=("--num_history_frames" "$NUM_HISTORY_FRAMES")
fi
if [ -n "$POOL_SPATIAL" ]; then
    extra_args+=("--pool_spatial" "$POOL_SPATIAL")
fi
if [ -n "$ASPECT_MODE" ]; then
    extra_args+=("--aspect_mode" "$ASPECT_MODE")
fi
if [ -n "$MAX_BATCH_SIZE" ]; then
    extra_args+=("--max_batch_size" "$MAX_BATCH_SIZE")
fi
if [ -n "$MAX_NEW_TOKENS" ]; then
    extra_args+=("--max_new_tokens" "$MAX_NEW_TOKENS")
fi
# Recording knobs shared by every server; --record_dir itself is per port (below) so
# servers started in the same second never share a run directory.
if [ -n "$RECORD_DIR" ]; then
    if [ -n "$RECORD_FPS" ]; then
        extra_args+=("--record_fps" "$RECORD_FPS")
    fi
    if [ -n "$RECORD_TIMELINE" ]; then
        extra_args+=("--record_timeline" "$RECORD_TIMELINE")
    fi
    case "$RECORD_IMAGES" in
        0|false|no|off) extra_args+=("--no_record_images") ;;
    esac
fi
if [ -n "$CAM_HFOV_DEG" ]; then
    extra_args+=("--cam_hfov_deg" "$CAM_HFOV_DEG")
fi
if [ -n "$CAM_HEIGHT" ]; then
    extra_args+=("--cam_height" "$CAM_HEIGHT")
fi
if [ -n "$TRAJ_FORWARD_OFFSET" ]; then
    extra_args+=("--traj_forward_offset" "$TRAJ_FORWARD_OFFSET")
fi
if [ -n "$WAYPOINT_DT_S" ]; then
    extra_args+=("--waypoint_dt_s" "$WAYPOINT_DT_S")
fi

TOTAL=$((NUM_GPUS * SERVERS_PER_GPU))

# CPU pinning + per-server thread caps avoid the contention seen when several
# servers (or simulator workers) share a node: without this every server's
# BLAS/video/ViT preprocessing fans out across all cores and N servers stomp on
# the same pool -- typical symptom: per-step latency jitters 100-500 ms instead
# of a stable ~150 ms, with GPU util waiting on CPU.
#
# CPU_PIN=0 disables the taskset bind (e.g. a single server, or a node shared
# with other CPU-heavy workloads that need elasticity).
CPU_PIN=${CPU_PIN:-1}
NPROC=$(nproc 2>/dev/null || echo 0)
if [ "$NPROC" -gt 0 ] && [ "$TOTAL" -gt 0 ]; then
    CORES_PER_SERVER=$((NPROC / TOTAL))
    [ "$CORES_PER_SERVER" -lt 1 ] && CORES_PER_SERVER=1
else
    CORES_PER_SERVER=4  # safe fallback
fi

# Cap BLAS/OMP threads well below the pinned core count: small per-step CPU ops
# (video patchify, embed assembly) are dominated by thread-launch overhead when
# fanned across ~96 threads -- measured 6-10x slower than ~32. Never exceed the
# cores actually pinned to the server (clamp below).
THREAD_CAP=32
[ "$THREAD_CAP" -gt "$CORES_PER_SERVER" ] && THREAD_CAP=$CORES_PER_SERVER

echo "Starting $TOTAL inference servers"
echo "  MODEL_PATH=$MODEL_PATH"
if [ -n "$ACTION_TOKENIZER_BUNDLE" ]; then
    echo "  ACTION_TOKENIZER_BUNDLE=$ACTION_TOKENIZER_BUNDLE"
fi
echo "  TASK=$TASK  BACKEND=$BACKEND  SERVERS_PER_GPU=$SERVERS_PER_GPU  GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "  CPU_PIN=$CPU_PIN  NPROC=$NPROC  CORES_PER_SERVER=$CORES_PER_SERVER  THREAD_CAP=$THREAD_CAP"
if [ -n "$RECORD_DIR" ]; then
    echo "  RECORD_DIR=$RECORD_DIR  (one port<PORT>/ subdir per server; render with: lightnav-render $RECORD_DIR)"
fi
URLS=()
IDX=0
for ((g = 0; g < NUM_GPUS; g++)); do
    for ((s = 0; s < SERVERS_PER_GPU; s++)); do
        PORT=$((BASE_PORT + IDX))
        LOG="$LOG_DIR/server_gpu${g}_port${PORT}.log"
        READY="$READY_DIR/port${PORT}.ready"

        # taskset bind (server IDX -> cores [IDX*N, (IDX+1)*N - 1])
        if [ "$CPU_PIN" = "1" ] && [ "$NPROC" -gt 0 ]; then
            CPU_START=$((IDX * CORES_PER_SERVER))
            CPU_END=$((CPU_START + CORES_PER_SERVER - 1))
            CPU_PREFIX=(taskset --cpu-list "${CPU_START}-${CPU_END}")
            CPU_DESC=" CPUS ${CPU_START}-${CPU_END}"
        else
            CPU_PREFIX=()
            CPU_DESC=""
        fi
        echo "  GPU $g  PORT $PORT${CPU_DESC}  LOG $LOG"

        record_args=()
        if [ -n "$RECORD_DIR" ]; then
            record_args=("--record_dir" "$RECORD_DIR/port${PORT}")
        fi

        # Per-port inductor cache dir avoids concurrent-write races on the shared
        # default cache that produce noisy (but harmless) warnings on cold starts.
        TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER:-root}_port${PORT}" \
        CUDA_VISIBLE_DEVICES=$g PORT=$PORT PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=$THREAD_CAP \
        MKL_NUM_THREADS=$THREAD_CAP \
        OPENBLAS_NUM_THREADS=$THREAD_CAP \
        NUMEXPR_NUM_THREADS=$THREAD_CAP \
            nohup "${CPU_PREFIX[@]}" "$PY" -u -m "$SERVER_MODULE" \
                --model_path "$MODEL_PATH" \
                --task "$TASK" \
                --K "$K" --horizon "$HORIZON" \
                --backend "$BACKEND" \
                --gpu_memory_utilization "$GPU_MEM_UTIL" \
                --port "$PORT" \
                --ready_file "$READY" \
                "${extra_args[@]}" "${record_args[@]}" \
                >"$LOG" 2>&1 &
        echo "$! $PORT $g" >> "$PID_FILE"
        URLS+=("ws://localhost:$PORT")
        IDX=$((IDX + 1))
    done
done

cat <<EOF

PIDs  -> $PID_FILE   (one line per server: "pid port gpu")
Logs  -> $LOG_DIR
Ready -> $READY_DIR  (one .ready file per server, created once its port is bound)

Wait for all servers to be ready (check ready files):
  until [ \$(find $READY_DIR -name '*.ready' | wc -l) -eq $TOTAL ]; do sleep 2; done && echo "all $TOTAL servers ready"

Or check log lines:
  grep -c '\[lightnav-ws\] READY' $LOG_DIR/*.log | awk -F: '{s+=\$2} END {print s}'

Stop them:
  awk '{print \$1}' $PID_FILE | xargs -r kill

Then drive any of them with the reference client:
  lightnav-ws-client --server ${URLS[0]} \\
      --video clip.mp4 --fps 4 --instruction "follow the person in the red shirt"

Server URLs: ${URLS[*]}
EOF
