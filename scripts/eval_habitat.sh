#!/usr/bin/env bash
# Parallel Habitat evaluation on the local GPUs, one shard per GPU:
#   1. start one Habitat env server per GPU (habitat conda env), each serving a
#      disjoint shard of the split (--split-id i --split-num N)
#   2. wait until every server has written its .ready file
#   3. start one eval client per GPU (the lightnav env; model on the same GPU)
#   4. merge the shards' results.jsonl into OUTPUT_ROOT/summary.json
# Servers and clients are killed on exit (trap).
#
# Knobs (env vars):
#   # ---- model ----
#   MODEL_PATH           checkpoint dir (required)
#   BACKEND              vllm_local | hf (default vllm_local)
#   GPU_MEM_UTIL         vLLM GPU memory fraction for the client (default 0.65 — the
#                        simulator renders on the same GPU)
#   ACTION_TOKENIZER_BUNDLE  RVQ bundle override (default: resolved from eval_config.json)
#   CLIENT_ARGS          extra lightnav-eval-habitat flags, e.g. "--save_video --verbose"
#   # ---- benchmark ----
#   TASK                 vlnce | objectnav (default vlnce)
#   HABITAT_CONFIG       yaml (default habitat_server/configs/vlnce_r2r.yaml)
#   SPLIT                dataset split (default: vlnce -> val_unseen; objectnav -> yaml)
#   SUCCESS_DISTANCE     e.g. 0.25 for HM3D-OVON (default: env default)
#   NAVMESH_CELL_HEIGHT  re-bake navmeshes at this cell_height; REQUIRED as 0.05 for
#                        HM3D ObjectNav v2, leave unset otherwise (objectnav only)
#   DATA_PATH, SCENES_DIR  optional dataset path overrides (see docs/HABITAT_SERVER.md)
#   LANGUAGES            RxR only, e.g. "en-US en-IN"
#   EPISODES             per shard; -1 = whole shard (default -1)
#   MAX_STEPS            default 500
#   SERVER_ARGS          extra lightnav_habitat.serve flags
#   # ---- topology ----
#   GPU_IDS              GPUs to use, e.g. "0 1 3" (default: all visible GPUs)
#   NUM_GPUS             alternative to GPU_IDS: use GPUs 0..NUM_GPUS-1
#   BASE_PORT            first ZMQ port (default 5555)
#   READY_TIMEOUT_S      max wait for the servers (default 900)
#   # ---- environments ----
#   HABITAT_CONDA_ENV    conda env with habitat-sim + lightnav_habitat (default habitat), or
#   HABITAT_PYTHON       explicit interpreter for the env server (overrides HABITAT_CONDA_ENV)
#   INFER_VENV           lightnav virtualenv (default <repo>/.venv), or
#   CLIENT_PYTHON        explicit interpreter for the eval client
#   # ---- output ----
#   OUTPUT_ROOT          default output/habitat_<task>_<timestamp>; shards in shard_<i>/
#   LOG_DIR              default $OUTPUT_ROOT/logs
#
# Examples:
#   MODEL_PATH=/path/to/checkpoint bash scripts/eval_habitat.sh                 # R2R on every GPU
#   MODEL_PATH=... HABITAT_CONFIG=habitat_server/configs/vlnce_rxr.yaml LANGUAGES="en-US en-IN" \
#       GPU_IDS="0 1" bash scripts/eval_habitat.sh                              # RxR on 2 GPUs
#   MODEL_PATH=... TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_ovon.yaml \
#       SPLIT=val_unseen SUCCESS_DISTANCE=0.25 bash scripts/eval_habitat.sh      # HM3D-OVON
#   MODEL_PATH=... TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_mp3d.yaml \
#       SPLIT=val bash scripts/eval_habitat.sh                                    # MP3D v1
set -euo pipefail

TAG="[eval_habitat]"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── model ────────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the checkpoint dir}
[ -d "$MODEL_PATH" ] || { echo "$TAG ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
BACKEND=${BACKEND:-vllm_local}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}
CLIENT_ARGS=${CLIENT_ARGS:-}

# ── benchmark ────────────────────────────────────────────────────────────────
TASK=${TASK:-vlnce}
case "$TASK" in vlnce|objectnav) ;; *) echo "$TAG ERROR: TASK must be vlnce or objectnav" >&2; exit 1;; esac
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
[ -f "$HABITAT_CONFIG" ] || { echo "$TAG ERROR: HABITAT_CONFIG not found: $HABITAT_CONFIG" >&2; exit 1; }
HABITAT_CONFIG="$(cd "$(dirname "$HABITAT_CONFIG")" && pwd)/$(basename "$HABITAT_CONFIG")"
if [ -z "${SPLIT+x}" ]; then
    [ "$TASK" = "vlnce" ] && SPLIT=val_unseen || SPLIT=""
fi
SUCCESS_DISTANCE=${SUCCESS_DISTANCE:-}
NAVMESH_CELL_HEIGHT=${NAVMESH_CELL_HEIGHT:-}
DATA_PATH=${DATA_PATH:-}
SCENES_DIR=${SCENES_DIR:-}
LANGUAGES=${LANGUAGES:-}
EPISODES=${EPISODES:--1}
MAX_STEPS=${MAX_STEPS:-500}
SERVER_ARGS=${SERVER_ARGS:-}

# ── topology ─────────────────────────────────────────────────────────────────
if [ -n "${GPU_IDS:-}" ]; then
    read -ra GPUS <<< "$GPU_IDS"
elif [ -n "${NUM_GPUS:-}" ]; then
    GPUS=(); for ((i = 0; i < NUM_GPUS; i++)); do GPUS+=("$i"); done
else
    DETECTED=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    [ "${DETECTED:-0}" -ge 1 ] || { echo "$TAG ERROR: no GPU detected; set GPU_IDS or NUM_GPUS" >&2; exit 1; }
    GPUS=(); for ((i = 0; i < DETECTED; i++)); do GPUS+=("$i"); done
fi
N=${#GPUS[@]}
BASE_PORT=${BASE_PORT:-5555}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-900}

# ── interpreters ─────────────────────────────────────────────────────────────
# Habitat side: `conda activate` only prepends PATH, so use the absolute
# $CONDA_PREFIX/bin/python afterwards and put the env's own libs first for habitat-sim.
HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV:-habitat}
HABITAT_PYTHON=${HABITAT_PYTHON:-}
if [ -z "$HABITAT_PYTHON" ]; then
    CONDA_SH="${CONDA_SH:-}"
    if [ -z "$CONDA_SH" ]; then
        for candidate in \
            "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "$HOME/miniforge3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then CONDA_SH="$candidate"; break; fi
        done
    fi
    [ -n "$CONDA_SH" ] || {
        echo "$TAG ERROR: cannot locate conda.sh; set CONDA_SH=/path/to/etc/profile.d/conda.sh or HABITAT_PYTHON=/path/to/python" >&2
        exit 1; }
    set +u   # conda's activate scripts reference unset variables
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$HABITAT_CONDA_ENV" || {
        echo "$TAG ERROR: conda activate $HABITAT_CONDA_ENV failed; available envs:" >&2
        conda env list >&2
        exit 1; }
    set -u
    HABITAT_PYTHON="$CONDA_PREFIX/bin/python"
    HABITAT_LD="$CONDA_PREFIX/lib"
    conda deactivate || true
else
    HABITAT_LD=""
fi
[ -x "$HABITAT_PYTHON" ] || { echo "$TAG ERROR: HABITAT_PYTHON is not executable: $HABITAT_PYTHON" >&2; exit 1; }
PYTHONPATH="$REPO_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}" "$HABITAT_PYTHON" \
    -c "import habitat, habitat_sim, lightnav_habitat.serve" 2>/dev/null || {
    echo "$TAG ERROR: $HABITAT_PYTHON cannot import habitat / habitat_sim / lightnav_habitat (see docs/HABITAT_SERVER.md)" >&2; exit 1; }

# Model side: the lightnav virtualenv.
CLIENT_PYTHON=${CLIENT_PYTHON:-}
INFER_VENV="${INFER_VENV:-$REPO_ROOT/.venv}"
CLIENT_LD=""
if [ -z "$CLIENT_PYTHON" ]; then
    if [ -x "$INFER_VENV/bin/python" ]; then
        CLIENT_PYTHON="$INFER_VENV/bin/python"
        # torch wheels need their own bundled nvJitLink first on LD_LIBRARY_PATH.
        CLIENT_LD=$(echo "$INFER_VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr ' ' ':')
    else
        echo "$TAG WARN: $INFER_VENV not found; using PATH python for the client."
        CLIENT_PYTHON="python"
    fi
fi
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat, zmq" 2>/dev/null || {
    echo "$TAG ERROR: $CLIENT_PYTHON cannot import lightnav (+ pyzmq): pip install -e '.[habitat]'" >&2; exit 1; }

# ── output ───────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/output/habitat_${TASK}_${TS}}
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs}
READY_DIR="$OUTPUT_ROOT/.ready"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$READY_DIR"
rm -f "$READY_DIR"/*.ready

echo "$TAG task=$TASK config=$HABITAT_CONFIG split=${SPLIT:-<yaml>} model=$MODEL_PATH backend=$BACKEND"
echo "$TAG gpus=(${GPUS[*]}) shards=$N base_port=$BASE_PORT output=$OUTPUT_ROOT"

# ── cleanup ──────────────────────────────────────────────────────────────────
SERVER_PIDS=()
CLIENT_PIDS=()
cleanup() {
    echo "$TAG cleaning up..."
    for pid in "${CLIENT_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${CLIENT_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ── 1. env servers ───────────────────────────────────────────────────────────
server_args=(--task "$TASK" --config "$HABITAT_CONFIG" --max-steps "$MAX_STEPS")
[ -n "$SPLIT" ] && server_args+=(--split "$SPLIT")
[ -n "$SUCCESS_DISTANCE" ] && server_args+=(--success-distance "$SUCCESS_DISTANCE")
[ -n "$NAVMESH_CELL_HEIGHT" ] && server_args+=(--navmesh-cell-height "$NAVMESH_CELL_HEIGHT")
[ -n "$DATA_PATH" ] && server_args+=(--data-path "$DATA_PATH")
[ -n "$SCENES_DIR" ] && server_args+=(--scenes-dir "$SCENES_DIR")
# shellcheck disable=SC2206
[ -n "$SERVER_ARGS" ] && server_args+=($SERVER_ARGS)

for ((i = 0; i < N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT + i))
    log="$LOG_DIR/server_gpu${g}_shard${i}.log"
    echo "$TAG server shard $i/$N on GPU $g -> tcp://localhost:$port  (log: $log)"
    HABITAT_SIM_GPU_ID=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="${HABITAT_LD}${HABITAT_LD:+:}${LD_LIBRARY_PATH:-}" \
        "$HABITAT_PYTHON" -m lightnav_habitat.serve "${server_args[@]}" \
            --port "$port" --split-id "$i" --split-num "$N" \
            --ready-file "$READY_DIR/shard${i}.ready" >"$log" 2>&1 &
    SERVER_PIDS+=("$!")
done

# ── 2. wait for ready ────────────────────────────────────────────────────────
DEADLINE=$((SECONDS + READY_TIMEOUT_S))
LAST=-1
while :; do
    # -mindepth 1 -type f: without it `find` also matches the $READY_DIR directory itself
    # (it is named ".ready"), so the count was always files+1. Combined with the old strict
    # `-eq $N` test that let the clients start one server early, and hung until
    # READY_TIMEOUT_S whenever a poll never observed exactly N-1 files -- which is the
    # normal case on a multi-GPU box, where the last servers become ready inside one tick.
    READY=$(find "$READY_DIR" -mindepth 1 -maxdepth 1 -type f -name '*.ready' 2>/dev/null | wc -l)
    [ "$READY" -ne "$LAST" ] && { echo "$TAG   $READY/$N servers ready"; LAST=$READY; }
    [ "$READY" -ge "$N" ] && break
    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$TAG ERROR: an env server exited before becoming ready; last log lines:" >&2
            for f in "$LOG_DIR"/server_*.log; do echo "--- $f ---" >&2; tail -15 "$f" >&2; done
            exit 1
        fi
    done
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "$TAG ERROR: only $READY/$N servers ready after ${READY_TIMEOUT_S}s" >&2
        for f in "$LOG_DIR"/server_*.log; do echo "--- $f ---" >&2; tail -15 "$f" >&2; done
        exit 1
    fi
    sleep 3
done

# ── 3. eval clients ──────────────────────────────────────────────────────────
client_args=(--model_path "$MODEL_PATH" --backend "$BACKEND" --episodes "$EPISODES" \
             --max_steps "$MAX_STEPS" --gpu_memory_utilization "$GPU_MEM_UTIL")
[ -n "$ACTION_TOKENIZER_BUNDLE" ] && client_args+=(--action_tokenizer_bundle "$ACTION_TOKENIZER_BUNDLE")
# shellcheck disable=SC2206
[ -n "$LANGUAGES" ] && client_args+=(--languages $LANGUAGES)
# shellcheck disable=SC2206
[ -n "$CLIENT_ARGS" ] && client_args+=($CLIENT_ARGS)

for ((i = 0; i < N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT + i))
    shard_dir="$OUTPUT_ROOT/shard_$i"
    log="$LOG_DIR/client_gpu${g}_shard${i}.log"
    echo "$TAG client shard $i/$N on GPU $g -> $shard_dir  (log: $log)"
    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="${CLIENT_LD}${CLIENT_LD:+:}${LD_LIBRARY_PATH:-}" \
        "$CLIENT_PYTHON" -m lightnav.cli.eval_habitat "${client_args[@]}" \
            --server "tcp://localhost:$port" --output_dir "$shard_dir" >"$log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

FAILED=0
for ((i = 0; i < N; i++)); do
    if wait "${CLIENT_PIDS[$i]}"; then
        echo "$TAG shard $i finished"
    else
        echo "$TAG WARN: shard $i client exited with an error (see $LOG_DIR/client_*_shard${i}.log)" >&2
        FAILED=$((FAILED + 1))
    fi
done

# ── 4. merge ─────────────────────────────────────────────────────────────────
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -m lightnav.cli.eval_merge "$OUTPUT_ROOT"/shard_* --output "$OUTPUT_ROOT" \
    | tee "$OUTPUT_ROOT/merge.log"
echo "$TAG done: $OUTPUT_ROOT/summary.json  (failed shards: $FAILED)"
[ "$FAILED" -eq 0 ]
