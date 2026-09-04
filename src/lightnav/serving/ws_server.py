#!/usr/bin/env python3
"""Minimal WebSocket inference server for trajectory-token checkpoints.

One shared engine plus a :class:`BatchedTrackingService` fronts every
connection; ``predict()`` calls from concurrent sessions are micro-batched so
the GPU runs one batched decode per tick. Each connection owns one session (a
frame buffer plus a per-session ViT cache); nothing persists across
connections, and ``clientId`` is log metadata only.

Protocol (JSON text frames, one response per request; see docs/PROTOCOL.md):

    login {clientId?}                          -> {rc:0, msg:"ok"}
    reset {}                                   -> {rc:0, msg:"ok"}
    next  {seq, image (JPEG b64), instruction} -> {rc:0, seq, actions:{step, actions},
                                                   stop, visible, latency_ms, timings_ms,
                                                   raw_text, [pointing]}

A ``next`` whose instruction is empty or null only buffers the frame and is
acknowledged with ``{rc:0, seq, msg:"image received"}``.

Usage (single server):
    PORT=8050 CUDA_VISIBLE_DEVICES=0 lightnav-serve \\
        --task tracking \\
        --model_path /path/to/hf_ckpt \\
        --traj_vocab_path /path/to/traj_vocab \\
        --K 256 --horizon 10

``--record_dir DIR`` additionally records every connection's episodes (the client's
JPEG frames + one JSON record per prediction) for offline rendering with
``lightnav-render DIR``; see docs/VISUALIZATION.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.exceptions import ConnectionClosed

from lightnav.inference.config import InferenceConfig
from lightnav.inference.engine import build_engine
from lightnav.serving.protocol import actions_payload, pointing_payload
from lightnav.serving.token_budget import (
    action_token_count,
    decode_token_budget,
    probe_grounding_tokens,
)
from lightnav.serving.tracking_service import BatchedTrackingService
from lightnav.tracking import load_centroids
from lightnav.vln_utils import DEFAULT_TRAJ_HORIZON, DEFAULT_TRAJ_K

__all__ = [
    "SERVER_LOG_TAG",
    "action_token_count",
    "decode_token_budget",
    "probe_grounding_tokens",
    "make_handler",
    "serve_forever",
    "main",
]

logger = logging.getLogger(__name__)
SERVER_LOG_TAG = "lightnav-ws"
# The instruction used for the one-shot startup warmup; fixed so its cost is
# comparable across restarts.
_WARMUP_INSTRUCTION = "walk forward through the corridor and stop"
# raw_text is model output: bounded by the decode cap in practice, never trusted to be.
_RAW_TEXT_MAX_CHARS = 256


def _bounded_raw_text(value: object) -> str:
    text = value if isinstance(value, str) else ""
    return text if len(text) <= _RAW_TEXT_MAX_CHARS else text[:_RAW_TEXT_MAX_CHARS] + "..."


def _decode_jpeg_b64_with_bytes(b64_str: str) -> tuple[np.ndarray, bytes]:
    """Base64 JPEG/PNG -> ``(HWC uint8 RGB array, the decoded image bytes)``.

    The bytes are what the client actually sent; the recorder stores them verbatim
    so a recorded frame is never re-encoded.
    """
    raw = base64.b64decode(b64_str, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8), raw


def _decode_jpeg_b64(b64_str: str) -> np.ndarray:
    """Base64 JPEG/PNG -> HWC uint8 RGB array."""
    return _decode_jpeg_b64_with_bytes(b64_str)[0]


# A clientId is only used as a directory name when it is plainly safe as one.
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _recorder_label(client_id: str | None) -> str | None:
    """Return ``client_id`` when it is a safe directory name, else None (recorder default)."""
    if isinstance(client_id, str) and _SAFE_LABEL_RE.match(client_id):
        return client_id
    return None


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from None


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from None


def _build_recorder(args: argparse.Namespace):
    """Create the local episode recorder for ``--record_dir`` (None when recording is off).

    ``lightnav.viz`` is imported here, not at module level, so a core install keeps
    serving with recording off; recording itself only needs numpy + Pillow, the
    offline renderer (``lightnav-render``) needs ``pip install 'lightnav[video]'``.
    """
    record_dir = getattr(args, "record_dir", "") or ""
    if not record_dir:
        return None
    from lightnav.viz import EpisodeRecorder

    forward_offset = getattr(args, "traj_forward_offset", None)
    return EpisodeRecorder(
        Path(record_dir),
        task=args.task,
        model_path=args.model_path,
        hfov_deg=float(args.cam_hfov_deg),
        cam_height=float(args.cam_height),
        forward_offset=None if forward_offset is None else float(forward_offset),
        video_fps=int(args.record_fps),
        timeline=args.record_timeline,
        waypoint_dt_s=float(args.waypoint_dt_s),
        save_images=bool(args.record_images),
    )


class _RequestDispatchError(Exception):
    """A client request error with a stable wire response code."""

    def __init__(self, message: str, *, rc: int = 400) -> None:
        super().__init__(message)
        self.rc = rc


def _engine_task_for_server_task(server_task: str) -> str:
    """Map the served protocol task onto the checkpoint's eval_config task key."""
    if server_task == "tracking":
        return "trackvla"
    if server_task == "vln":
        return "vlnce"
    raise ValueError(f"task must be 'tracking' or 'vln', got {server_task!r}")


def _build_engine(args: argparse.Namespace):
    cfg = InferenceConfig(
        model_path=args.model_path,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_batch_size,
        num_history_frames=args.num_history_frames,
        aspect_mode=getattr(args, "aspect_mode", "stretch"),
        quantization=getattr(args, "quantization", None),
        vit_cache_entries=getattr(args, "vit_cache_entries", None),
    )
    if args.pool_spatial is not None:
        cfg.pool_spatial = args.pool_spatial
    engine, bundle = build_engine(
        cfg,
        task_type=_engine_task_for_server_task(args.task),
        max_new_tokens=args.max_new_tokens,
    )
    return engine, bundle


def _request_action(value: object) -> str:
    return value if isinstance(value, str) and value else "error"


def _parse_seq(data: Mapping[str, object]) -> int:
    if "seq" not in data:
        raise _RequestDispatchError("missing seq")
    raw_seq = data["seq"]
    if isinstance(raw_seq, bool):
        raise _RequestDispatchError("seq must be an integer")
    try:
        seq = int(raw_seq)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _RequestDispatchError("seq must be an integer") from exc
    if isinstance(raw_seq, float) and (not math.isfinite(raw_seq) or raw_seq != seq):
        raise _RequestDispatchError("seq must be an integer")
    return seq


def _success_response(pred, seq: int, latency_ms: float, session, rgb: np.ndarray) -> dict:
    raw_text = str(getattr(pred, "raw_text", "") or "")
    data: dict[str, object] = {
        "rc": 0,
        "seq": seq,
        "actions": actions_payload(pred.waypoints, step=int(getattr(session, "_buffer_len", 0))),
        "latency_ms": float(latency_ms),
        "stop": bool(pred.stop),
        "visible": None if pred.visible is None else bool(pred.visible),
        "timings_ms": dict(getattr(pred, "timings_ms", {}) or {}),
        "raw_text": _bounded_raw_text(raw_text),
    }
    # Pixels in the frame THIS request carried, so the client can place a marker
    # without knowing the token encoding or what the server resized to. Absent for
    # checkpoints that emit no pointing tokens.
    pointing = pointing_payload(raw_text, width=int(rgb.shape[1]), height=int(rgb.shape[0]))
    if pointing is not None:
        data["pointing"] = pointing
    return {"action": "next", "data": data}


# ── WebSocket handler ─────────────────────────────────────────────────────────


def make_handler(service, recorder=None):
    """Build the per-connection handler around a shared ``service``.

    ``service`` needs ``make_session(client_id=None)`` and ``async predict(session)``;
    the session needs ``reset(instruction)``, ``observe(rgb)``, ``instruction`` and
    (optionally) ``_buffer_len``. Each connection owns one session, created lazily on
    its first message.

    ``recorder`` (optional, an ``EpisodeRecorder``-like object with
    ``begin_connection(label=None)``) enables local episode recording: one connection
    recorder per WebSocket connection, a new episode on every ``reset`` (or on the first
    prediction when none is open), one record per predicted ``next``. Recording is
    diagnostics only: it never changes a response, and any recorder error is logged
    and dropped.
    """

    async def handler(websocket, path=None):
        addr = getattr(websocket, "remote_address", "?")
        logger.info("[%s] client connected: %s", SERVER_LOG_TAG, addr)
        session = None
        client_id: str | None = None
        conn_recorder = None
        episode_open = False

        def ensure_session():
            nonlocal session
            if session is None:
                if client_id is None:
                    session = service.make_session()
                else:
                    session = service.make_session(client_id=client_id)
                session.reset(instruction="")
            elif client_id is not None and getattr(session, "client_id", None) is None:
                session.client_id = client_id
            return session

        def record_begin_episode(*, only_if_closed: bool) -> None:
            """Start a recorded episode; a no-op when recording is off."""
            nonlocal conn_recorder, episode_open
            if recorder is None or (only_if_closed and episode_open):
                return
            try:
                if conn_recorder is None:
                    conn_recorder = recorder.begin_connection(label=_recorder_label(client_id))
                conn_recorder.begin_episode()
                episode_open = True
            except Exception as exc:
                logger.warning(
                    "[%s] recorder begin_episode failed: %s: %s",
                    SERVER_LOG_TAG, type(exc).__name__, exc,
                )

        def record_step(
            *, current_session, seq: int, image: bytes, instruction: str, pred,
            latency_ms: float, response: Mapping[str, object],
        ) -> None:
            """Append one predicted step; a no-op when recording is off."""
            if recorder is None:
                return
            record_begin_episode(only_if_closed=True)
            if conn_recorder is None:
                return
            try:
                response_data = response.get("data", {})
                waypoints = getattr(pred, "waypoints", None)
                conn_recorder.record_step(
                    step=int(getattr(current_session, "_buffer_len", 0)),
                    seq=int(seq),
                    image=image,
                    instruction=instruction,
                    waypoints=None if waypoints is None else np.asarray(waypoints).tolist(),
                    stop=bool(getattr(pred, "stop", False)),
                    visible=response_data.get("visible") if isinstance(response_data, Mapping) else None,
                    raw_text=str(getattr(pred, "raw_text", "") or ""),
                    latency_ms=float(latency_ms),
                    pointing=response_data.get("pointing") if isinstance(response_data, Mapping) else None,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] recorder record_step failed (seq=%s): %s: %s",
                    SERVER_LOG_TAG, seq, type(exc).__name__, exc,
                )

        try:
            while True:
                try:
                    msg = await websocket.recv()
                except ConnectionClosed:
                    break

                action = "error"
                seq: int | None = None
                response_attempted = False
                response_sent = False

                async def send_response(response: Mapping[str, object]) -> None:
                    nonlocal response_attempted, response_sent
                    response_attempted = True
                    await websocket.send(json.dumps(response))
                    response_sent = True

                try:
                    try:
                        payload = json.loads(msg)
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise _RequestDispatchError(f"bad json: {exc}") from exc
                    if not isinstance(payload, Mapping):
                        raise _RequestDispatchError("payload must be an object")
                    action = _request_action(payload.get("action"))
                    data = payload.get("data", {})
                    if not isinstance(data, Mapping):
                        raise _RequestDispatchError("data must be an object")

                    if action == "login":
                        if "clientId" in data:
                            maybe_client_id = data.get("clientId")
                            if maybe_client_id is not None and not isinstance(maybe_client_id, str):
                                raise _RequestDispatchError("clientId must be a string")
                            client_id = maybe_client_id
                        ensure_session()
                        logger.info(
                            "[%s] client login: %s client_id=%s", SERVER_LOG_TAG, addr, client_id or "-"
                        )
                        await send_response({"action": "login", "data": {"rc": 0, "msg": "ok"}})

                    elif action == "reset":
                        ensure_session().reset(instruction="")
                        record_begin_episode(only_if_closed=False)
                        await send_response({"action": "reset", "data": {"rc": 0, "msg": "ok"}})

                    elif action == "next":
                        seq = _parse_seq(data)
                        img_b64 = data.get("image")
                        if not isinstance(img_b64, str) or not img_b64:
                            raise _RequestDispatchError("missing image")
                        instruction_value = data.get("instruction", "")
                        if instruction_value is None:
                            instruction = ""
                        elif not isinstance(instruction_value, str):
                            raise _RequestDispatchError("instruction must be a string")
                        else:
                            instruction = instruction_value
                        current_session = ensure_session()
                        try:
                            rgb, image_bytes = _decode_jpeg_b64_with_bytes(img_b64)
                        except Exception as exc:
                            raise _RequestDispatchError(f"bad image: {exc}") from exc
                        # The frame is buffered even when there is nothing to predict yet.
                        current_session.observe(rgb)
                        if not instruction:
                            await send_response(
                                {"action": "next", "data": {"rc": 0, "seq": seq, "msg": "image received"}}
                            )
                            continue
                        current_session.instruction = instruction
                        t0 = time.monotonic()
                        try:
                            pred = await service.predict(current_session)
                        except Exception as exc:
                            logger.warning(
                                "[%s] predict failed (seq=%s): %s: %s",
                                SERVER_LOG_TAG, seq, type(exc).__name__, exc,
                            )
                            await send_response(
                                {"action": "next", "data": {"rc": 500, "seq": seq, "msg": str(exc)}}
                            )
                            continue
                        latency_ms = (time.monotonic() - t0) * 1000.0
                        # A prediction that cannot be serialised or delivered ends the
                        # connection; it is never retried and never answered twice.
                        try:
                            response = _success_response(pred, seq, latency_ms, current_session, rgb)
                            await send_response(response)
                        except ConnectionClosed:
                            break
                        except Exception:
                            logger.exception(
                                "[%s] failed to deliver prediction (seq=%s)", SERVER_LOG_TAG, seq
                            )
                            break
                        # Recorded after the reply is on the wire, so recording never
                        # delays or reorders a response. The record carries the very
                        # JPEG bytes the client sent and the pointing payload it received.
                        record_step(
                            current_session=current_session,
                            seq=seq,
                            image=image_bytes,
                            instruction=instruction,
                            pred=pred,
                            latency_ms=latency_ms,
                            response=response,
                        )

                    else:
                        raise _RequestDispatchError(f"unknown action: {action!r}")

                except ConnectionClosed:
                    break
                except Exception as exc:
                    if response_sent:
                        logger.exception(
                            "[%s] error after the response was sent (seq=%s)", SERVER_LOG_TAG, seq
                        )
                        continue
                    if response_attempted:
                        logger.error("[%s] send failed (seq=%s): %s", SERVER_LOG_TAG, seq, exc)
                        break
                    rc = exc.rc if isinstance(exc, _RequestDispatchError) else 500
                    logger.log(
                        logging.WARNING if rc == 400 else logging.ERROR,
                        "[%s] request error rc=%d action=%s seq=%s: %s",
                        SERVER_LOG_TAG, rc, action, seq, exc,
                    )
                    response_data: dict[str, object] = {"rc": rc, "msg": str(exc)}
                    if seq is not None:
                        response_data["seq"] = seq
                    try:
                        await send_response({"action": action, "data": response_data})
                    except ConnectionClosed:
                        break
                    except Exception as send_exc:
                        logger.error("[%s] send failed (seq=%s): %s", SERVER_LOG_TAG, seq, send_exc)
                        break
        finally:
            if conn_recorder is not None:
                try:
                    conn_recorder.close()
                except Exception as exc:
                    logger.warning(
                        "[%s] recorder close failed: %s: %s", SERVER_LOG_TAG, type(exc).__name__, exc
                    )
            logger.info(
                "[%s] client disconnected: %s client_id=%s", SERVER_LOG_TAG, addr, client_id or "-"
            )

    return handler


# ── Server lifecycle ──────────────────────────────────────────────────────────


def _log_ready(args: argparse.Namespace, recorder=None) -> None:
    logger.info(
        "[%s] READY host=%s port=%d task=%s K=%d H=%d backend=%s model_path=%s",
        SERVER_LOG_TAG,
        args.host,
        args.port,
        args.task,
        args.K,
        args.horizon,
        args.backend,
        args.model_path,
    )
    if recorder is not None:
        logger.info(
            "[%s] recording episodes to %s (fps=%s timeline=%s images=%s); render with "
            "`lightnav-render %s`",
            SERVER_LOG_TAG,
            getattr(recorder, "run_dir", args.record_dir),
            args.record_fps,
            args.record_timeline,
            "on" if args.record_images else "off",
            args.record_dir,
        )


async def _warmup(service: BatchedTrackingService, bundle) -> None:
    """One synthetic frame through the full predict path before the port binds.

    The first inference on a cold process pays for lazy CUDA graphs, kernel
    autotuning and allocator growth; running it here means "listening" really
    means "ready". The GPU work is the point, so a decode ``ValueError`` on the
    synthetic frame (an all-zero image can legitimately produce an off-vocabulary
    token) is logged and ignored; any other failure is a broken engine and is
    re-raised so the server does not report READY while every request would 500.
    """
    height, width = (int(v) for v in bundle.video_size)
    session = service.make_session()
    session.reset(instruction="")
    session.observe(np.zeros((height, width, 3), dtype=np.uint8))
    session.instruction = _WARMUP_INSTRUCTION
    t0 = time.monotonic()
    try:
        await service.predict(session)
    except ValueError as exc:
        logger.warning(
            "[%s] warmup decode raised ValueError: %s (continuing)", SERVER_LOG_TAG, exc
        )
        return
    logger.info("[%s] warmup done in %.0f ms", SERVER_LOG_TAG, (time.monotonic() - t0) * 1000.0)


async def serve_forever(args: argparse.Namespace) -> None:
    """Build the engine and service, bind the port, and run until SIGTERM/SIGINT."""
    import signal

    from websockets import serve

    # Before the engine: a misconfigured --record_dir should fail in seconds, not after
    # the weights are loaded.
    recorder = _build_recorder(args)

    rvq_bundle = None
    centroids = None
    if args.action_tokenizer_bundle:
        from lightnav.traj_vocab import load_rvq_bundle

        rvq_bundle = load_rvq_bundle(
            Path(args.action_tokenizer_bundle),
            args.horizon,
            num_frames=0,
            load_cluster_ids=False,
        )

    # Engine first: the grounding-prefix probe needs its tokenizer.
    engine, bundle = _build_engine(args)

    grounding_tokens = probe_grounding_tokens(bundle.tokenizer)
    decode_budget = decode_token_budget(grounding_tokens, rvq_bundle)
    # Loose, never truncating: the exact cap only saves one decode step.
    max_new_tokens = max(int(args.max_new_tokens), decode_budget)
    logger.info(
        "[%s] decode cap: %d tokens (= %d grounding (ckpt vocab probe) + %d action token(s); "
        "--max_new_tokens=%d)",
        SERVER_LOG_TAG,
        max_new_tokens,
        grounding_tokens,
        action_token_count(rvq_bundle),
        args.max_new_tokens,
    )
    if rvq_bundle is None:
        centroids = load_centroids(args.traj_vocab_path, args.K, args.horizon)

    service = BatchedTrackingService(
        engine=engine,
        bundle=bundle,
        centroids=centroids,
        num_history_frames=int(args.num_history_frames or bundle.num_history_frames),
        max_new_tokens=max_new_tokens,
        max_batch_size=int(args.max_batch_size),
        max_wait_ms=float(args.max_wait_ms),
        serve_task=args.task,
        rvq_bundle=rvq_bundle,
    )
    await service.start()
    ready_file = Path(args.ready_file) if args.ready_file else None
    try:
        if not args.no_warmup:
            await _warmup(service, bundle)
        handler = make_handler(service, recorder=recorder)

        # Orchestrators stop the process with SIGTERM; handle it (and SIGINT) so the
        # listening socket closes cleanly and the ready file is removed.
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. restricted event loops
                pass

        server = await serve(handler, args.host, args.port, max_size=64 * 1024 * 1024)
        try:
            _log_ready(args, recorder)
            if ready_file is not None:
                ready_file.parent.mkdir(parents=True, exist_ok=True)
                ready_file.touch()
            await stop.wait()
            logger.info("[%s] shutdown signal received", SERVER_LOG_TAG)
        finally:
            if ready_file is not None:
                try:
                    ready_file.unlink(missing_ok=True)
                except OSError:
                    pass
            server.close()
            await server.wait_closed()
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                logger.warning(
                    "[%s] recorder shutdown failed: %s: %s", SERVER_LOG_TAG, type(exc).__name__, exc
                )
        await service.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WebSocket inference server for trajectory-token checkpoints."
    )
    parser.add_argument("--model_path", required=True, help="HF checkpoint directory.")
    parser.add_argument(
        "--traj_vocab_path",
        default=None,
        help="Directory containing centroids_whole_chunk_K{K}_h{H}.npy, or the .npy itself "
        "(flat action tokenizer; mutually exclusive with --action_tokenizer_bundle). "
        "Optional when the checkpoint ships / references its decoder in eval_config.json.",
    )
    parser.add_argument(
        "--action_tokenizer_bundle",
        default=os.environ.get("ACTION_TOKENIZER_BUNDLE") or None,
        help="RVQ bundle dir (manifest.json + codebooks) to serve an RVQ action-tokenizer "
        "checkpoint. Mutually exclusive with --traj_vocab_path. Env: ACTION_TOKENIZER_BUNDLE.",
    )
    parser.add_argument("--K", type=int, default=DEFAULT_TRAJ_K, help="Trajectory vocab size.")
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_TRAJ_HORIZON, help="Waypoints per chunk (H)."
    )
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK", "tracking"),
        choices=["tracking", "vln"],
        help="Prompt family to serve: tracking (target following) or vln (instruction following). "
        "Env: TASK.",
    )
    parser.add_argument(
        "--backend",
        default="vllm_local",
        choices=["hf", "vllm_local"],
        help="vllm_local = in-process vLLM (default, faster per step); "
        "hf = transformers.generate (lighter, useful for debugging).",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for the hf backend.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization (vllm_local only). 0.85 suits one engine per GPU.",
    )
    parser.add_argument(
        "--quantization",
        choices=["fp8_llm_only"],
        default=(os.environ.get("VLLM_QUANT") or None),
        help="Weight quantization (vllm_local only). Default bf16. 'fp8_llm_only' "
        "quantizes the LLM to fp8 and keeps the ViT bf16 (~1.5x on Jetson Thor). "
        "On SM 11.0 also set VLLM_DISABLED_KERNELS; scripts/serve_thor.sh does both. "
        "Env: VLLM_QUANT.",
    )
    parser.add_argument(
        "--vit_cache_entries",
        type=int,
        default=_int_env("VLN_VIT_CACHE_ENTRIES"),
        help="ViT tubelet LRU capacity (vllm_local; speed only, never output). "
        "Default: auto (512 with SlowFast). Long robot sessions want 1024. "
        "Env: VLN_VIT_CACHE_ENTRIES.",
    )
    parser.add_argument(
        "--num_history_frames",
        type=int,
        default=None,
        help="Override the history window. Default: the checkpoint's eval_config.json.",
    )
    parser.add_argument(
        "--pool_spatial",
        type=int,
        default=None,
        help="Override the post-ViT spatial pooling factor (default: checkpoint eval_config.json).",
    )
    parser.add_argument(
        "--aspect_mode",
        choices=["stretch", "keep"],
        default=os.environ.get("ASPECT_MODE", "stretch"),
        help="How client frames with another aspect ratio are fitted to the checkpoint's "
        "video_size: stretch (default; what the checkpoints were trained with) or keep "
        "(per-session size with the camera's aspect ratio at the same pixel budget, e.g. a 4:3 "
        "camera -> 288x384 for a 256x448 checkpoint). Env: ASPECT_MODE.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Lower bound on the per-step decode cap; raised automatically to fit the "
        "checkpoint's grounding + action tokens.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Env: HOST.")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "8050")), help="Env: PORT."
    )
    parser.add_argument(
        "--ready_file", default=None, help="Touch this path once the server is listening."
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=int(os.environ.get("MAX_BATCH_SIZE", "8")),
        help="Micro-batch width: sessions decoded together per tick (also the vLLM "
        "max_num_seqs). Env: MAX_BATCH_SIZE.",
    )
    parser.add_argument(
        "--max_wait_ms",
        type=float,
        default=float(os.environ.get("MAX_WAIT_MS", "8")),
        help="Max time to wait filling a batch before running it. Env: MAX_WAIT_MS.",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Skip the one-shot synthetic warmup inference before binding the port.",
    )

    # Local episode recording (docs/VISUALIZATION.md). Off unless --record_dir is set; the
    # server never renders video online -- run `lightnav-render <record_dir>` afterwards.
    rec = parser.add_argument_group("recording")
    rec.add_argument(
        "--record_dir",
        default=os.environ.get("RECORD_DIR", ""),
        help="Record every connection's episodes (frames + per-step JSON) under this "
        "directory for `lightnav-render`. Empty (default) = off. Env: RECORD_DIR.",
    )
    rec.add_argument(
        "--record_fps",
        type=int,
        default=int(os.environ.get("RECORD_FPS", "10")),
        help="Frame rate written into the recording manifest (used by lightnav-render). "
        "Env: RECORD_FPS.",
    )
    rec.add_argument(
        "--record_timeline",
        default=os.environ.get("RECORD_TIMELINE", "realtime"),
        choices=["realtime", "per_step"],
        help="Default video timebase for the recording: realtime (steps repeat by wall-clock "
        "gap) or per_step (one frame per step). Env: RECORD_TIMELINE.",
    )
    rec.add_argument(
        "--record_images",
        dest="record_images",
        action="store_true",
        default=_env_flag("RECORD_IMAGES", True),
        help="Store the client's JPEG frames alongside the records (default on). "
        "Env: RECORD_IMAGES=0/1.",
    )
    rec.add_argument(
        "--no_record_images",
        dest="record_images",
        action="store_false",
        help="Record the per-step JSON only (no frames; the video cannot be rendered).",
    )
    rec.add_argument(
        "--cam_hfov_deg",
        type=float,
        default=float(os.environ.get("CAM_HFOV_DEG", "90.0")),
        help="Horizontal FOV (degrees) of the CLIENT camera; only the trajectory-ribbon "
        "projection in the rendered video uses it. Env: CAM_HFOV_DEG.",
    )
    rec.add_argument(
        "--cam_height",
        type=float,
        default=float(os.environ.get("CAM_HEIGHT", "0.5")),
        help="Height of the client camera above the floor (m), for the ribbon projection. "
        "Env: CAM_HEIGHT.",
    )
    rec.add_argument(
        "--traj_forward_offset",
        type=float,
        default=_optional_float_env("TRAJ_FORWARD_OFFSET"),
        help="Forward displacement (m) applied to the ribbon before projection; default "
        "unset = automatic (the depth of the image's bottom edge). Env: TRAJ_FORWARD_OFFSET.",
    )
    rec.add_argument(
        "--waypoint_dt_s",
        type=float,
        default=float(os.environ.get("WAYPOINT_DT_S", "0.1")),
        help="Seconds per waypoint row assumed by the HUD velocity readout only. "
        "Env: WAYPOINT_DT_S.",
    )
    return parser


def resolve_decoder_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in ``--traj_vocab_path`` / ``--action_tokenizer_bundle`` from the checkpoint.

    Exactly one decoder must end up set. When neither flag is given the checkpoint's
    ``eval_config.json`` snapshot (or a sibling ``action_tokenizer/`` / ``traj_vocab/``
    directory) is used, so a checkpoint that ships its decoder needs no extra flags.
    Raises ``ValueError`` when both flags are given or nothing can be resolved.
    """
    if args.traj_vocab_path and args.action_tokenizer_bundle:
        raise ValueError("pass only one of --traj_vocab_path (flat) or --action_tokenizer_bundle (rvq)")
    if args.traj_vocab_path or args.action_tokenizer_bundle:
        return args
    from lightnav.tracking import resolve_action_decoder_from_config

    task_key = _engine_task_for_server_task(args.task)
    resolved = resolve_action_decoder_from_config(args.model_path, task_key)
    if resolved is None:
        raise ValueError(
            "no action decoder found for this checkpoint: pass --traj_vocab_path (+ --K/--horizon) "
            "or --action_tokenizer_bundle, or ship it next to the weights as referenced by "
            "eval_config.json"
        )
    if resolved["method"] == "rvq":
        args.action_tokenizer_bundle = str(resolved["bundle_path"])
    else:
        args.traj_vocab_path = str(resolved["traj_vocab_path"])
        args.K = int(resolved["K"])
    args.horizon = int(resolved["horizon"])
    logger.info("[%s] action decoder resolved from the checkpoint: %s", SERVER_LOG_TAG, resolved)
    return args


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        resolve_decoder_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    import torch

    torch.set_grad_enabled(False)

    try:
        asyncio.run(serve_forever(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
