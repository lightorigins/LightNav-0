"""Shared-engine batched prediction service.

One engine + one action decoder (flat centroid table or RVQ bundle), fronted by
a :class:`MicroBatchScheduler`. Each WebSocket connection owns a
:class:`TrackingAgent` session (frame buffer + per-session ViT cache only);
``predict()`` enqueues the session and awaits its batched result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from lightnav.inference.samples import build_tracking_sample, build_vln_traj_sample
from lightnav.serving.batcher import MicroBatchScheduler
from lightnav.serving.protocol import decode_prediction_signals
from lightnav.serving.token_budget import action_token_count
from lightnav.tracking import TrackingAgent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PredictRequest:
    session: TrackingAgent
    submitted_at: float


@dataclass(frozen=True)
class TrackingPrediction:
    waypoints: np.ndarray
    stop: bool
    stop_reason: str | None
    visible: bool | None
    traj_id: int | None   # None for RVQ ckpts
    tpos_id: int | None
    timings_ms: dict[str, float]
    # Dual-pointing ckpts only (None otherwise): frame-pixel grounding ids.
    apos_id: int | None = None
    opos_id: int | None = None
    # The model's own output text for this step, e.g.
    # "<apos_500><opos_900><act_l0_1><act_l1_2><act_l2_3>".
    raw_text: str = ""


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class BatchedTrackingService:
    def __init__(
        self,
        engine,
        bundle,
        centroids: np.ndarray | None,
        num_history_frames: int,
        max_new_tokens: int,
        max_batch_size: int,
        max_wait_ms: float,
        serve_task: str = "tracking",
        *,
        rvq_bundle=None,
    ) -> None:
        if serve_task not in ("tracking", "vln"):
            raise ValueError(f"serve_task must be 'tracking' or 'vln', got {serve_task!r}")
        self.engine = engine
        self.bundle = bundle
        self.centroids = centroids
        self.rvq_bundle = rvq_bundle
        self.num_history_frames = num_history_frames
        self.max_new_tokens = max_new_tokens
        self.serve_task = serve_task
        self.scheduler = MicroBatchScheduler(
            self._infer_batch,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )

    async def start(self) -> None:
        await self.scheduler.start()

    async def stop(self) -> None:
        await self.scheduler.stop()

    def make_session(self, client_id: str | None = None) -> TrackingAgent:
        # The engine is used only for buffer/decode helpers; inference goes through
        # the service.
        if self.rvq_bundle is not None:
            return TrackingAgent(
                engine=self.engine,
                num_history_frames=self.num_history_frames,
                rvq_bundle=self.rvq_bundle,
                client_id=client_id,
            )
        return TrackingAgent(
            engine=self.engine,
            centroids=self.centroids,
            num_history_frames=self.num_history_frames,
            client_id=client_id,
        )

    def _engine_task_type(self) -> str:
        return "vlnce_traj" if self.serve_task == "vln" else "tracking"

    def _build_sample_for_session(self, session: TrackingAgent) -> dict:
        video = session._get_video_tensor()
        frame_ids = list(session._history_frame_ids)
        if self.serve_task == "vln":
            sample = build_vln_traj_sample(video, session.instruction, frame_ids, self.bundle)
        else:
            sample = build_tracking_sample(video, session.instruction, frame_ids, self.bundle)
        pointing = getattr(session, "prompt_pointing_ids", None)
        if pointing is not None:
            sample["_prompt_pointing_ids"] = pointing
        prompt_opos = getattr(session, "prompt_opos_id", None)
        if prompt_opos is not None:
            sample["_prompt_opos_id"] = int(prompt_opos)
            sample["_action_token_count"] = action_token_count(self.rvq_bundle)
        return sample

    def _infer_batch(self, requests: list[_PredictRequest]) -> list[TrackingPrediction | BaseException]:
        """Runs in the scheduler's executor thread.

        Returns one entry per session: a :class:`TrackingPrediction`, or the
        exception raised while decoding THAT session's output (so one malformed
        model output does not fail the other sessions in the batch; ``predict``
        re-raises it for the owning caller only).
        """
        batch_t0 = time.monotonic()
        sessions = [r.session for r in requests]
        n = len(sessions)
        queue_waits = [(batch_t0 - r.submitted_at) * 1000.0 for r in requests]

        if getattr(self.engine, "backend", None) == "hf":
            # transformers.generate path: ViT + LLM run inside one call per session,
            # so the two stages are not separable here (vit_ms is reported as 0).
            task_type = self._engine_task_type()
            build_sample_ms = [0.0] * n
            vit_ms_list = [0.0] * n
            llm_ms_list: list[float] = []
            passthrough: list[dict[str, float]] = []
            texts: list[str] = []
            for s in sessions:
                t_gen = time.monotonic()
                generate_kwargs = {}
                pointing = getattr(s, "prompt_pointing_ids", None)
                if pointing is not None:
                    generate_kwargs["prompt_pointing_ids"] = pointing
                prompt_opos = getattr(s, "prompt_opos_id", None)
                if prompt_opos is not None:
                    generate_kwargs["prompt_opos_id"] = int(prompt_opos)
                    generate_kwargs["action_token_count"] = action_token_count(self.rvq_bundle)
                text, _ = self.engine.generate_from_frames(
                    s._get_video_tensor(),
                    s.instruction,
                    predict_horizon=1,
                    frame_ids=list(s._history_frame_ids),
                    max_new_tokens=self.max_new_tokens,
                    task_type=task_type,
                    **generate_kwargs,
                )
                llm_ms_list.append((time.monotonic() - t_gen) * 1000.0)
                texts.append(text)
                passthrough.append({})
        else:
            # 1. Build a sample per session from its current buffer.
            samples = []
            build_sample_ms = []
            for s in sessions:
                t_build = time.monotonic()
                samples.append(self._build_sample_for_session(s))
                build_sample_ms.append((time.monotonic() - t_build) * 1000.0)
            # 2. ViT forward through each session's tubelet cache.
            t_vit = time.monotonic()
            vit_caches = [getattr(s, "_vit_cache", None) for s in sessions]
            vit_results = self.engine.vit_forward_batch(samples, vit_caches=vit_caches)
            vit_ms = (time.monotonic() - t_vit) * 1000.0
            # 3. Keep the legacy one-shot batch intact. Staged sessions use two
            # decode calls because vLLM 0.19 has no continuation/KV-cache API.
            staged = [
                i for i, s in enumerate(sessions)
                if getattr(s, "prompt_opos_id", None) is not None
            ]
            normal = [i for i in range(n) if i not in staged]
            texts = [""] * n
            llm_ms_list = [0.0] * n
            if normal:
                t_llm = time.monotonic()
                normal_texts = self.engine.llm_generate_batch(
                    [vit_results[i] for i in normal], self.max_new_tokens
                )
                normal_ms = (time.monotonic() - t_llm) * 1000.0
                for i, text in zip(normal, normal_texts):
                    texts[i] = text
                    llm_ms_list[i] = normal_ms
            if staged:
                t_llm = time.monotonic()
                staged_texts = self.engine.llm_generate_two_stage_batch(
                    [vit_results[i] for i in staged],
                    [int(getattr(sessions[i], "prompt_opos_id")) for i in staged],
                    self.max_new_tokens,
                    action_token_count=action_token_count(self.rvq_bundle),
                )
                staged_ms = (time.monotonic() - t_llm) * 1000.0
                for i, text in zip(staged, staged_texts):
                    texts[i] = text
                    llm_ms_list[i] = staged_ms
            vit_ms_list = [vit_ms] * n
            passthrough = [
                {k: float(v) for k, v in (getattr(r, "timings", None) or {}).items()}
                for r in vit_results
            ]

        # 4. Decode each output to waypoints (per-session error isolation).
        out: list[TrackingPrediction | BaseException] = []
        decode_ms: list[float] = []
        batch_size = float(n)
        for idx, (s, text) in enumerate(zip(sessions, texts)):
            t_decode = time.monotonic()
            try:
                waypoints, raw_text = s.decode_waypoints(text)
                signals = decode_prediction_signals(
                    raw_text,
                    vocab_size=getattr(s, "K", None),
                    is_rvq=getattr(s, "rvq", None) is not None,
                    waypoints=waypoints,
                    rvq_stop_l0=getattr(getattr(s, "rvq", None), "stop_l0", None),
                )
            except Exception as exc:
                decode_ms.append((time.monotonic() - t_decode) * 1000.0)
                logger.warning("decode failed for one session: %s (raw=%r)", exc, text)
                out.append(exc)
                continue
            one_decode_ms = (time.monotonic() - t_decode) * 1000.0
            decode_ms.append(one_decode_ms)
            timings = dict(passthrough[idx])
            timings.update(
                {
                    "batch_size": batch_size,
                    "queue_wait_ms": queue_waits[idx],
                    "build_sample_ms": build_sample_ms[idx],
                    "vit_ms": vit_ms_list[idx],
                    "llm_ms": llm_ms_list[idx],
                    "decode_waypoints_ms": one_decode_ms,
                    "batch_total_ms": 0.0,
                }
            )
            out.append(
                TrackingPrediction(
                    waypoints=waypoints,
                    stop=signals.stop,
                    stop_reason=signals.stop_reason,
                    visible=signals.visible,
                    traj_id=signals.traj_id,
                    tpos_id=signals.tpos_id,
                    apos_id=signals.apos_id,
                    opos_id=signals.opos_id,
                    raw_text=raw_text,
                    timings_ms=timings,
                )
            )
        batch_total_ms = (time.monotonic() - batch_t0) * 1000.0
        for pred in out:
            if isinstance(pred, TrackingPrediction):
                pred.timings_ms["batch_total_ms"] = batch_total_ms
        logger.info(
            "batch=%d queue_wait_max=%.0fms vit=%.0fms llm=%.0fms decode=%.0fms total=%.0fms",
            n,
            max(queue_waits) if queue_waits else 0.0,
            _mean(vit_ms_list),
            _mean(llm_ms_list),
            _mean(decode_ms),
            batch_total_ms,
        )
        return out

    async def predict(self, session: TrackingAgent) -> TrackingPrediction:
        """Enqueue ``session`` for batched inference and await its prediction.

        Re-raises the per-session decode error when this session's model output
        could not be decoded; other sessions in the same batch are unaffected.
        """
        if session._buffer_len == 0:
            raise RuntimeError("predict called with empty observation buffer")
        result = await self.scheduler.submit(_PredictRequest(session, time.monotonic()))
        if isinstance(result, BaseException):
            raise result
        return result
