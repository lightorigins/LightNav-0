"""BatchedTrackingService._infer_batch with fake engines: sample-builder dispatch, the
ViT -> LLM -> decode chain, protocol signals, timings keys, the hf branch and per-session
decode isolation. CPU-only."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from lightnav.inference.engine import VitResult
from lightnav.serving import tracking_service as svc_mod
from lightnav.serving.tracking_service import BatchedTrackingService, TrackingPrediction

FIXED_TIMING_KEYS = {
    "batch_size",
    "queue_wait_ms",
    "build_sample_ms",
    "vit_ms",
    "llm_ms",
    "decode_waypoints_ms",
    "batch_total_ms",
}


class _FakeEngine:
    def vit_forward_batch(self, samples, vit_caches=None):
        assert samples == [{"video": "tensor-a"}, {"video": "tensor-b"}]
        assert vit_caches == ["cache-a", "cache-b"]
        return [
            VitResult(
                prompt_ids=[1],
                video_embeds=None,
                video_grid_thw=None,
                timings={
                    "process_sample_ms": 10.0,
                    "prepare_inputs_ms": 3.0,
                    "post_vit_pool_ms": 2.0,
                    "cpu_copy_ms": 4.0,
                    "vit_cache_hits": 30.0,
                    "vit_cache_misses": 2.0,
                    "vit_cache_size": 32.0,
                },
            ),
            VitResult(
                prompt_ids=[2],
                video_embeds=None,
                video_grid_thw=None,
                timings={
                    "process_sample_ms": 14.0,
                    "prepare_inputs_ms": 5.0,
                    "post_vit_pool_ms": 6.0,
                    "cpu_copy_ms": 8.0,
                    "vit_cache_hits": 29.0,
                    "vit_cache_misses": 3.0,
                    "vit_cache_size": 32.0,
                },
            ),
        ]

    def llm_generate_batch(self, items, max_new_tokens):
        assert len(items) == 2
        assert max_new_tokens == 3
        return ["<traj_1>", "<traj_2>"]


class _FakeSession:
    def __init__(self, name: str):
        self.name = name
        self.instruction = f"instruction-{name}"
        self._history_frame_ids = [1, 2, 3]
        self._vit_cache = f"cache-{name}"
        self._buffer_len = 3
        self.K = 4

    def _get_video_tensor(self):
        return f"tensor-{self.name}"

    def decode_waypoints(self, text: str):
        return np.array([[1.0, 2.0, 3.0]], dtype=np.float32), text


def _service(engine, **overrides) -> BatchedTrackingService:
    kwargs = dict(
        engine=engine,
        bundle=object(),
        centroids=np.zeros((4, 1, 3), dtype=np.float32),
        num_history_frames=64,
        max_new_tokens=3,
        max_batch_size=4,
        max_wait_ms=1,
    )
    kwargs.update(overrides)
    return BatchedTrackingService(**kwargs)


def test_infer_batch_returns_prediction_timings(monkeypatch):
    def fake_build_tracking_sample(video, instruction, frame_ids, bundle):
        assert instruction.startswith("instruction-")
        assert frame_ids == [1, 2, 3]
        return {"video": video}

    monkeypatch.setattr(svc_mod, "build_tracking_sample", fake_build_tracking_sample)
    service = _service(_FakeEngine())
    now = time.monotonic()

    results = service._infer_batch(
        [
            svc_mod._PredictRequest(_FakeSession("a"), submitted_at=now - 0.020),
            svc_mod._PredictRequest(_FakeSession("b"), submitted_at=now - 0.010),
        ]
    )

    assert all(isinstance(r, TrackingPrediction) for r in results)
    assert [r.stop for r in results] == [False, False]
    assert [r.visible for r in results] == [None, None]
    assert [r.traj_id for r in results] == [1, 2]
    assert [r.raw_text for r in results] == ["<traj_1>", "<traj_2>"]
    assert results[0].waypoints.shape == (1, 3)
    timings = results[0].timings_ms
    assert FIXED_TIMING_KEYS <= set(timings)
    assert timings["batch_size"] == 2.0
    assert timings["queue_wait_ms"] >= 15.0
    assert timings["build_sample_ms"] >= 0.0
    assert timings["vit_ms"] >= 0.0
    assert timings["llm_ms"] >= 0.0
    assert timings["decode_waypoints_ms"] >= 0.0
    assert timings["batch_total_ms"] >= timings["llm_ms"]
    # VitResult timings pass through under their own names.
    assert timings["process_sample_ms"] == 10.0
    assert timings["vit_cache_hits"] == 30.0
    assert results[1].timings_ms["vit_cache_misses"] == 3.0
    assert results[1].timings_ms["queue_wait_ms"] < timings["queue_wait_ms"]


def test_infer_batch_uses_vln_sample_builder_and_exposes_protocol_signals(monkeypatch):
    calls = []

    class FakeEngine:
        def vit_forward_batch(self, samples, vit_caches=None):
            assert samples == [{"task": "vln", "video": "tensor-a"}]
            assert vit_caches == ["cache-a"]
            return [
                VitResult(
                    prompt_ids=[1],
                    video_embeds=None,
                    video_grid_thw=None,
                    timings={"process_sample_ms": 1.0},
                )
            ]

        def llm_generate_batch(self, items, max_new_tokens):
            assert len(items) == 1
            assert max_new_tokens == 3
            return ["<tpos_0><traj_0>"]

    class FakeSession:
        instruction = "go to the chair"
        _history_frame_ids = [4, 5]
        _vit_cache = "cache-a"
        K = 4

        def _get_video_tensor(self):
            return "tensor-a"

        def decode_waypoints(self, text: str):
            assert text == "<tpos_0><traj_0>"
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32), text

    def fake_build_tracking_sample(video, instruction, frame_ids, bundle):
        raise AssertionError("tracking sample builder must not be used for serve_task='vln'")

    def fake_build_vln_traj_sample(video, instruction, frame_ids, bundle):
        calls.append((video, instruction, frame_ids, bundle))
        return {"task": "vln", "video": video}

    monkeypatch.setattr(svc_mod, "build_tracking_sample", fake_build_tracking_sample)
    monkeypatch.setattr(svc_mod, "build_vln_traj_sample", fake_build_vln_traj_sample)

    bundle = object()
    service = _service(FakeEngine(), bundle=bundle, max_batch_size=1, serve_task="vln")

    result = service._infer_batch(
        [svc_mod._PredictRequest(FakeSession(), submitted_at=time.monotonic())]
    )[0]

    assert calls == [("tensor-a", "go to the chair", [4, 5], bundle)]
    assert result.stop is True
    assert result.visible is False
    assert result.traj_id == 0
    assert result.tpos_id == 0
    assert result.waypoints.tolist() == [[0.0, 0.0, 0.0]]


def test_vln_sample_carries_optional_prompt_pointing(monkeypatch):
    monkeypatch.setattr(
        svc_mod,
        "build_vln_traj_sample",
        lambda video, instruction, frame_ids, bundle: {"video": video},
    )
    service = _service(_FakeEngine(), serve_task="vln")
    session = _FakeSession("a")
    session.prompt_pointing_ids = (1273, 1223)
    assert service._build_sample_for_session(session) == {
        "video": "tensor-a",
        "_prompt_pointing_ids": (1273, 1223),
    }


def test_vln_sample_carries_staged_opos(monkeypatch):
    monkeypatch.setattr(
        svc_mod,
        "build_vln_traj_sample",
        lambda video, instruction, frame_ids, bundle: {"video": video},
    )
    service = _service(_FakeEngine(), serve_task="vln")
    session = _FakeSession("a")
    session.prompt_opos_id = 1223
    assert service._build_sample_for_session(session) == {
        "video": "tensor-a",
        "_prompt_opos_id": 1223,
        "_action_token_count": 1,
    }


def test_serve_task_is_validated():
    with pytest.raises(ValueError, match="serve_task"):
        _service(_FakeEngine(), serve_task="objectnav")


def test_make_session_attaches_optional_client_id():
    service = _service(_FakeEngine())

    session = service.make_session(client_id="robot-client-7")

    assert session.client_id == "robot-client-7"
    assert service.make_session().client_id is None
    assert session.K == 4 and session.rvq is None


def test_make_session_uses_the_rvq_bundle_when_given():
    class _Rvq:
        levels = [2, 2]
        horizon = 10

    service = _service(_FakeEngine(), centroids=None, rvq_bundle=_Rvq())
    session = service.make_session()
    assert session.rvq is not None
    assert session.K is None
    assert session.H == 10


class _HfEngine:
    backend = "hf"

    def __init__(self, text: str = "<traj_1>"):
        self.calls = []
        self._text = text

    def vit_forward_batch(self, samples, vit_caches=None):
        raise AssertionError("hf backend must not use the batched ViT path")

    def llm_generate_batch(self, items, max_new_tokens):
        raise AssertionError("hf backend has no vLLM engine to batch on")

    def generate_from_frames(
        self,
        video,
        instruction,
        predict_horizon=1,
        frame_ids=None,
        max_new_tokens=None,
        task_type="tracking",
    ):
        self.calls.append((video, instruction, predict_horizon, frame_ids, max_new_tokens, task_type))
        return self._text, 5.0


@pytest.mark.parametrize(
    ("serve_task", "expected_task_type"), [("tracking", "tracking"), ("vln", "vlnce_traj")]
)
def test_hf_backend_generates_per_session_via_generate_from_frames(
    monkeypatch, serve_task, expected_task_type
):
    def no_builder(*args, **kwargs):
        raise AssertionError("hf branch must not build vLLM samples")

    monkeypatch.setattr(svc_mod, "build_tracking_sample", no_builder)
    monkeypatch.setattr(svc_mod, "build_vln_traj_sample", no_builder)
    engine = _HfEngine()
    service = _service(engine, serve_task=serve_task)
    now = time.monotonic()

    results = service._infer_batch(
        [
            svc_mod._PredictRequest(_FakeSession("a"), submitted_at=now),
            svc_mod._PredictRequest(_FakeSession("b"), submitted_at=now),
        ]
    )

    assert engine.calls == [
        ("tensor-a", "instruction-a", 1, [1, 2, 3], 3, expected_task_type),
        ("tensor-b", "instruction-b", 1, [1, 2, 3], 3, expected_task_type),
    ]
    assert [r.traj_id for r in results] == [1, 1]
    assert [r.stop for r in results] == [False, False]
    timings = results[0].timings_ms
    assert FIXED_TIMING_KEYS <= set(timings)
    assert timings["batch_size"] == 2.0
    assert timings["vit_ms"] == 0.0
    assert timings["llm_ms"] >= 0.0


def test_one_bad_decode_yields_an_exception_for_that_session_only(monkeypatch):
    monkeypatch.setattr(
        svc_mod, "build_tracking_sample", lambda video, instruction, frame_ids, bundle: {"video": video}
    )

    class BadDecodeSession(_FakeSession):
        def decode_waypoints(self, text: str):
            raise ValueError(f"traj id 999 out of vocab range from {text!r}")

    service = _service(_FakeEngine())
    now = time.monotonic()
    results = service._infer_batch(
        [
            svc_mod._PredictRequest(_FakeSession("a"), submitted_at=now),
            svc_mod._PredictRequest(BadDecodeSession("b"), submitted_at=now),
        ]
    )

    assert len(results) == 2
    assert isinstance(results[0], TrackingPrediction)
    assert results[0].traj_id == 1
    assert isinstance(results[1], ValueError)
    assert "out of vocab range" in str(results[1])


class _AnyBatchEngine:
    """Returns one result per sample whatever the batch composition."""

    def vit_forward_batch(self, samples, vit_caches=None):
        return [VitResult(prompt_ids=[1], video_embeds=None, video_grid_thw=None) for _ in samples]

    def llm_generate_batch(self, items, max_new_tokens):
        return ["<traj_1>"] * len(items)


async def test_predict_reraises_only_the_failing_sessions_decode_error(monkeypatch):
    monkeypatch.setattr(
        svc_mod, "build_tracking_sample", lambda video, instruction, frame_ids, bundle: {"video": video}
    )

    class BadDecodeSession(_FakeSession):
        def decode_waypoints(self, text: str):
            raise ValueError("boom")

    service = _service(_AnyBatchEngine(), max_wait_ms=20)
    await service.start()
    try:
        good, bad = _FakeSession("a"), BadDecodeSession("b")
        results = await asyncio.gather(
            service.predict(good), service.predict(bad), return_exceptions=True
        )
    finally:
        await service.stop()

    assert isinstance(results[0], TrackingPrediction)
    assert results[0].traj_id == 1
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "boom"


async def test_predict_rejects_an_empty_observation_buffer():
    service = _service(_AnyBatchEngine())
    session = _FakeSession("a")
    session._buffer_len = 0
    with pytest.raises(RuntimeError, match="empty observation buffer"):
        await service.predict(session)
