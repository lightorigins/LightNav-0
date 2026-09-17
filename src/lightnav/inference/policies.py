"""Per-session frame buffer that feeds the inference engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from lightnav.inference.frame_preprocessing import choose_video_size, rgb_frame_to_model_tensor

if TYPE_CHECKING:
    from lightnav.inference.engine import VLNInferenceEngine


class NavigationPolicy:
    """
    Stateful frame buffer for online inference.

    Accumulates RGB frames in a history buffer (a ring of ``num_history_frames``
    frames, or the whole episode for SlowFast checkpoints), assigns absolute
    frame ids, and owns a per-session ViT tubelet cache. Subclasses (or callers)
    hand ``_get_video_tensor()`` / ``_history_frame_ids`` to the engine.
    """

    def __init__(
        self,
        engine: "VLNInferenceEngine",
        num_history_frames: int = 64,
        predict_horizon: int = 1,
    ):
        self.engine = engine
        self.num_history_frames = num_history_frames
        self.predict_horizon = predict_horizon
        # SlowFast keeps the FULL episode (span tier reaches frame 0), not a ring.
        self.slowfast = bool(getattr(getattr(engine, "bundle", None), "slowfast_tiers", None))
        # SlowFast: pre-allocated growing buffer (write-in-place) so _get_video_tensor
        # is an O(1) view, NOT an O(episode) torch.stack of a frame list every step.
        self._sf_buffer: torch.Tensor | None = None
        self._history: list[Any] = []
        self._history_frame_ids: list[int] = []
        self._video_buffer: torch.Tensor | None = None
        self._buffer_len: int = 0
        self._write_pos: int = 0
        self._next_frame_id: int = 0
        self.instruction = ""
        # 客户端可按帧提供输入 APOS/OPOS；None 表示保持原始模型执行。
        self.prompt_pointing_ids: tuple[int, int] | None = None
        # Opt-in staged mode: the model generates APOS, while the client supplies OPOS.
        self.prompt_opos_id: int | None = None
        self._vit_cache = engine.new_vit_cache() if hasattr(engine, "new_vit_cache") else None
        # Model-frame size of this session. "stretch": always the checkpoint's video_size.
        # "keep": chosen from the first frame's aspect ratio at the same pixel budget.
        self.aspect_mode = str(getattr(engine, "aspect_mode", "stretch"))
        self.video_size: tuple[int, int] | None = None

    def _size_multiple(self) -> int:
        """Frame sides must be multiples of patch*merge (32), and of the pre-ViT pooling
        factors on top of that (pre-ViT pooling needs a divisible merged grid)."""
        bundle = getattr(self.engine, "bundle", None)
        vp = getattr(getattr(bundle, "processor", None), "video_processor", None)
        patch = int(getattr(vp, "patch_size", 16)) if vp is not None else 16
        merge = int(getattr(vp, "merge_size", 2)) if vp is not None else 2
        multiple = patch * merge
        if getattr(bundle, "pool_stage", "post_vit") == "pre_vit":
            # The MERGED grid (side / (patch*merge)) must be divisible by each factor.
            factors = [int(getattr(bundle, "pool_spatial", 1) or 1)]
            for tier in getattr(bundle, "slowfast_tiers", None) or []:
                factors.append(int(tier.get("pool_spatial", 1) or 1))
            for f in factors:
                if f > 1:
                    multiple = math.lcm(multiple, patch * merge * f)
        return multiple

    def _resolve_video_size(self, rgb_frame) -> tuple[int, int]:
        if self.video_size is None:
            base = tuple(int(v) for v in self.engine.bundle.video_size)
            if self.aspect_mode == "keep":
                self.video_size = choose_video_size(
                    rgb_frame.shape[:2], base, multiple=self._size_multiple()
                )
            else:
                self.video_size = base
        return self.video_size

    def reset(self, instruction: str = ""):
        self._history.clear()
        self._history_frame_ids.clear()
        self._sf_buffer = None
        self._video_buffer = None
        self._buffer_len = 0
        self._write_pos = 0
        self._next_frame_id = 0
        self.instruction = instruction
        self.prompt_pointing_ids = None
        self.prompt_opos_id = None
        self.video_size = None
        if self._vit_cache is not None:
            self._vit_cache.clear()
        self.engine.reset_episode_state()

    def _convert_frame(self, rgb_frame) -> torch.Tensor:
        """Source frame -> model frame, via the one authoritative implementation.

        ``rgb_frame_to_model_tensor`` enforces the checkpoint's spatial contract,
        so a frame reaching the buffer below already matches ``video_size``. Do
        not resize anywhere else: a second implementation is how preprocessing
        drifts between entry points.
        """
        return rgb_frame_to_model_tensor(rgb_frame, self._resolve_video_size(rgb_frame))

    def observe(self, rgb_frame) -> None:
        """Add a frame to history without running inference."""
        self._history.append(rgb_frame.copy())
        self._history_frame_ids.append(self._next_frame_id)
        self._next_frame_id += 1

        frame_tensor = self._convert_frame(rgb_frame)

        if self.slowfast:
            # Keep the whole episode (frame i at index i) so the span tier can
            # reach frame 0; no ring truncation. Write in place into a growing
            # pre-allocated buffer (double on overflow).
            if self._sf_buffer is None:
                self._sf_buffer = torch.zeros(64, *frame_tensor.shape, dtype=frame_tensor.dtype)
            elif self._buffer_len >= self._sf_buffer.shape[0]:
                grown = torch.zeros(
                    self._sf_buffer.shape[0] * 2, *frame_tensor.shape, dtype=frame_tensor.dtype
                )
                grown[: self._buffer_len] = self._sf_buffer[: self._buffer_len]
                self._sf_buffer = grown
            self._sf_buffer[self._buffer_len] = frame_tensor
            self._buffer_len += 1  # universal "frames observed" counter
        else:
            if self._video_buffer is None:
                self._video_buffer = torch.zeros(
                    self.num_history_frames,
                    *frame_tensor.shape,
                    dtype=frame_tensor.dtype,
                )

            if self._buffer_len < self.num_history_frames:
                self._video_buffer[self._buffer_len] = frame_tensor
                self._buffer_len += 1
            else:
                self._video_buffer[self._write_pos] = frame_tensor
                self._write_pos = (self._write_pos + 1) % self.num_history_frames

            if len(self._history) > self.num_history_frames:
                self._history = self._history[-self.num_history_frames :]
                self._history_frame_ids = self._history_frame_ids[-self.num_history_frames :]

    def _get_video_tensor(self) -> torch.Tensor:
        """Return the current video frames in chronological order."""
        if self.slowfast:
            if self._sf_buffer is None or self._buffer_len == 0:
                raise RuntimeError("No frames observed")
            return self._sf_buffer[: self._buffer_len]  # O(1) view, full episode
        if self._video_buffer is None:
            raise RuntimeError("No frames observed")
        if self._buffer_len < self.num_history_frames:
            return self._video_buffer[: self._buffer_len]
        if self._write_pos == 0:
            return self._video_buffer
        return torch.cat(
            [
                self._video_buffer[self._write_pos :],
                self._video_buffer[: self._write_pos],
            ]
        )
