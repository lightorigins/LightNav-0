"""Runtime configuration for building an inference engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Flat inference config. Only ``model_path`` is required.

    Processing parameters (video size, pooling, fps, history window, slowfast
    tiers, action tokenizer) are read from the checkpoint's ``eval_config.json``;
    the ``pool_*`` / ``video_size`` / ``num_history_frames`` fields here are
    explicit overrides (``None`` = auto).
    """

    model_path: str = ""
    backend: str = "hf"  # "hf" | "vllm_local"
    max_new_tokens: int = 64
    device: str = "cuda"

    # vLLM engine sizing (vllm_local backend only).
    gpu_memory_utilization: float = 0.65
    max_num_seqs: int = 1  # vLLM batch width; >1 enables micro-batching.

    # Weight quantization for the vllm_local backend. None -> bf16 (default).
    # "fp8_llm_only" quantizes the LLM linears to fp8 and keeps the vision tower
    # in bf16 (see vllm_utils._patch_fp8_llm_only). Full-model "fp8" is refused:
    # quantizing the ViT corrupts perception and the embedding path reuses that
    # same tower. On SM 11.0 (Jetson Thor) also set VLLM_DISABLED_KERNELS — see
    # docs/JETSON_THOR.md; scripts/serve_thor.sh does both.
    quantization: str | None = None

    # ViT tubelet LRU capacity (vllm_local, speed only, never output). None ->
    # auto (512 with SlowFast). Long robot sessions want 1024; see JETSON_THOR.md.
    vit_cache_entries: int | None = None

    # HF backend attention kernel override (None -> LIGHTNAV_ATTN env or "sdpa").
    attn_implementation: str | None = None

    # How a client frame whose aspect ratio differs from video_size is fitted:
    # "stretch" (default, what the checkpoints were trained with) resizes to video_size
    # ignoring the aspect ratio; "keep" picks a per-session size with the source aspect
    # ratio at the same pixel budget (see frame_preprocessing.choose_video_size).
    aspect_mode: str = "stretch"

    # Processing overrides (None = auto from eval_config.json).
    video_size: list[int] | None = None
    pool_enable: bool | None = None
    pool_spatial: int | None = None
    pool_mode: str | None = None
    pool_stage: str | None = None
    num_history_frames: int | None = None
