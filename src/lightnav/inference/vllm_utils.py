"""vLLM helpers: version guard, pre-computed-embedding monkeypatch, in-process engine access."""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from lightnav.inference.config import InferenceConfig
from lightnav.inference.model import resolve_model_paths

# The embedding/mrope monkeypatch below binds vLLM private internals that are
# specific to the 0.19.x line (BaseRenderer._process_multimodal, the
# vllm.inputs.engine.MultiModalInput TypedDict, MultiModalKwargsItems.from_hf_inputs,
# Qwen3VLForConditionalGeneration.get_mrope_input_positions). These move across minor
# versions and can drift across patch releases. Verified end-to-end against vllm 0.19.1
# / transformers 5.8.0. Set LIGHTNAV_SKIP_VERSION_GUARD=1 to bypass at your own risk.
_VLLM_SUPPORTED_PREFIX = "0.19."
_TRANSFORMERS_SUPPORTED_PREFIX = "5.8."


def _assert_vllm_version() -> None:
    """Fail fast with an actionable message if vllm/transformers are outside the
    tested range, instead of crashing deep inside a private-API monkeypatch."""
    if os.environ.get("LIGHTNAV_SKIP_VERSION_GUARD", "0").lower() in ("1", "true", "yes"):
        return
    from importlib.metadata import PackageNotFoundError, version

    for pkg, prefix in (
        ("vllm", _VLLM_SUPPORTED_PREFIX),
        ("transformers", _TRANSFORMERS_SUPPORTED_PREFIX),
    ):
        try:
            found = version(pkg)
        except PackageNotFoundError:
            raise RuntimeError(
                f"lightnav vllm_local backend requires {pkg} {prefix}x but {pkg} is not "
                f"installed. Install with: pip install -e '.[vllm]'"
            ) from None
        if not found.startswith(prefix):
            raise RuntimeError(
                f"lightnav's vLLM embedding monkeypatch targets {pkg} {prefix}x internals, "
                f"but found {pkg}=={found}. This version is untested and the patch WILL likely "
                f"break (it binds private APIs). Pin {pkg} to the verified version, or set "
                f"LIGHTNAV_SKIP_VERSION_GUARD=1 to override at your own risk."
            )


def apply_vllm_embedding_monkeypatch():
    """
    Fix vLLM Qwen3-VL: when passing pre-computed vision embeddings instead of
    raw video, the renderer's multimodal processing crashes. This patch
    bypasses the multimodal processor for embedding dict inputs and constructs
    the multimodal input manually from prompt_token_ids + video_embeds.

    Must be called before creating the vLLM LLM engine.
    """
    _assert_vllm_version()

    import hashlib
    import types

    # vllm 0.19.1 moved multimodal processing from InputPreprocessor to the
    # renderer: LLM.generate now calls BaseRenderer._process_multimodal (which
    # delegates to mm_processor.apply), so the embedding bypass must patch THAT.
    # Return type is vllm.inputs.engine.MultiModalInput (TypedDict).
    from vllm.inputs.engine import MultiModalInput
    from vllm.multimodal.inputs import (
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.renderers.base import BaseRenderer

    _original = BaseRenderer._process_multimodal

    # BaseRenderer._process_multimodal(self, prompt, mm_data, mm_uuids,
    # mm_processor_kwargs, tokenization_kwargs) -> MultiModalInput. self is a
    # BaseRenderer, which carries .model_config.
    def _patched(self, prompt, mm_data, mm_uuids, mm_processor_kwargs, tokenization_kwargs):
        video_data = mm_data.get("video") if mm_data else None
        is_dict_embed = isinstance(video_data, dict) and "video_embeds" in video_data
        if not (is_dict_embed and isinstance(prompt, list)):
            return _original(self, prompt, mm_data, mm_uuids, mm_processor_kwargs, tokenization_kwargs)

        prompt_ids = list(prompt)
        video_embeds = video_data["video_embeds"]
        video_grid_thw = video_data["video_grid_thw"]

        hf_config = self.model_config.hf_config
        video_token_id = hf_config.video_token_id
        merge_size = hf_config.vision_config.spatial_merge_size

        placeholders: list[PlaceholderRange] = []
        i = 0
        while i < len(prompt_ids):
            if prompt_ids[i] == video_token_id:
                start = i
                while i < len(prompt_ids) and prompt_ids[i] == video_token_id:
                    i += 1
                placeholders.append(PlaceholderRange(offset=start, length=i - start))
            else:
                i += 1

        grid_sizes = video_grid_thw.prod(dim=-1)
        embed_sizes = (grid_sizes // (merge_size * merge_size)).tolist()
        num_videos = video_grid_thw.shape[0]

        video_placeholders: list[PlaceholderRange] = []
        run_idx = 0
        for vid_idx in range(num_videos):
            expected = embed_sizes[vid_idx]
            collected = 0
            first_offset = None
            consumed_runs: list[PlaceholderRange] = []
            while collected < expected and run_idx < len(placeholders):
                pr = placeholders[run_idx]
                if first_offset is None:
                    first_offset = pr.offset
                collected += pr.length
                consumed_runs.append(pr)
                run_idx += 1
            if first_offset is not None and consumed_runs:
                last_pr = consumed_runs[-1]
                full_span = (last_pr.offset + last_pr.length) - first_offset
                if full_span == collected:
                    video_placeholders.append(PlaceholderRange(offset=first_offset, length=collected))
                else:
                    is_embed = torch.zeros(full_span, dtype=torch.bool)
                    for pr in consumed_runs:
                        rel_start = pr.offset - first_offset
                        is_embed[rel_start : rel_start + pr.length] = True
                    video_placeholders.append(
                        PlaceholderRange(offset=first_offset, length=full_span, is_embed=is_embed)
                    )

        from transformers.feature_extraction_utils import BatchFeature

        hf_inputs = BatchFeature(
            {
                "video_embeds": video_embeds,
                "video_grid_thw": video_grid_thw,
            }
        )
        video_embed_grid_sizes = grid_sizes // merge_size // merge_size
        fields_config = {
            "video_embeds": MultiModalFieldConfig.flat_from_sizes("video", video_embed_grid_sizes),
            "video_grid_thw": MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        }
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields_config)
        video_hashes = [
            hashlib.sha256(f"embed_video_{v}_{time.monotonic()}".encode()).hexdigest()[:16]
            for v in range(num_videos)
        ]
        return MultiModalInput(
            type="multimodal",
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes={"video": video_hashes},
            mm_placeholders={"video": video_placeholders},
        )

    BaseRenderer._process_multimodal = _patched
    print("[lightnav] vLLM BaseRenderer._process_multimodal patched for embedding dict path")

    # vllm 0.19.1 recomputes mrope on the worker via
    # Qwen3VLForConditionalGeneration.get_mrope_input_positions -> _iter_mm_grid_hw,
    # which scans for a <|vision_start|> per video FRAME. Our prompts wrap each video
    # with a single per-video <|vision_start|>...<|vision_end|> (the training
    # convention), so that per-frame scan raises "vision_start not in list". Replace
    # it with the HF Qwen3VLModel.get_rope_index, which locates vision spans from
    # mm_token_type_ids (image==1 / video==2) and matches our tokenization.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _HFQwen3VLModel
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as _VllmQ3VL

    def _patched_get_mrope_input_positions(self, input_tokens, mm_features):
        cfg = self.config
        image_token_id = getattr(cfg, "image_token_id", None)
        video_token_id = getattr(cfg, "video_token_id", None)
        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2

        img = [f.data["image_grid_thw"].data.reshape(-1) for f in mm_features if f.modality == "image"]
        vid = [f.data["video_grid_thw"].data.reshape(-1) for f in mm_features if f.modality == "video"]
        image_grid_thw = torch.stack(img) if img else None
        video_grid_thw = torch.stack(vid) if vid else None

        # HF get_rope_index needs self.config + self.get_vision_position_ids (stateless).
        fake = types.SimpleNamespace(config=cfg)
        fake.get_vision_position_ids = types.MethodType(_HFQwen3VLModel.get_vision_position_ids, fake)
        position_ids, mrope_delta = _HFQwen3VLModel.get_rope_index(
            fake, input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )
        delta = mrope_delta.flatten()[0].item() if torch.is_tensor(mrope_delta) else int(mrope_delta)
        # vllm expects (mrope_positions [3, seq], mrope_position_delta int).
        return position_ids[:, 0, :].to(torch.long), int(delta)

    _VllmQ3VL.get_mrope_input_positions = _patched_get_mrope_input_positions
    print("[lightnav] vLLM get_mrope_input_positions patched for single-vision_start video format")


def get_vllm_model(llm) -> torch.nn.Module:
    """Extract the underlying nn.Module from a vLLM LLM.

    Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so the model lives in-process.
    Handles vLLM V1 CUDAGraphWrapper by calling .unwrap() if present.
    """
    paths = [
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
    ]
    for dotpath in paths:
        obj = llm
        try:
            for attr in dotpath.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        else:
            if isinstance(obj, torch.nn.Module):
                return obj
            if hasattr(obj, "unwrap"):
                unwrapped = obj.unwrap()
                if isinstance(unwrapped, torch.nn.Module):
                    return unwrapped
    raise RuntimeError(
        "Cannot access vLLM internal model. Make sure VLLM_ENABLE_V1_MULTIPROCESSING=0 is set "
        "before any vllm import."
    )


def load_vllm_engine(config: InferenceConfig, num_frames: int = 64) -> Any:
    """Load a local vLLM LLM engine (bf16, pre-computed video embeddings enabled).

    ``num_frames`` sizes the worst-case video the engine profiles for memory
    (encoder cache + dummy ViT pass). Setting it to the actual history window
    instead of an arbitrary large value frees a large slice of GPU memory per
    instance, which is what allows several servers to share one GPU.
    """
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    from vllm import LLM

    config_dir, _, _ = resolve_model_paths(config.model_path)
    gpu_mem_util = float(getattr(config, "gpu_memory_utilization", 0.65) or 0.65)
    max_num_seqs = int(getattr(config, "max_num_seqs", 1) or 1)
    # KV cache scales with batch width. ~150 KiB/token (bf16, 36 layers, 8 KV
    # heads, head_dim 128) x max_model_len 2048 x max_num_seqs, floored at 2 GiB.
    # Override with VLN_KV_CACHE_GIB.
    kv_gib_env = os.environ.get("VLN_KV_CACHE_GIB")
    if kv_gib_env:
        kv_cache_bytes = int(float(kv_gib_env) * 1024**3)
    else:
        kv_cache_bytes = max(2 * 1024**3, max_num_seqs * 2048 * 150_000)
    async_scheduling = os.environ.get("VLN_VLLM_ASYNC_SCHEDULING", "0").lower() in (
        "1", "true", "yes"
    )
    enforce_eager = os.environ.get("VLN_VLLM_ENFORCE_EAGER", "0").lower() in (
        "1", "true", "yes"
    )
    # Eager mode alone does not disable vLLM's O2 custom RMSNorm path.  That
    # path can segfault in the native kernel on some Blackwell/driver
    # combinations during the startup profile, before Python can recover.
    # Keep the fast O2 path by default, but make the documented eager escape
    # hatch safe by falling back to PyTorch's native normalization as well.
    compilation_config = None
    optimization_level = None
    if enforce_eager:
        optimization_level = 0
        compilation_config = {
            # O0/eager is the recovery path for GPUs where vLLM's native
            # CUDA extensions fault during startup.  Disabling all custom
            # ops is intentionally conservative; the normal O2 path is
            # unchanged when VLN_VLLM_ENFORCE_EAGER is unset.
            "custom_ops": ["none"],
            "pass_config": {
                "fuse_norm_quant": False,
                "fuse_act_quant": False,
            },
        }
    # FlashAttention is vLLM's default on recent NVIDIA GPUs, but its
    # split-KV kernel can be unstable on some driver/GPU combinations (the
    # failure is a native cuLaunchKernel segfault, so Python cannot recover).
    # Keep the default automatic and expose a narrow escape hatch for a
    # Triton/Flex fallback without changing the normal benchmark path.
    attention_backend_env = os.environ.get("VLN_VLLM_ATTENTION_BACKEND", "").strip()
    # The eager recovery path also avoids FlashInfer's JIT module loader, which
    # can segfault on the same driver/GPU combinations as the native norm op.
    # An explicit environment override still wins.
    attention_backend = attention_backend_env or ("TRITON_ATTN" if enforce_eager else None)

    print(
        f"[lightnav] Loading vLLM engine from: {config_dir} (gpu_mem={gpu_mem_util}, "
        f"max_num_seqs={max_num_seqs}, kv_cache_gib={kv_cache_bytes / 1024**3:.1f}, "
        f"num_frames={num_frames}, eager={enforce_eager})"
    )
    engine_overrides: dict[str, Any] = {"compilation_config": compilation_config}
    if optimization_level is not None:
        engine_overrides["optimization_level"] = optimization_level

    llm = LLM(
        model=config_dir,
        dtype="bfloat16",
        quantization=None,
        # Size to the history window: vision tokens scale with num_frames (~12-20
        # pooled tokens/frame) + long instructions (300-600 tokens) + template
        # overhead. A hardcoded 2048 rejects 128-frame prompts, so scale with
        # num_frames (floor 2048 so <=64-frame windows are unchanged); 64->2560, 128->4096.
        max_model_len=max(2048, num_frames * 24 + 1024),
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
        # CUDA graphs (enforce_eager=False) cut decode latency several-fold; prefill
        # is unaffected and greedy output is identical. Set VLN_VLLM_ENFORCE_EAGER=1
        # to revert (faster startup / lower capture memory) if a tight
        # gpu_memory_utilization OOMs during graph capture.
        enforce_eager=enforce_eager,
        **engine_overrides,
        # The custom multimodal embedding path reuses host/device buffers across
        # requests. Keep vLLM's CUDA-event scheduler synchronous by default; the
        # async path can turn a delayed device error into a native segfault.
        async_scheduling=async_scheduling,
        # Optional vLLM AttentionBackendEnum name, e.g. TRITON_ATTN.  Leaving
        # this unset preserves vLLM's automatic backend selection.
        attention_backend=attention_backend,
        # Vision Transformers have a separate backend selector in vLLM.  Keep
        # it aligned with the language backend so a FlashAttention workaround
        # actually covers the multimodal encoder too.
        mm_encoder_attn_backend=attention_backend,
        enable_mm_embeds=True,
        enable_chunked_prefill=False,
        allowed_local_media_path="/",
        limit_mm_per_prompt={"video": 2},
        media_io_kwargs={"video": {"num_frames": num_frames}},
        # Skip vLLM's profile run by directly specifying the KV cache size. The
        # profile run would pull in the un-pooled ViT forward (the embedding
        # monkeypatch does not trigger on profile dummy inputs), eating tens of
        # GiB and failing with a negative KV budget under multi-server contention.
        # Serving is short-output, so a small KV cache is plenty.
        kv_cache_memory_bytes=kv_cache_bytes,
    )
    print("[lightnav] vLLM engine loaded.")
    return llm
