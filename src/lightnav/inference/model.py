"""Model loading: full HF model or vLLM engine + extracted ViT, with eval_config.json auto-detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lightnav.inference.config import InferenceConfig


@dataclass
class ModelBundle:
    """Holds a loaded model and its associated processors, ready for inference.

    For the ``hf`` backend ``model`` is the full Qwen3-VL model; for ``vllm_local``
    it is the vision tower extracted from the in-process vLLM engine.
    """

    model: Any
    processor: Any
    data_processor: Any  # Qwen3VLDataProcessor
    device: torch.device
    position_id_func: Any = None

    video_size: tuple[int, int] = (224, 320)
    video_fps: int = 4
    pool_enable: bool = False
    pool_spatial: int = 1
    pool_mode: str = "avg"
    pool_stage: str = "pre_vit"
    max_seq_len: int = 8192
    num_history_frames: int = 16
    predict_horizon: int = 1
    # SlowFast multi-tier history (opt-in, mirrors training). None -> plain window.
    slowfast_tiers: Any = None
    # Action tokenizer method ("flat" | "rvq"). rvq checkpoints train with the
    # coarse-to-fine output sentence (to_rvq_prompt), so the inference prompt must
    # apply the same swap; "flat" leaves the unified template wording unchanged.
    action_method: str = "flat"

    @property
    def tokenizer(self):
        return self.processor.tokenizer


def resolve_model_paths(model_path: str) -> tuple[str, str, str]:
    """
    Resolve config / weights / processor directories from a model path.

    Handles plain HF checkpoint dirs and ``hf_ckpt/`` subdirectory layouts.
    """
    mp = Path(model_path)

    if (mp / "config.json").exists():
        has_weights = (
            any(mp.glob("*.safetensors")) or any(mp.glob("*.bin")) or (mp / ".metadata").exists()
        )
        if has_weights:
            return str(mp), str(mp), str(mp)

    hf_sub = mp / "hf_ckpt"
    if hf_sub.is_dir() and (hf_sub / "config.json").exists():
        return str(hf_sub), str(hf_sub), str(hf_sub)

    config_dir = str(mp)
    processor_dir = str(mp)

    for candidate in [mp, mp.parent, mp.parent.parent, mp.parent.parent.parent]:
        if not candidate.exists():
            continue
        if config_dir == str(mp) and (candidate / "config.json").exists():
            config_dir = str(candidate)
        if config_dir == str(mp) and (candidate / "model_assets" / "config.json").exists():
            config_dir = str(candidate / "model_assets")
        if processor_dir == str(mp) and (candidate / "tokenizer_config.json").exists():
            processor_dir = str(candidate)

    return config_dir, str(mp), processor_dir


def _resolve_processing_params(
    model_path: str,
    task_type: str = "vlnce",
    caller_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve processing parameters with 3-tier priority:
      1. Explicit caller overrides (not None)
      2. eval_config.json from the model checkpoint
      3. INFERENCE_FALLBACK_DEFAULTS
    """
    from lightnav.eval_config import (
        get_task_params,
        load_eval_config,
        resolve_inference_params,
    )

    cfg_params: dict[str, Any] = {}
    config = load_eval_config(model_path)
    if config is not None:
        cfg_params = get_task_params(config, task_type)
        if cfg_params:
            print(f"[lightnav] Loaded eval_config.json (task={task_type})")
        # Every frame is resized to the fixed ``video_size`` here; checkpoints trained in
        # native-resolution mode (aspect-preserving smart_resize) would silently get the
        # wrong input, so refuse them instead of degrading quietly.
        if bool(cfg_params.get("native_resolution", False)):
            raise NotImplementedError(
                "this checkpoint's eval_config.json sets native_resolution=true; the inference "
                "engine only supports fixed-size inputs (common.video_size) and would feed the "
                "model a different frame geometry than it was trained on"
            )

    resolved, from_config = resolve_inference_params(
        caller_overrides or {},
        cfg_params,
    )
    if from_config:
        print(f"[lightnav] Auto-configured from eval_config.json: {', '.join(from_config)}")

    return resolved


def _caller_overrides_from_config(config: InferenceConfig) -> dict[str, Any]:
    overrides = {}
    if config.video_size is not None:
        overrides["video_size"] = tuple(config.video_size)
    if config.pool_enable is not None:
        overrides["pool_enable"] = config.pool_enable
    if config.pool_spatial is not None:
        overrides["pool_spatial"] = config.pool_spatial
    if config.pool_mode is not None:
        overrides["pool_mode"] = config.pool_mode
    if config.pool_stage is not None:
        overrides["pool_stage"] = config.pool_stage
    if getattr(config, "num_history_frames", None) is not None:
        overrides["num_history_frames"] = config.num_history_frames
    return overrides


def _make_bundle(model, processor, data_processor, device, pos_id_func, params):
    from lightnav.slowfast import validate_slowfast_tiers

    # Bridge the checkpoint's recorded timestamp mode into the env flag that
    # _calculate_timestamps reads, so inference auto-matches training without
    # anyone manually setting VLN_TIMESTAMP_RELATIVE. setdefault: an explicit env
    # still overrides the recorded value (e.g. for ad-hoc experiments).
    if params.get("timestamp_relative"):
        os.environ.setdefault("VLN_TIMESTAMP_RELATIVE", "1")

    return ModelBundle(
        model=model,
        processor=processor,
        data_processor=data_processor,
        device=device,
        position_id_func=pos_id_func,
        video_size=params["video_size"],
        video_fps=params["video_fps"],
        pool_enable=bool(params["pool_enable"]),
        pool_spatial=params["pool_spatial"],
        pool_mode=params["pool_mode"],
        pool_stage=params.get("pool_stage", "pre_vit"),
        max_seq_len=params["max_seq_len"],
        num_history_frames=params["num_history_frames"],
        predict_horizon=params["predict_horizon"],
        slowfast_tiers=(
            validate_slowfast_tiers(params["slowfast_tiers"]) if params.get("slowfast_tiers") else None
        ),
        action_method=(params.get("action_tokenizer") or {}).get("method") or "flat",
    )


def _build_data_processor(processor, params, pos_id_func):
    from lightnav.data_processor import Qwen3VLDataProcessor

    return Qwen3VLDataProcessor(
        processor=processor,
        model_max_length=params["max_seq_len"],
        video_fps=params["video_fps"],
        video_pool_enable=bool(params["pool_enable"]),
        video_pool_spatial=params["pool_spatial"],
        video_pool_mode=params["pool_mode"],
        video_pool_stage=params.get("pool_stage", "pre_vit"),
        video_size=params["video_size"],
        position_id_func=pos_id_func,
    )


def load_hf_model(config: InferenceConfig, task_type: str = "vlnce") -> ModelBundle:
    """Load full model + processor + data processor from a checkpoint (``hf`` backend)."""
    from transformers import AutoModelForImageTextToText

    from lightnav.processing import VLNQwen3VLProcessor

    if not config.model_path:
        raise ValueError("--model_path is required.")

    params = _resolve_processing_params(
        config.model_path,
        task_type=task_type,
        caller_overrides=_caller_overrides_from_config(config),
    )

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config_dir, weights_dir, processor_dir = resolve_model_paths(config.model_path)

    print(f"[lightnav] Loading model from: {config.model_path}")
    print(f"[lightnav]   device={device}, video_size={params['video_size']}")

    # The trajectory-token checkpoints export the stock Qwen3-VL architecture
    # (flat vocab: <traj_*>/<tpos_*> are ordinary embedding rows), so stock
    # transformers loads them directly.
    #
    # Default to 'sdpa': it is numerically equivalent to flash_attention_2 at
    # inference and has no flash-attn dependency / version-sensitivity (some
    # flash-attn builds reject Qwen3-VL's varlen ViT call at forward time).
    # Override via InferenceConfig.attn_implementation or the LIGHTNAV_ATTN env var.
    attn = (
        getattr(config, "attn_implementation", None)
        or os.environ.get("LIGHTNAV_ATTN")
        or "sdpa"
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            weights_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        )
    except (ImportError, ValueError) as e:
        print(f"[lightnav] attn_implementation={attn!r} unavailable ({e}); falling back to 'sdpa'")
        model = AutoModelForImageTextToText.from_pretrained(
            weights_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model = model.to(device)
    model.eval()

    processor = VLNQwen3VLProcessor.from_pretrained(
        processor_dir,
        padding_side="left",
    )

    # If the checkpoint was trained with post-ViT pooling, the data processor
    # emits pooled video tokens via _original_video_grid_thw + _post_vit_pool_factors,
    # and the model's pos_embeds expect the pooled grid. Apply the same runtime
    # patch used at training time; otherwise model.generate() runs the ViT on the
    # un-pooled grid and pos_embeds adds against the pooled grid -> shape mismatch.
    # The patch is a no-op when pool_stage != 'post_vit'.
    model._post_vit_target = None
    if bool(params.get("pool_enable")) and params.get("pool_stage") == "post_vit":
        from lightnav.processing import enable_post_vit_pool, get_post_vit_target

        merge_size = int(getattr(processor.video_processor, "merge_size", 2))
        enable_post_vit_pool(
            model,
            spatial_factor=int(params.get("pool_spatial", 1)),
            merge_size=merge_size,
        )
        model._post_vit_target = get_post_vit_target(model)
        print(
            f"[lightnav] Enabled post-ViT pooling: spatial={params['pool_spatial']}, "
            f"merge_size={merge_size}"
        )

    # Stock transformers models have no get_position_id_func; derive the same
    # mrope position-id closure from the config so the HF path computes
    # position_ids consistently with the vllm_local path.
    if hasattr(model, "get_position_id_func"):
        pos_id_func = model.get_position_id_func()
    else:
        pos_id_func = _position_id_func_from_config(config.model_path)

    data_processor = _build_data_processor(processor, params, pos_id_func)

    print(f"[lightnav] Model loaded: {type(model).__name__}")
    return _make_bundle(model, processor, data_processor, device, pos_id_func, params)


def _get_position_id(main_func, self, **kwargs):
    """Adapter around transformers ``Qwen3VLModel.get_rope_index`` (top-level so it stays picklable).

    transformers 5.8's Qwen3VLModel.get_rope_index calls self.get_vision_position_ids(...),
    but we pass a minimal SimpleNamespace(config=...) as ``self``, which lacks that
    method. get_vision_position_ids uses no instance state, so bind it on demand.
    """
    import types

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    if not hasattr(self, "get_vision_position_ids") and hasattr(Qwen3VLModel, "get_vision_position_ids"):
        self.get_vision_position_ids = types.MethodType(Qwen3VLModel.get_vision_position_ids, self)
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


def _position_id_func_from_config(model_path: str):
    """Build position_id_func from the model config only (no weights loaded)."""
    from functools import partial
    from types import SimpleNamespace

    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config_dir, _, _ = resolve_model_paths(model_path)
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    fake_model = SimpleNamespace(config=config)
    return partial(_get_position_id, Qwen3VLModel.get_rope_index, fake_model)


def load_vllm_local(config: InferenceConfig, task_type: str = "vlnce"):
    """Load a vLLM engine and extract its ViT for local inference. Returns ``(bundle, llm)``.

    The vLLM ViT returns a single concatenated tensor (base + deepstack along the
    hidden dim); ``bundle.model`` is that vision tower and ``llm`` is the engine.
    """
    # Must be set before the first `import vllm` anywhere in the process so the
    # model lives in-process and get_vllm_model can reach it.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from lightnav.inference.vllm_utils import (
        apply_vllm_embedding_monkeypatch,
        get_vllm_model,
        load_vllm_engine,
    )
    from lightnav.processing import VLNQwen3VLProcessor

    if not config.model_path:
        raise ValueError("--model_path is required.")

    params = _resolve_processing_params(
        config.model_path,
        task_type=task_type,
        caller_overrides=_caller_overrides_from_config(config),
    )

    apply_vllm_embedding_monkeypatch()
    llm = load_vllm_engine(config, num_frames=int(params["num_history_frames"]))

    # fp8_llm_only reuses vLLM's own visual tower for embeddings and relies on the
    # quant patch having kept it bf16 -- verify the prefix guard actually matched.
    from lightnav.inference.vllm_utils import _assert_visual_kept_bf16
    _assert_visual_kept_bf16()

    vit = get_vllm_model(llm).visual
    device = next(get_vllm_model(llm).parameters()).device
    vit = vit.to(device)

    _, _, processor_dir = resolve_model_paths(config.model_path)
    processor = VLNQwen3VLProcessor.from_pretrained(
        processor_dir,
        padding_side="left",
    )

    pos_id_func = _position_id_func_from_config(config.model_path)
    data_processor = _build_data_processor(processor, params, pos_id_func)

    vit_params = sum(p.numel() for p in vit.parameters()) / 1e6
    print(f"[lightnav] vLLM ViT extracted: {vit_params:.0f}M params on {device}")
    bundle = _make_bundle(vit, processor, data_processor, device, pos_id_func, params)
    return bundle, llm
