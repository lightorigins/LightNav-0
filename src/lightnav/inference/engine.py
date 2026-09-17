"""Inference engine: sample dict -> generated action text, on the HF or in-process vLLM backend."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from lightnav.inference.model import ModelBundle
from lightnav.inference.samples import build_tracking_sample, build_vln_traj_sample
from lightnav.inference.vit_cache import VitTubeletCache


@dataclass
class VitResult:
    """Per-request ViT output, ready to feed the vLLM engine."""

    prompt_ids: list[int]
    video_embeds: "torch.Tensor | None"  # CPU tensor, post-pool
    video_grid_thw: "torch.Tensor | None"  # CPU tensor
    timings: dict = field(default_factory=dict)
    # 可选的输入 pointing 语义 id，不是 tokenizer 内部 id。
    prompt_pointing_ids: tuple[int, int] | None = None
    prompt_opos_id: int | None = None


def _get_sampling_params() -> dict[str, float | int | bool]:
    """Read VLN_EVAL_TEMPERATURE / VLN_EVAL_TOP_P / VLN_EVAL_TOP_K / VLN_EVAL_TRAJ_TOP1 env vars.

    Defaults give greedy decoding (temperature=0.0). Set temperature>0 to enable
    sampling. VLN_EVAL_TRAJ_TOP1=1 restricts generation to the action-token
    vocabulary present in the checkpoint tokenizer (plus eos), i.e. greedy
    argmax over the trajectory tokens -- only meaningful when every prompt
    expects a trajectory output.
    """
    return {
        "temperature": float(os.environ.get("VLN_EVAL_TEMPERATURE", "0.0")),
        "top_p": float(os.environ.get("VLN_EVAL_TOP_P", "1.0")),
        "top_k": int(os.environ.get("VLN_EVAL_TOP_K", "0")),
        "traj_top1": os.environ.get("VLN_EVAL_TRAJ_TOP1", "0").lower() in ("1", "true", "yes"),
    }


def _build_prompt_dict(prompt_ids, video_embeds, video_grid_thw):
    """One vLLM prompt entry. video_embeds MUST be CPU (vLLM 0.19 pins host
    memory for the async H2D copy; a CUDA tensor raises "Only dense CPU tensors
    can be pinned")."""
    d = {"prompt_token_ids": list(prompt_ids)}
    if video_embeds is not None and video_grid_thw is not None:
        d["multi_modal_data"] = {
            "video": {
                "video_embeds": video_embeds.cpu(),
                "video_grid_thw": video_grid_thw.cpu(),
            }
        }
    return d


class VLNInferenceEngine:
    """
    Core inference: video_segments sample -> action text.

    Two backends:
      - ``hf``: ``transformers`` ``model.generate`` on the full model.
      - ``vllm_local``: local ViT forward (through a per-session tubelet cache)
        + in-process vLLM decode over pre-computed video embeddings.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        backend: str = "hf",
        vllm_engine: Any = None,
        max_new_tokens: int = 64,
        enable_vit_cache: bool = True,
        aspect_mode: str = "stretch",
    ):
        if aspect_mode not in ("stretch", "keep"):
            raise ValueError(f"aspect_mode must be 'stretch' or 'keep', got {aspect_mode!r}")
        self.aspect_mode = aspect_mode
        self.bundle = bundle
        self.backend = backend
        self.vllm_engine = vllm_engine
        self.max_new_tokens = max_new_tokens
        self.enable_vit_cache = enable_vit_cache
        cache_cap = max(32, int(self.bundle.num_history_frames) * 2)
        if getattr(self.bundle, "slowfast_tiers", None):
            # SlowFast reuses the SAME absolute tubelets across the whole episode
            # (burst/span pairs recur every few steps). A 128-entry LRU evicts
            # them before reuse; 512 holds a full episode so only the newest
            # tubelet misses each step. Cache size affects speed only, never output.
            cache_cap = max(cache_cap, 512)
        self._vit_cache_max_entries = cache_cap
        self._vit_cache = VitTubeletCache(max_entries=cache_cap)
        # Pinned staging buffers for the packed-embeds D2H copy, one per batch slot
        # (see _embeds_to_cpu).
        self._embeds_pin_bufs: dict[int, torch.Tensor] = {}
        self._traj_allowed_token_ids: list[int] | None = None
        self._last_generate_timings: dict[str, float] = {}

    def _get_traj_allowed_token_ids(self) -> list[int]:
        """IDs of {<traj_*>, <tpos_*>, <act_l*>, <apos_*>, <opos_*>, <pos_*>, eos} present
        in the checkpoint tokenizer, for VLN_EVAL_TRAJ_TOP1.

        Sized by probing the tokenizer rather than from a fixed-size constant, so a
        checkpoint with a larger trajectory vocabulary is never silently masked.
        <tpos_*> is included when present because tracking checkpoints emit
        "<tpos_k><traj_k>"; a traj-only allowlist would force the first generated
        token off the trained format. Dual-pointing checkpoints emit
        "<apos_k><opos_k><act_l0_*>..." and posxy checkpoints emit the shared
        <pos_*> prefix / sentinels, so their leading grounding tokens must be
        allowed too. Checkpoints without such rows probe to nothing and are unaffected.
        """
        if self._traj_allowed_token_ids is not None:
            return self._traj_allowed_token_ids
        from lightnav.vln_utils import (
            POSXY_SENTINELS,
            POSXY_TOKENS,
            apos_token,
            opos_token,
            rvq_action_token,
            tpos_token,
            traj_token,
        )

        tok = self.bundle.tokenizer

        def _probe(fmt) -> list[int]:
            out = []
            for i in range(65536):
                tid = tok.convert_tokens_to_ids(fmt(i))
                if tid is None or tid == tok.unk_token_id:
                    break
                out.append(int(tid))
            return out

        def _probe_rvq() -> list[int]:
            """All <act_l{level}_{code}> ids across levels (union allowlist). The rvq
            checkpoint emits one token per level in order; a union (not per-level)
            constraint keeps generation on action tokens and the decoder parses by
            level index."""
            out, level = [], 0
            while True:
                level_ids = _probe(lambda code, _lvl=level: rvq_action_token(_lvl, code))
                if not level_ids:
                    break
                out += level_ids
                level += 1
            return out

        traj_ids = _probe(traj_token)
        tpos_ids = _probe(tpos_token)
        act_ids = _probe_rvq()
        apos_ids = _probe(apos_token)
        opos_ids = _probe(opos_token)
        posxy_ids = [
            int(t)
            for t in (tok.convert_tokens_to_ids(x) for x in (POSXY_TOKENS + POSXY_SENTINELS))
            if t is not None and t != tok.unk_token_id
        ]
        if not traj_ids and not act_ids:
            raise RuntimeError(
                "No <traj_*> or <act_l*> tokens in tokenizer -- refusing to apply top-1 constraint."
            )
        ids = traj_ids + tpos_ids + act_ids + apos_ids + opos_ids + posxy_ids
        if tok.eos_token_id is not None:
            ids.append(int(tok.eos_token_id))
        print(
            f"[lightnav] TRAJ_TOP1 allowlist from checkpoint vocab: {len(traj_ids)} traj + "
            f"{len(tpos_ids)} tpos + {len(act_ids)} rvq-act + {len(apos_ids)} apos + "
            f"{len(opos_ids)} opos + {len(posxy_ids)} posxy tokens"
        )
        self._traj_allowed_token_ids = ids
        return ids

    def reset_episode_state(self) -> None:
        """Reset episode-local inference cache state."""
        self._vit_cache.clear()

    def new_vit_cache(self) -> VitTubeletCache:
        """Create a fresh ViT cache for one online session."""
        return VitTubeletCache(max_entries=self._vit_cache_max_entries)

    def _embeds_to_cpu(self, embeds: torch.Tensor, slot: int = 0) -> torch.Tensor:
        """D2H copy of the packed video embeds through a reusable pinned buffer.

        Plain ``.cpu()`` re-allocates pageable memory every step with a page-fault
        tail; a pinned staging buffer keeps the copy on the DMA path at a stable
        latency. Output is identical to ``embeds.cpu()``.

        Buffers are keyed by BATCH SLOT: a whole tick's ``VitResult``s go into a
        single ``llm_generate_batch`` call, so every session's CPU tensor must stay
        valid simultaneously. Reuse ACROSS ticks is safe because the serving
        scheduler runs ``infer_fn`` serially and awaits each batch to completion
        before pulling the next.
        """
        if not embeds.is_cuda or embeds.dim() != 2:
            return embeds.cpu()
        rows, dim = embeds.shape
        buf = self._embeds_pin_bufs.get(slot)
        if buf is None or buf.dtype != embeds.dtype or buf.shape[1] != dim or buf.shape[0] < rows:
            cap = max(rows, int(rows * 1.25))
            buf = torch.empty((cap, dim), dtype=embeds.dtype, pin_memory=True)
            self._embeds_pin_bufs[slot] = buf
        out = buf[:rows]
        out.copy_(embeds)  # stream-ordered: waits for the producing ViT/pool kernels
        return out

    def _processor_vit_cached_keys(
        self, vit_cache: VitTubeletCache, frame_hw: tuple[int, int] | None = None
    ) -> set[tuple[int, int, int, int]]:
        """Return raw cache keys plus keys normalized to the processor's hit-mask grid.

        ``frame_hw`` is the actual model-frame size of the sample (it equals the
        checkpoint's ``video_size`` unless ``aspect_mode="keep"`` chose another size).
        """
        raw = vit_cache.cached_keys()
        if not raw:
            return raw
        vp = getattr(self.bundle.processor, "video_processor", None)
        patch_size = int(getattr(vp, "patch_size", 16)) if vp is not None else 16
        h, w = frame_hw if frame_hw is not None else self.bundle.video_size
        gh = int(h) // patch_size
        gw = int(w) // patch_size
        return raw | {(int(f0), int(f1), gh, gw) for f0, f1, _h, _w in raw}

    def generate(
        self,
        sample: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        """
        Generate from a video_segments sample dict.

        Args:
            sample: raw sample with "video_segments", "conversations", "video_fps"
            max_new_tokens: override default generation length

        Returns:
            (generated_text, latency_ms)
        """
        max_tok = max_new_tokens or self.max_new_tokens
        prompt_pointing_ids = sample.get("_prompt_pointing_ids")
        prompt_opos_id = sample.get("_prompt_opos_id")

        if self.backend == "hf":
            inputs = self.bundle.data_processor.process_sample(
                sample,
                add_generation_prompt=True,
                validate_video_shapes=False,
            )
            if prompt_opos_id is not None:
                return self._generate_hf_staged(
                    inputs,
                    max_tok,
                    int(prompt_opos_id),
                    int(sample.get("_action_token_count", max_tok)),
                )
            return self._generate_hf(inputs, max_tok, prompt_pointing_ids)

        if self.backend == "vllm_local":
            # ViT through the episode-local tubelet cache (the same path the batched
            # server uses per session), then one vLLM decode.
            if self.enable_vit_cache and sample.get("_allow_vit_cache"):
                res = self._vit_forward_one_with_cache(sample, self._vit_cache, 0)
            else:
                res = self.vit_forward(sample)
            t0 = time.monotonic()
            if prompt_opos_id is not None:
                text = self.llm_generate_two_stage_batch(
                    [res], [int(prompt_opos_id)], max_tok,
                    action_token_count=int(sample.get("_action_token_count", max_tok)),
                )[0]
            else:
                text = self.llm_generate_batch([res], max_tok)[0]
            llm_ms = (time.monotonic() - t0) * 1000
            data_prep_ms = res.timings.get(
                "data_prep_ms",
                res.timings.get("process_sample_ms", 0.0) + res.timings.get("prepare_inputs_ms", 0.0),
            )
            self._last_generate_timings = {
                "data_prep_ms": data_prep_ms,
                "vit_ms": res.timings.get("vit_ms", 0.0),
                "llm_ms": llm_ms,
            }
            return text, llm_ms

        raise ValueError(f"Unknown backend: {self.backend}")

    def generate_from_frames(
        self,
        video_tensor: torch.Tensor,
        instruction: str,
        predict_horizon: int = 1,
        frame_ids: list[int] | None = None,
        max_new_tokens: int | None = None,
        task_type: str = "tracking",
        prompt_pointing_ids: tuple[int, int] | None = None,
        prompt_opos_id: int | None = None,
        action_token_count: int | None = None,
    ) -> tuple[str, float]:
        """
        Generate from a raw video tensor + instruction text.

        ``task_type`` selects the prompt template family:
          - "tracking":   TRACKING_PROMPT_TEMPLATE[_POOLED] (single trajectory id token)
          - "vlnce_traj": VLN_TRAJ_PROMPT_TEMPLATE[_POOLED] (single trajectory id token)

        ``predict_horizon`` is accepted for interface stability and ignored: both
        families predict one trajectory token whose horizon is fixed by the
        checkpoint. The _POOLED variant is selected when the bundle was built with
        ``pool_enable and pool_spatial > 1`` and the clip has >2 frames, mirroring
        the training-time prompt selection.
        """
        if task_type == "vlnce_traj":
            sample = build_vln_traj_sample(video_tensor, instruction, frame_ids, self.bundle)
            if prompt_pointing_ids is not None:
                sample["_prompt_pointing_ids"] = prompt_pointing_ids
            if prompt_opos_id is not None:
                sample["_prompt_opos_id"] = int(prompt_opos_id)
                sample["_action_token_count"] = int(
                    action_token_count or self._default_action_token_count()
                )
            return self.generate(sample, max_new_tokens)

        if task_type != "tracking":
            raise ValueError(f"Unknown task_type: {task_type}")

        sample = build_tracking_sample(video_tensor, instruction, frame_ids, self.bundle)
        if prompt_pointing_ids is not None:
            sample["_prompt_pointing_ids"] = prompt_pointing_ids
        if prompt_opos_id is not None:
            sample["_prompt_opos_id"] = int(prompt_opos_id)
            sample["_action_token_count"] = int(
                action_token_count or self._default_action_token_count()
            )
        return self.generate(sample, max_new_tokens)

    def _prepare_inputs(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Move processed inputs to device, strip metadata, trim to answer boundary."""
        device = self.bundle.device
        moved: dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    moved[k] = v.to(device=device, dtype=torch.bfloat16)
                else:
                    moved[k] = v.to(device)
            else:
                moved[k] = v

        for k in ("motion_sample_mask", "episode_index", "frame_index", "raw_action_text", "dataset_name"):
            moved.pop(k, None)

        labels = moved.get("labels")
        gen = {k: v for k, v in moved.items() if k != "labels"}
        if labels is not None:
            answer_tokens = (labels != -100).nonzero(as_tuple=False)
            if answer_tokens.numel() > 0:
                answer_start = answer_tokens[0].item()
                for key in ("input_ids", "attention_mask", "image_mask", "video_mask"):
                    if key in gen:
                        gen[key] = gen[key][:answer_start].unsqueeze(0)
                if "position_ids" in gen:
                    gen["position_ids"] = gen["position_ids"][:, :answer_start].unsqueeze(0)
                if "pixel_values_videos" in gen:
                    pv = gen["pixel_values_videos"]
                    if pv.dim() == 2:
                        pv = pv.unsqueeze(0)
                    gen["pixel_values_videos"] = pv
            else:
                for key in ("input_ids", "attention_mask"):
                    if key in gen:
                        gen[key] = gen[key].unsqueeze(0)
        else:
            for key in ("input_ids", "attention_mask"):
                if key in gen:
                    gen[key] = gen[key].unsqueeze(0)

        return gen

    def _generate_hf(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        prompt_pointing_ids: tuple[int, int] | None = None,
    ) -> tuple[str, float]:
        prep_t0 = time.monotonic()
        gen = self._prepare_inputs(inputs)
        prepare_inputs_ms = (time.monotonic() - prep_t0) * 1000

        # Drop the data-processor's precomputed mrope position_ids: it is built by
        # the data processor's position_ids closure, whose layout differs from
        # stock transformers' Qwen3-VL. With position_ids absent, stock Qwen3-VL
        # computes the 3D mrope itself from {image,video}_grid_thw — the correct,
        # version-native path. (The vLLM backend never uses this; it recomputes
        # positions inside vLLM.)
        gen.pop("position_ids", None)
        pointing_token_ids = self._prompt_pointing_token_ids(prompt_pointing_ids)
        if pointing_token_ids:
            suffix = torch.tensor(
                [pointing_token_ids], dtype=gen["input_ids"].dtype, device=gen["input_ids"].device
            )
            gen["input_ids"] = torch.cat((gen["input_ids"], suffix), dim=1)
            for key in ("attention_mask", "image_mask", "video_mask"):
                value = gen.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    fill = 1 if key == "attention_mask" else 0
                    extension = torch.full(
                        (value.shape[0], len(pointing_token_ids)),
                        fill,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    gen[key] = torch.cat((value, extension), dim=1)

        # Post-ViT pooling kwargs are emitted by the data processor and consumed
        # by model.forward() via the _post_vit_target attribute (set at build
        # time by enable_post_vit_pool, see model.load_hf_model). They are
        # NOT model.generate() kwargs, so pop and propagate to the target.
        original_grid = gen.pop("_original_video_grid_thw", None)
        pool_factors = gen.pop("_post_vit_pool_factors", None)
        gen.pop("_post_vit_pool_spatial", None)
        post_vit_target = getattr(self.bundle.model, "_post_vit_target", None)
        if post_vit_target is not None and original_grid is not None:
            post_vit_target._post_vit_original_grid = original_grid
            post_vit_target._post_vit_pool_factors = pool_factors

        # Patch model to accept VLN-specific kwargs (image_mask, video_mask)
        # that model.forward() needs but generate() rejects during validation.
        # Idempotent: each _generate_hf call would otherwise wrap the previous
        # patch, causing unbounded recursion across many predict calls (online
        # inference loop, e.g. tracking websocket server, fires this 30x per
        # episode and hits Python's recursion limit).
        model = self.bundle.model
        if not getattr(model, "_lr_validate_kwargs_patched", False):
            orig_validate = model._validate_model_kwargs
            model._validate_model_kwargs = lambda kwargs: orig_validate(
                {k: v for k, v in kwargs.items() if k not in ("image_mask", "video_mask")}
            )
            model._lr_validate_kwargs_patched = True

        torch.cuda.synchronize()
        t0 = time.monotonic()

        sp = _get_sampling_params()
        do_sample = sp["temperature"] > 0.0
        gen_kwargs = dict(
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.bundle.tokenizer.pad_token_id,
            eos_token_id=self.bundle.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = sp["temperature"]
            if sp["top_p"] < 1.0:
                gen_kwargs["top_p"] = sp["top_p"]
            if sp["top_k"] > 0:
                gen_kwargs["top_k"] = sp["top_k"]

        if sp["traj_top1"]:
            from transformers import LogitsProcessorList

            allowed_ids = self._get_traj_allowed_token_ids()
            vocab_size = int(model.get_output_embeddings().weight.shape[0])
            mask = torch.full((vocab_size,), float("-inf"))
            mask[torch.tensor(allowed_ids, dtype=torch.long)] = 0.0
            mask = mask.to(device=self.bundle.device)

            def _allowed_logits_processor(input_ids, scores):
                return scores + mask

            gen_kwargs["logits_processor"] = LogitsProcessorList([_allowed_logits_processor])

        with torch.no_grad():
            generated_ids = model.generate(**gen, **gen_kwargs)

        torch.cuda.synchronize()
        latency_ms = (time.monotonic() - t0) * 1000

        input_len = gen["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        decode_t0 = time.monotonic()
        text = self.bundle.tokenizer.decode(new_tokens, skip_special_tokens=False)
        decode_ms = (time.monotonic() - decode_t0) * 1000
        self._last_generate_timings = {
            "data_prep_ms": prepare_inputs_ms,
            "prepare_inputs_ms": prepare_inputs_ms,
            "llm_ms": latency_ms,
            "decode_ms": decode_ms,
        }
        return text, latency_ms

    def _default_action_token_count(self) -> int:
        """Return the number of action tokens emitted by this checkpoint."""
        if getattr(self.bundle, "action_method", "flat") == "rvq":
            # The serving layer normally supplies the exact count from its RVQ bundle.
            # Keep a conservative fallback for direct engine callers.
            return max(1, int(self.max_new_tokens))
        return 1

    def _probe_token_family(self, formatter) -> list[int]:
        tok = self.bundle.tokenizer
        out: list[int] = []
        for i in range(65536):
            tid = tok.convert_tokens_to_ids(formatter(i))
            if tid is None or tid == tok.unk_token_id:
                break
            out.append(int(tid))
        return out

    def _get_apos_token_ids(self) -> list[int]:
        from lightnav.vln_utils import apos_token

        ids = self._probe_token_family(apos_token)
        if not ids:
            raise ValueError(
                "staged OPOS mode requires a grid pointing checkpoint with <apos_*> tokens"
            )
        return ids

    def _get_action_token_ids(self) -> list[int]:
        from lightnav.vln_utils import rvq_action_token, traj_token

        if getattr(self.bundle, "action_method", "flat") == "rvq":
            ids: list[int] = []
            level = 0
            while True:
                level_ids = self._probe_token_family(
                    lambda code, _level=level: rvq_action_token(_level, code)
                )
                if not level_ids:
                    break
                ids.extend(level_ids)
                level += 1
            if ids:
                return ids
        ids = self._probe_token_family(traj_token)
        if not ids:
            raise ValueError("No action tokens found in tokenizer for staged OPOS mode")
        return ids

    def _semantic_pointing_token_id(self, channel: str, value: int) -> int:
        from lightnav.vln_utils import APOS_VOCAB_SIZE, OPOS_VOCAB_SIZE

        limit = APOS_VOCAB_SIZE if channel == "apos" else OPOS_VOCAB_SIZE
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < limit:
            raise ValueError(f"{channel}_id out of range: {value}")
        tid = self.bundle.tokenizer.convert_tokens_to_ids(f"<{channel}_{value}>")
        if tid is None or tid == self.bundle.tokenizer.unk_token_id:
            raise ValueError(f"Token not found in tokenizer: <{channel}_{value}>")
        return int(tid)

    @staticmethod
    def _append_input_tokens(gen: dict[str, Any], token_ids: list[int]) -> None:
        if not token_ids:
            return
        suffix = torch.tensor(
            [token_ids], dtype=gen["input_ids"].dtype, device=gen["input_ids"].device
        )
        gen["input_ids"] = torch.cat((gen["input_ids"], suffix), dim=1)
        for key in ("attention_mask", "image_mask", "video_mask"):
            value = gen.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                fill = 1 if key == "attention_mask" else 0
                extension = torch.full(
                    (value.shape[0], len(token_ids)), fill,
                    dtype=value.dtype, device=value.device,
                )
                gen[key] = torch.cat((value, extension), dim=1)

    def _generate_hf_staged(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        prompt_opos_id: int,
        action_token_count: int,
    ) -> tuple[str, float]:
        """Generate APOS, append external OPOS, then generate action tokens.

        This intentionally uses two ``generate`` calls. The processed video tensors are
        reused, but transformers KV state is not exposed by the current backend.
        """
        prep_t0 = time.monotonic()
        base = self._prepare_inputs(inputs)
        base.pop("position_ids", None)
        original_grid = base.pop("_original_video_grid_thw", None)
        pool_factors = base.pop("_post_vit_pool_factors", None)
        base.pop("_post_vit_pool_spatial", None)
        post_vit_target = getattr(self.bundle.model, "_post_vit_target", None)
        if post_vit_target is not None and original_grid is not None:
            post_vit_target._post_vit_original_grid = original_grid
            post_vit_target._post_vit_pool_factors = pool_factors
        prepare_inputs_ms = (time.monotonic() - prep_t0) * 1000
        apos_ids = self._get_apos_token_ids()
        opos_tid = self._semantic_pointing_token_id("opos", prompt_opos_id)

        model = self.bundle.model
        if not getattr(model, "_lr_validate_kwargs_patched", False):
            orig_validate = model._validate_model_kwargs
            model._validate_model_kwargs = lambda kwargs: orig_validate(
                {k: v for k, v in kwargs.items() if k not in ("image_mask", "video_mask")}
            )
            model._lr_validate_kwargs_patched = True

        from transformers import LogitsProcessorList

        sp = _get_sampling_params()
        do_sample = sp["temperature"] > 0.0

        def run(gen: dict[str, Any], count: int, allowed: list[int]):
            vocab_size = int(model.get_output_embeddings().weight.shape[0])
            mask = torch.full((vocab_size,), float("-inf"), device=self.bundle.device)
            mask[torch.tensor(allowed, dtype=torch.long, device=self.bundle.device)] = 0.0
            kwargs: dict[str, Any] = {
                "min_new_tokens": count,
                "max_new_tokens": count,
                "do_sample": do_sample,
                "pad_token_id": self.bundle.tokenizer.pad_token_id,
                "eos_token_id": self.bundle.tokenizer.eos_token_id,
                "logits_processor": LogitsProcessorList([lambda _ids, scores: scores + mask]),
            }
            if do_sample:
                kwargs["temperature"] = sp["temperature"]
            with torch.no_grad():
                return model.generate(**gen, **kwargs)

        t0 = time.monotonic()
        first = run(dict(base), 1, apos_ids)
        base_len = base["input_ids"].shape[1]
        generated_apos = int(first[0][base_len].item())
        stage1_text = self.bundle.tokenizer.decode([generated_apos], skip_special_tokens=False)
        staged = dict(base)
        self._append_input_tokens(staged, [generated_apos, opos_tid])
        second = run(staged, max(1, int(action_token_count)), self._get_action_token_ids())
        stage2_len = staged["input_ids"].shape[1]
        stage2_text = self.bundle.tokenizer.decode(
            second[0][stage2_len:], skip_special_tokens=False
        )
        latency_ms = (time.monotonic() - t0) * 1000
        self._last_generate_timings = {
            "data_prep_ms": prepare_inputs_ms,
            "prepare_inputs_ms": prepare_inputs_ms,
            "llm_ms": latency_ms,
        }
        return (
            stage1_text
            + self.bundle.tokenizer.decode([opos_tid], skip_special_tokens=False)
            + stage2_text,
            latency_ms,
        )

    def _vit_forward_from_inputs(self, inputs: dict[str, Any]) -> "VitResult":
        """ViT forward + post-ViT pool from ALREADY-processed inputs. Cache bypassed.

        ``data_prep_ms`` covers only _prepare_inputs (NOT process_sample);
        ``vit_forward`` adds process_sample_ms on top.
        """
        prep_t0 = time.monotonic()
        gen = self._prepare_inputs(inputs)
        prompt_ids = gen["input_ids"][0].tolist()
        data_prep_ms = (time.monotonic() - prep_t0) * 1000

        pv = gen.get("pixel_values_videos")
        vgt = gen.get("video_grid_thw")
        original_vgt = gen.pop("_original_video_grid_thw", None)
        post_vit_pool_factors = gen.pop("_post_vit_pool_factors", None)
        post_vit_pool_spatial = gen.pop("_post_vit_pool_spatial", None)
        vit_grid = original_vgt if original_vgt is not None else vgt

        video_embeds_packed = None
        vit_ms = 0.0
        if pv is not None and vit_grid is not None:
            vit_t0 = time.monotonic()
            vit = self.bundle.model
            pv = pv.to(device=self.bundle.device, dtype=vit.dtype)
            if pv.ndim > 2:
                pv = pv.reshape(-1, pv.shape[-1])
            with torch.no_grad():
                video_embeds_packed = vit(pv, grid_thw=vit_grid.cpu())

            sf = post_vit_pool_factors if post_vit_pool_factors is not None else post_vit_pool_spatial
            run_pool = video_embeds_packed is not None and sf is not None
            if run_pool and isinstance(sf, int):
                run_pool = sf > 1
            if run_pool:
                from lightnav.processing import post_vit_spatial_pool

                merge_size = int(getattr(self.bundle.processor.video_processor, "merge_size", 2))
                video_embeds_packed, vgt, _ = post_vit_spatial_pool(
                    video_embeds_packed,
                    vit_grid.to(video_embeds_packed.device),
                    spatial_factor=sf,
                    merge_size=merge_size,
                )
            torch.cuda.synchronize()
            vit_ms = (time.monotonic() - vit_t0) * 1000

        return VitResult(
            prompt_ids=prompt_ids,
            video_embeds=None if video_embeds_packed is None else video_embeds_packed.cpu(),
            video_grid_thw=None if vgt is None else vgt.cpu(),
            timings={"data_prep_ms": data_prep_ms, "vit_ms": vit_ms},
        )

    def vit_forward(self, sample: dict[str, Any]) -> "VitResult":
        """process_sample + ViT forward + post-ViT pool for ONE request (raw sample entry point).

        Returns CPU embeds ready for vLLM. The tubelet cache is bypassed here
        (cache affects speed only, never output).
        """
        proc_t0 = time.monotonic()
        inputs = self.bundle.data_processor.process_sample(
            sample, add_generation_prompt=True, validate_video_shapes=False,
        )
        process_sample_ms = (time.monotonic() - proc_t0) * 1000
        res = self._vit_forward_from_inputs(inputs)
        res.prompt_pointing_ids = sample.get("_prompt_pointing_ids")
        res.prompt_opos_id = sample.get("_prompt_opos_id")
        res.timings["data_prep_ms"] = res.timings.get("data_prep_ms", 0.0) + process_sample_ms
        return res

    def vit_forward_batch(
        self,
        samples: list[dict[str, Any]],
        vit_caches: list[VitTubeletCache | None] | None = None,
    ) -> list["VitResult"]:
        """ViT for a batch of requests: one call per request, each through its session cache.

        When every request comes with a session ``VitTubeletCache`` and allows
        caching, each is processed via ``_vit_forward_one_with_cache`` (the cache
        self-warms: the first step of a session encodes the full window, later
        steps encode only the new tubelet). Otherwise the uncached per-request
        ``vit_forward`` is used.
        """
        if (
            self.enable_vit_cache
            and vit_caches is not None
            and len(vit_caches) == len(samples)
            and all(c is not None for c in vit_caches)
            and all(s.get("_allow_vit_cache") for s in samples)
        ):
            return [
                self._vit_forward_one_with_cache(sm, cache, slot)  # type: ignore[arg-type]
                for slot, (sm, cache) in enumerate(zip(samples, vit_caches))
            ]
        return [self.vit_forward(s) for s in samples]

    def _vit_forward_one_with_cache(
        self,
        sample: dict[str, Any],
        vit_cache: VitTubeletCache,
        slot: int = 0,
    ) -> "VitResult":
        """Process one request through its session-local ViT cache.

        The processor is told which tubelets are already cached so it patchifies
        only the misses; the cache then runs the ViT on those rows and reassembles
        the full window. ``slot`` is this request's index within the batch; it
        selects the pinned staging buffer for the embeds D2H so concurrent
        sessions in one tick do not share one (see :meth:`_embeds_to_cpu`).
        """
        proc_t0 = time.monotonic()
        frame_hw = tuple(int(v) for v in sample["video_segments"][0]["video"].shape[-2:])
        cached_sample = {
            **sample,
            "_vit_cached_keys": self._processor_vit_cached_keys(vit_cache, frame_hw),
        }
        inputs = self.bundle.data_processor.process_sample(
            cached_sample, add_generation_prompt=True, validate_video_shapes=False,
        )
        process_sample_ms = (time.monotonic() - proc_t0) * 1000

        prep_t0 = time.monotonic()
        gen = self._prepare_inputs(inputs)
        prepare_inputs_ms = (time.monotonic() - prep_t0) * 1000
        prompt_ids = gen["input_ids"][0].tolist()

        pv = gen.get("pixel_values_videos")
        vgt = gen.get("video_grid_thw")
        if pv is None or vgt is None:
            return self.vit_forward(sample)

        original_vgt = gen.pop("_original_video_grid_thw", None)
        post_vit_pool_factors = gen.pop("_post_vit_pool_factors", None)
        post_vit_pool_spatial = gen.pop("_post_vit_pool_spatial", None)
        vit_grid = original_vgt if original_vgt is not None else vgt

        vit_t0 = time.monotonic()
        embeds, vgt_dev, cache_stats = self._compute_vit_embeds_with_cache(
            gen,
            sample,
            grid_thw_override=vit_grid,
            vit_cache=vit_cache,
        )
        torch.cuda.synchronize()
        vit_ms = (time.monotonic() - vit_t0) * 1000

        sf = post_vit_pool_factors if post_vit_pool_factors is not None else post_vit_pool_spatial
        run_pool = embeds is not None and sf is not None
        if run_pool and isinstance(sf, int):
            run_pool = sf > 1
        pool_t0 = time.monotonic()
        if run_pool:
            from lightnav.processing import post_vit_spatial_pool

            merge_size = int(getattr(self.bundle.processor.video_processor, "merge_size", 2))
            embeds, vgt_dev, _ = post_vit_spatial_pool(
                embeds,
                vit_grid.to(embeds.device),
                spatial_factor=sf,
                merge_size=merge_size,
            )
        post_vit_pool_ms = (time.monotonic() - pool_t0) * 1000

        copy_t0 = time.monotonic()
        video_embeds_cpu = None if embeds is None else self._embeds_to_cpu(embeds, slot)
        video_grid_cpu = None if vgt_dev is None else vgt_dev.cpu()
        cpu_copy_ms = (time.monotonic() - copy_t0) * 1000

        return VitResult(
            prompt_ids=prompt_ids,
            video_embeds=video_embeds_cpu,
            video_grid_thw=video_grid_cpu,
            timings={
                "process_sample_ms": process_sample_ms,
                "prepare_inputs_ms": prepare_inputs_ms,
                "vit_ms": vit_ms,
                "post_vit_pool_ms": post_vit_pool_ms,
                "cpu_copy_ms": cpu_copy_ms,
                "vit_cache_hits": float(cache_stats.get("hit", 0)),
                "vit_cache_misses": float(cache_stats.get("miss", 0)),
                "vit_cache_size": float(len(vit_cache.cached_keys())),
            },
            prompt_pointing_ids=sample.get("_prompt_pointing_ids"),
            prompt_opos_id=sample.get("_prompt_opos_id"),
        )

    def _prompt_pointing_token_ids(
        self,
        pointing: tuple[int, int] | list[int] | None,
    ) -> list[int]:
        """把协议中的 APOS/OPOS 语义 id 转成 checkpoint tokenizer id。"""
        if pointing is None:
            return []
        from lightnav.vln_utils import APOS_VOCAB_SIZE, OPOS_VOCAB_SIZE

        if not isinstance(pointing, (tuple, list)) or len(pointing) != 2:
            raise ValueError("prompt_pointing_ids must be (apos_id, opos_id)")
        apos_id, opos_id = pointing
        if isinstance(apos_id, bool) or not isinstance(apos_id, int):
            raise ValueError("apos_id must be an integer")
        if isinstance(opos_id, bool) or not isinstance(opos_id, int):
            raise ValueError("opos_id must be an integer")
        if not 0 <= apos_id < APOS_VOCAB_SIZE:
            raise ValueError(f"apos_id out of range: {apos_id}")
        if not 0 <= opos_id < OPOS_VOCAB_SIZE:
            raise ValueError(f"opos_id out of range: {opos_id}")

        tokenizer = self.bundle.tokenizer
        result = []
        for token in (f"<apos_{apos_id}>", f"<opos_{opos_id}>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(f"Token not found in tokenizer: {token}")
            result.append(int(token_id))
        return result

    def llm_generate_batch(self, items: list["VitResult"], max_new_tokens: int) -> list[str]:
        """Batched LLM decode: one vllm.LLM.generate call for all requests."""
        from vllm import SamplingParams

        prompt_dicts = [
            _build_prompt_dict(
                it.prompt_ids + self._prompt_pointing_token_ids(it.prompt_pointing_ids),
                it.video_embeds,
                it.video_grid_thw,
            )
            for it in items
        ]
        sp = _get_sampling_params()
        sp_kwargs: dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": sp["temperature"],
            "top_p": sp["top_p"],
        }
        if sp["top_k"] > 0:
            sp_kwargs["top_k"] = sp["top_k"]
        if sp["traj_top1"]:
            sp_kwargs["allowed_token_ids"] = self._get_traj_allowed_token_ids()
        params = SamplingParams(**sp_kwargs)

        outputs = self.vllm_engine.generate(prompt_dicts, params, use_tqdm=False)
        texts = []
        for out in outputs:
            o = out.outputs[0]
            if o.token_ids:
                texts.append(self.bundle.tokenizer.decode(list(o.token_ids), skip_special_tokens=False))
            else:
                texts.append(o.text)
        return texts

    def llm_generate_two_stage_batch(
        self,
        items: list["VitResult"],
        prompt_opos_ids: list[int],
        max_new_tokens: int,
        *,
        action_token_count: int | None = None,
    ) -> list[str]:
        """vLLM counterpart of staged APOS -> external OPOS -> action decoding."""
        if len(items) != len(prompt_opos_ids):
            raise ValueError("items and prompt_opos_ids must have equal length")
        from vllm import SamplingParams

        apos_ids = self._get_apos_token_ids()
        opos_token_ids = [self._semantic_pointing_token_id("opos", int(v)) for v in prompt_opos_ids]
        sp = _get_sampling_params()
        base_prompts = [
            _build_prompt_dict(it.prompt_ids, it.video_embeds, it.video_grid_thw) for it in items
        ]
        stage1_params = SamplingParams(
            max_tokens=1,
            temperature=sp["temperature"],
            top_p=sp["top_p"],
            **({"top_k": sp["top_k"]} if sp["top_k"] > 0 else {}),
            allowed_token_ids=apos_ids,
        )
        first = self.vllm_engine.generate(base_prompts, stage1_params, use_tqdm=False)
        generated_apos: list[int] = []
        for out in first:
            if not out.outputs or not out.outputs[0].token_ids:
                raise RuntimeError("staged APOS decode returned no token")
            generated_apos.append(int(out.outputs[0].token_ids[0]))

        stage2_prompts = []
        for it, apos_tid, opos_tid in zip(items, generated_apos, opos_token_ids):
            stage2_prompts.append(
                _build_prompt_dict(
                    it.prompt_ids + [apos_tid, opos_tid], it.video_embeds, it.video_grid_thw
                )
            )
        action_count = max(1, int(action_token_count or max_new_tokens))
        stage2_kwargs: dict[str, Any] = {
            "max_tokens": action_count,
            "temperature": sp["temperature"],
            "top_p": sp["top_p"],
            "allowed_token_ids": self._get_action_token_ids(),
        }
        if sp["top_k"] > 0:
            stage2_kwargs["top_k"] = sp["top_k"]
        second = self.vllm_engine.generate(
            stage2_prompts, SamplingParams(**stage2_kwargs), use_tqdm=False
        )
        texts: list[str] = []
        for apos_tid, opos_tid, out in zip(generated_apos, opos_token_ids, second):
            if out.outputs and out.outputs[0].token_ids:
                action_text = self.bundle.tokenizer.decode(
                    list(out.outputs[0].token_ids), skip_special_tokens=False
                )
            else:
                action_text = out.outputs[0].text if out.outputs else ""
            texts.append(
                self.bundle.tokenizer.decode([apos_tid, opos_tid], skip_special_tokens=False)
                + action_text
            )
        return texts

    def _compute_vit_embeds_with_cache(
        self,
        gen_inputs: dict[str, Any],
        sample: dict[str, Any],
        grid_thw_override: torch.Tensor | None = None,
        vit_cache: VitTubeletCache | None = None,
    ):
        """Compute packed video embeddings with a tubelet cache (episode cache when none is given).

        Returns ``(embeds on device, grid on device, {"hit", "miss"})``; ``(None, None, zeros)``
        when the inputs carry no video.
        """
        if "pixel_values_videos" not in gen_inputs or "video_grid_thw" not in gen_inputs:
            return None, None, {"hit": 0, "miss": 0}

        pv = gen_inputs["pixel_values_videos"]
        if pv.dim() == 3:
            pv = pv.squeeze(0)
        if pv.dim() != 2:
            raise ValueError(f"Expected pixel_values_videos as 2D tensor, got shape={tuple(pv.shape)}")

        vgt = grid_thw_override if grid_thw_override is not None else gen_inputs["video_grid_thw"]
        if vgt.dim() != 2 or vgt.shape[1] != 3:
            raise ValueError(f"Expected video_grid_thw shape (num_videos, 3), got {tuple(vgt.shape)}")

        vit = self.bundle.model
        merge_size = int(getattr(self.bundle.processor.video_processor, "merge_size", 2))

        segments = sample.get("video_segments", [])
        seg_frame_indices = [[int(x) for x in seg.get("frame_indices", [])] for seg in segments]

        pv_gpu = pv.to(device=self.bundle.device, dtype=vit.dtype)
        cache = vit_cache if vit_cache is not None else self._vit_cache
        embeds, hits, misses = cache.get_embeddings(
            pv_gpu,
            vgt.tolist(),
            seg_frame_indices,
            vit=vit,
            merge_size=merge_size,
        )
        return embeds, vgt.to(self.bundle.device), {"hit": hits, "miss": misses}


def build_engine(
    config,
    task_type: str = "vlnce",
    max_new_tokens: int | None = None,
) -> tuple[VLNInferenceEngine, ModelBundle]:
    """
    Build a VLNInferenceEngine from an InferenceConfig.

    ``task_type`` is the ``eval_config.json`` ``tasks[]`` key ("vlnce" for VLN,
    "trackvla" for tracking). Returns (engine, bundle) so callers can read the
    resolved processing params from the bundle.
    """
    from lightnav.inference.model import load_hf_model, load_vllm_local

    max_tok = max_new_tokens if max_new_tokens is not None else config.max_new_tokens
    backend = config.backend

    if backend == "vllm_local":
        bundle, llm = load_vllm_local(config, task_type=task_type)
        engine = VLNInferenceEngine(
            bundle,
            backend="vllm_local",
            vllm_engine=llm,
            max_new_tokens=max_tok,
            aspect_mode=config.aspect_mode,
        )
    elif backend == "hf":
        bundle = load_hf_model(config, task_type=task_type)
        engine = VLNInferenceEngine(
            bundle,
            backend="hf",
            max_new_tokens=max_tok,
            aspect_mode=config.aspect_mode,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'hf' or 'vllm_local')")

    return engine, bundle
