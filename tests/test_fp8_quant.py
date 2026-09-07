"""fp8_llm_only quantization helpers. CPU-only: no vLLM, no CUDA, no model load.

Covers the pieces the Jetson-Thor PR added to lightnav.inference.vllm_utils:
the version-release parser, quantization resolution / rejection, and the
visual-kept-bf16 assertion.
"""

from __future__ import annotations

import sys
import types

import pytest

from lightnav.inference import vllm_utils
from lightnav.inference.config import InferenceConfig


# -- _vllm_release: build tags accepted, patch releases distinguished ----------

@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.19.1", (0, 19, 1)),
        ("0.19.1+cu132", (0, 19, 1)),   # local build tag must not change the release
        ("0.19.1+cu130", (0, 19, 1)),
        ("0.19.10", (0, 19, 10)),        # must NOT collapse to 0.19.1
        ("0.19.0", (0, 19, 0)),
        ("1.2.3rc4", (1, 2, 3)),
        ("", None),
        ("garbage", None),
    ],
)
def test_vllm_release_parse(version, expected):
    assert vllm_utils._vllm_release(version) == expected


def test_vllm_release_distinguishes_patch_from_local_suffix():
    # the exact bug the guard must avoid: 0.19.10 is NOT 0.19.1
    assert vllm_utils._vllm_release("0.19.1+cu132") == (0, 19, 1)
    assert vllm_utils._vllm_release("0.19.10") != (0, 19, 1)


# -- _resolve_quantization: only fp8_llm_only is supported ---------------------

def _cfg(quant):
    return InferenceConfig(model_path="/m", backend="vllm_local", quantization=quant)


def test_resolve_quantization_bf16_default(monkeypatch):
    called = {"patched": False}
    monkeypatch.setattr(vllm_utils, "_patch_fp8_llm_only",
                        lambda: called.__setitem__("patched", True))
    assert vllm_utils._resolve_quantization(_cfg(None)) is None
    assert called["patched"] is False


def test_resolve_quantization_fp8_llm_only(monkeypatch):
    called = {"patched": False}
    monkeypatch.setattr(vllm_utils, "_patch_fp8_llm_only",
                        lambda: called.__setitem__("patched", True))
    assert vllm_utils._resolve_quantization(_cfg("fp8_llm_only")) == "fp8"
    assert called["patched"] is True


def test_resolve_quantization_refuses_full_fp8():
    with pytest.raises(ValueError, match="corrupt perception|unsupported"):
        vllm_utils._resolve_quantization(_cfg("fp8"))


# -- _assert_visual_kept_bf16: fails loudly when the prefix guard misses -------

def test_assert_visual_bf16_raises_when_no_visual_matched(monkeypatch):
    monkeypatch.setattr(vllm_utils, "_visual_seen_counter", {"n": 0})
    with pytest.raises(RuntimeError, match="visual"):
        vllm_utils._assert_visual_kept_bf16()


def test_assert_visual_bf16_ok_when_matched(monkeypatch):
    monkeypatch.setattr(vllm_utils, "_visual_seen_counter", {"n": 5})
    vllm_utils._assert_visual_kept_bf16()  # no raise


def test_assert_visual_bf16_noop_without_patch(monkeypatch):
    # bf16 path never patched -> counter is None -> nothing to assert
    monkeypatch.setattr(vllm_utils, "_visual_seen_counter", None)
    vllm_utils._assert_visual_kept_bf16()  # no raise


# -- the monkeypatch's prefix predicate keeps visual.* bf16 --------------------

def test_patch_fp8_prefix_predicate(monkeypatch):
    """Drive _patch_fp8_llm_only against a stub vllm.fp8 module and confirm the
    wrapper unquantizes visual.* linears while delegating the rest."""
    # stub the vllm modules the patch imports
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.19.1+cu132"

    class LinearBase: ...
    class UnquantizedLinearMethod: ...
    sentinel_orig = object()

    class Fp8Config:
        def get_quant_method(self, layer, prefix):
            return sentinel_orig

    fp8_mod = types.SimpleNamespace(
        Fp8Config=Fp8Config, LinearBase=LinearBase,
        UnquantizedLinearMethod=UnquantizedLinearMethod,
    )
    quant_pkg = types.ModuleType("vllm.model_executor.layers.quantization")
    quant_pkg.fp8 = fp8_mod

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules,
                        "vllm.model_executor.layers.quantization", quant_pkg)
    monkeypatch.setitem(sys.modules,
                        "vllm.model_executor.layers.quantization.fp8", fp8_mod)
    monkeypatch.setattr(vllm_utils, "_fp8_llm_only_patched", False)

    vllm_utils._patch_fp8_llm_only()

    cfg, lin = Fp8Config(), LinearBase()
    # visual linears -> unquantized (kept bf16)
    assert isinstance(cfg.get_quant_method(lin, "visual.blocks.0.attn.qkv"),
                      UnquantizedLinearMethod)
    assert isinstance(cfg.get_quant_method(lin, "model.visual.mlp.fc1"),
                      UnquantizedLinearMethod)
    # LLM linears -> delegate to the original method
    assert cfg.get_quant_method(lin, "model.layers.0.mlp.gate_proj") is sentinel_orig
    # and the guard counted the visual hits
    assert vllm_utils._visual_seen_counter["n"] == 2
