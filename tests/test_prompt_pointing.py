"""客户端输入 pointing token 的 CPU 单测。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lightnav.inference.engine import VLNInferenceEngine


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<apos_1273>": 501, "<opos_1223>": 502}.get(token, self.unk_token_id)


def _engine():
    engine = VLNInferenceEngine.__new__(VLNInferenceEngine)
    engine.bundle = SimpleNamespace(tokenizer=_Tokenizer())
    return engine


def test_prompt_pointing_converts_semantic_ids_to_tokenizer_ids():
    assert _engine()._prompt_pointing_token_ids((1273, 1223)) == [501, 502]


def test_prompt_pointing_is_optional():
    assert _engine()._prompt_pointing_token_ids(None) == []


@pytest.mark.parametrize("pointing", [(1300, 0), (1, 1297), (True, 0)])
def test_prompt_pointing_rejects_invalid_ids(pointing):
    with pytest.raises(ValueError):
        _engine()._prompt_pointing_token_ids(pointing)
