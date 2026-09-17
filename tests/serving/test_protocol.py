from __future__ import annotations

import numpy as np
import pytest

from lightnav.serving import protocol
from lightnav.serving.protocol import (
    actions_payload,
    decode_prediction_signals,
    parse_actions_payload,
)


def test_decode_prediction_signals_for_stop_without_tpos():
    signals = decode_prediction_signals("<traj_0>", vocab_size=4)

    assert signals.traj_id == 0
    assert signals.stop is True
    assert signals.stop_reason == "flat_stop_token"
    assert signals.visible is None
    assert signals.tpos_id is None


def test_decode_prediction_signals_for_invisible_tpos():
    signals = decode_prediction_signals("<tpos_0><traj_3>", vocab_size=4)

    assert signals.traj_id == 3
    assert signals.stop is False
    assert signals.stop_reason is None
    assert signals.visible is False
    assert signals.tpos_id == 0


def test_decode_prediction_signals_rejects_traj_outside_vocab():
    with pytest.raises(ValueError, match=r"traj id 5 out of vocab range \[0, 4\)"):
        decode_prediction_signals("<traj_5>", vocab_size=4)


def test_decode_prediction_signals_rvq_moving_from_waypoints():
    # RVQ output has <act_l*> not <traj_k>: the flat path would raise, so is_rvq
    # takes stop from the (non-zero) decoded waypoints and reads visibility from tpos.
    waypoints = np.array([[1.0, 0.0, 0.1]], dtype=np.float32)
    signals = decode_prediction_signals(
        "<tpos_1><act_l0_186><act_l1_128><act_l2_196>", is_rvq=True, waypoints=waypoints
    )

    assert signals.traj_id is None
    assert signals.stop is False
    assert signals.visible is True
    assert signals.tpos_id == 1


def test_decode_prediction_signals_rvq_stop_is_zero_waypoints():
    signals = decode_prediction_signals(
        "<tpos_0><act_l0_6><act_l1_0><act_l2_0>",
        is_rvq=True,
        waypoints=np.zeros((10, 3), dtype=np.float32),
    )

    assert signals.traj_id is None
    assert signals.stop is True
    assert signals.stop_reason == "rvq_zero_waypoints"
    assert signals.visible is False


def test_decode_prediction_signals_distinguishes_rvq_stop_code_from_near_zero_decode():
    waypoints = np.zeros((10, 3), dtype=np.float32)

    explicit = decode_prediction_signals(
        "<act_l0_6><act_l1_122><act_l2_174>",
        is_rvq=True,
        waypoints=waypoints,
        rvq_stop_l0=6,
    )
    near_zero = decode_prediction_signals(
        "<act_l0_5><act_l1_10><act_l2_20>",
        is_rvq=True,
        waypoints=waypoints,
        rvq_stop_l0=6,
    )

    assert explicit.stop_reason == "rvq_stop_code"
    assert near_zero.stop_reason == "rvq_near_zero_waypoints"


def test_decode_prediction_signals_rvq_requires_waypoints():
    with pytest.raises(ValueError, match="needs waypoints"):
        decode_prediction_signals("<act_l0_1>", is_rvq=True)


def test_actions_payload_matches_wire_contract():
    waypoints = np.array([[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]], dtype=np.float32)

    payload = actions_payload(waypoints, step=7)

    assert payload == {
        "step": 7,
        "actions": [[1.0, 0.25, 0.10000000149011612], [2.0, -0.5, -0.20000000298023224]],
    }


def test_parse_actions_payload_accepts_new_dict_shape():
    data = {"step": 7, "actions": [[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]}

    assert parse_actions_payload(data) == [[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]


def test_parse_actions_payload_accepts_legacy_wrapped_shape():
    data = [[[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]]

    assert parse_actions_payload(data) == [[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]


def test_parse_actions_payload_accepts_flat_shape():
    data = [1.0, 0.25, 0.1, 2.0, -0.5, -0.2]

    assert parse_actions_payload(data) == [[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]


def test_parse_actions_payload_rejects_incomplete_flat_shape():
    with pytest.raises(ValueError, match="flat actions length must be divisible by 3"):
        parse_actions_payload([1.0, 2.0])


# -- pointing pixels on the wire (both token formats) -------------------------


def test_pointing_payload_converts_grid_ids_to_frame_pixels():
    """The grid is 48x27 cells; a client gets the cell CENTRE in its own frame."""
    from lightnav.vln_utils import encode_point_pixel

    apos = encode_point_pixel(0.25 * 480, 0.40 * 270, 480, 270)
    opos = encode_point_pixel(0.70 * 480, 0.60 * 270, 480, 270)
    out = protocol.pointing_payload(f"<apos_{apos}><opos_{opos}>", width=480, height=270)

    assert out["mode"] == "grid"
    assert out["frame_size"] == [480, 270]
    # Cell centres, so within half a cell (10 px wide, 10 px tall at this resolution).
    assert out["apos_px"][0] == pytest.approx(0.25 * 480, abs=5)
    assert out["apos_px"][1] == pytest.approx(0.40 * 270, abs=5)
    assert out["opos_px"][0] == pytest.approx(0.70 * 480, abs=5)
    assert not out["apos_clamped"] and not out["opos_clamped"]


def test_pointing_payload_converts_posxy_bins_to_frame_pixels():
    """posxy is two shared axis tokens at 1000 bins each -- ~0.5px on a 480 frame."""
    out = protocol.pointing_payload(
        "<apos><pos_412><pos_650><opos><pos_205><pos_88><act_l0_91>",
        width=480,
        height=270,
    )

    assert out["mode"] == "posxy"
    assert out["apos_px"] == [pytest.approx(198.0), pytest.approx(175.64, abs=0.01)]
    assert out["opos_px"] == [pytest.approx(98.64, abs=0.01), pytest.approx(23.9, abs=0.01)]
    assert not out["apos_clamped"] and not out["opos_clamped"]


def test_pointing_payload_scales_to_the_frame_the_client_actually_sent():
    """Pixels are in THIS request's frame, which is why frame_size travels with them."""
    for mode_text in ("<apos_650>", "<apos><pos_500><pos_500>"):
        small = protocol.pointing_payload(mode_text, width=480, height=270)
        large = protocol.pointing_payload(mode_text, width=1920, height=1080)
        assert small["frame_size"] == [480, 270] and large["frame_size"] == [1920, 1080]
        # Same relative spot, four times the pixels. abs=0.05 because each payload is
        # rounded to 2dp before the comparison, not because the mapping is approximate.
        assert large["apos_px"][0] == pytest.approx(small["apos_px"][0] * 4, abs=0.05)
        assert large["apos_px"][1] == pytest.approx(small["apos_px"][1] * 4, abs=0.05)


def test_pointing_payload_omits_the_key_for_a_checkpoint_without_pointing():
    """A tpos-only or legacy checkpoint sends nothing, not a bag of nulls."""
    assert protocol.pointing_payload("<tpos_37><traj_128>", width=480, height=270) is None
    assert protocol.pointing_payload("", width=480, height=270) is None


def test_pointing_payload_reports_null_pixels_for_non_positional_values():
    """opos "not visible" and the apos rotate/stop directives have no pixel -- in either
    format. The client learns "no position", not a fabricated one."""
    grid = protocol.pointing_payload("<apos_1299><opos_0>", width=480, height=270)
    assert grid["apos_px"] is None and grid["opos_px"] is None
    assert grid["mode"] == "grid"

    half = protocol.pointing_payload(
        "<apos><rotr><opos><pos_205><pos_88>",
        width=480,
        height=270,
    )
    assert half["mode"] == "posxy"
    assert half["apos_px"] is None, "a rotate directive has no pixel"
    assert half["opos_px"] is not None


@pytest.mark.parametrize(
    ("raw_text", "apos_state", "opos_state"),
    [
        # grid ids and posxy sentinels are the same four states, spelled differently.
        ("<apos_1297><opos_114>", "rot_left", "point"),
        ("<apos_1298><opos_114>", "rot_right", "point"),
        ("<apos_1299><opos_114>", "stop", "point"),
        ("<apos_650><opos_0>", "point", "not_visible"),
        ("<apos_650><opos_114>", "point", "point"),
        ("<opos_114>", "none", "point"),
        ("<apos><rotl><opos><pos_205><pos_88>", "rot_left", "point"),
        ("<apos><rotr><opos><pos_205><pos_88>", "rot_right", "point"),
        ("<apos><stop><opos><pos_205><pos_88>", "stop", "point"),
        ("<apos><pos_412><pos_650><opos><novis>", "point", "not_visible"),
        ("<apos><rotr><opos><novis>", "rot_right", "not_visible"),
        ("<opos><pos_205><pos_88>", "none", "point"),
    ],
)
def test_pointing_payload_says_why_a_pixel_is_null(raw_text, apos_state, opos_state):
    """A null pixel alone makes "rotate left", "arrived, stop" and "target not visible"
    the same message. The state field is what keeps them distinct, in both formats."""
    out = protocol.pointing_payload(raw_text + "<act_l0_9>", width=480, height=270)

    assert out is not None, raw_text
    assert out["apos_state"] == apos_state
    assert out["opos_state"] == opos_state
    # A state other than "point" must come with a null pixel, and vice versa.
    assert (out["apos_px"] is None) == (apos_state != "point")
    assert (out["opos_px"] is None) == (opos_state != "point")


def test_posxy_sentinels_alone_still_produce_a_payload():
    """A step can be all directives -- rotate in place with the target lost."""
    out = protocol.pointing_payload("<apos><rotr><opos><novis>", width=480, height=270)

    assert out["mode"] == "posxy"
    assert out["apos_px"] is None and out["opos_px"] is None
    assert (out["apos_state"], out["opos_state"]) == ("rot_right", "not_visible")


def test_bare_stop_action_token_is_not_read_as_a_pointing_directive():
    """<stop> is also a legacy vln ACTION token; only <apos><stop> is a directive."""
    assert protocol.pointing_payload("<stop><traj_0>", width=480, height=270) is None


def test_posxy_visible_is_decoded_like_the_grid_spelling():
    """Without this a posxy checkpoint reports visible: null on every step."""
    wp = np.array([[1.0, 0.0, 0.0]], dtype="float32")

    seen = decode_prediction_signals(
        "<apos><pos_412><pos_650><opos><pos_205><pos_88><act_l0_9>", is_rvq=True, waypoints=wp
    )
    lost = decode_prediction_signals(
        "<apos><pos_412><pos_650><opos><novis><act_l0_9>", is_rvq=True, waypoints=wp
    )
    silent = decode_prediction_signals(
        "<apos><pos_412><pos_650><act_l0_9>", is_rvq=True, waypoints=wp
    )

    assert seen.visible is True
    assert lost.visible is False
    assert silent.visible is None, "no opos channel at all says nothing about visibility"


def test_pointing_payload_flags_clamped_edges_in_both_formats():
    """Both encoders clamp before quantising, so an edge value means "there OR beyond".
    The client cannot see that from the pixel, so the flag is the only carrier."""
    grid = protocol.pointing_payload("<apos_1296>", width=480, height=270)
    assert grid["apos_clamped"] is True

    axis = protocol.pointing_payload("<apos><pos_999><pos_500>", width=480, height=270)
    assert axis["apos_clamped"] is True
    interior = protocol.pointing_payload("<apos><pos_500><pos_500>", width=480, height=270)
    assert interior["apos_clamped"] is False


def test_pointing_payload_prefers_posxy_when_both_families_appear():
    """The bare <apos> marker cannot collide with <apos_K>, but if a checkpoint ever
    emitted both, the high-resolution one wins rather than a coin flip."""
    out = protocol.pointing_payload(
        "<apos_650><apos><pos_412><pos_650>",
        width=480,
        height=270,
    )
    assert out["mode"] == "posxy"
    assert out["apos_px"] == [pytest.approx(198.0), pytest.approx(175.64, abs=0.01)]
