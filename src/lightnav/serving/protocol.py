"""Wire-protocol helpers shared by the WebSocket server and local clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lightnav.vln_utils import (
    APOS_ROTL_ID,
    APOS_ROTR_ID,
    APOS_STOP_ID,
    decode_point_pixel_center,
    decode_posxy_center,
    decode_target_pos,
    parse_rvq_action_tokens,
    parse_traj_token,
    point_cell_is_clamped,
    posxy_channels,
    posxy_is_clamped,
    posxy_sentinels,
    safe_parse_apos_token,
    safe_parse_opos_token,
    safe_parse_tpos_token,
)


@dataclass(frozen=True)
class PredictionSignals:
    traj_id: int | None   # None for RVQ ckpts (no single <traj_k> id)
    stop: bool
    stop_reason: str | None
    visible: bool | None
    tpos_id: int | None
    apos_id: int | None = None   # dual-pointing ckpts only; None otherwise
    opos_id: int | None = None


def decode_prediction_signals(
    raw_text: str,
    vocab_size: int | None = None,
    *,
    is_rvq: bool = False,
    waypoints: np.ndarray | None = None,
    rvq_stop_l0: int | None = None,
) -> PredictionSignals:
    """Parse server-visible status fields from model text.

    Flat ckpts encode the action as one ``<traj_k>`` id (stop == id 0). RVQ ckpts
    emit ``<act_l*>`` level tokens instead — there is no traj id, so ``parse_traj_token``
    would raise; stop is read from the already-decoded ``waypoints`` (the agent zeros
    them on the stop action).

    Grounding prefix, in precedence order (a ckpt emits one family, never both):
      * v2 tracking ckpts emit ``<tpos_K>`` — a coarse (visible, azimuth, distance)
        bin, so ``visible`` comes from its decode.
      * dual-pointing ckpts emit ``<apos_K><opos_K>`` instead, where opos is the
        target's frame pixel and id 0 means "not visible" — so ``visible`` falls
        back to ``opos_id > 0``. Without it a pointing ckpt reports ``visible: null``
        on every step.
      * neither present (v1 / legacy vln flat) -> ``visible`` stays None.
    """
    tpos_id = safe_parse_tpos_token(raw_text)
    apos_id = safe_parse_apos_token(raw_text)
    opos_id = safe_parse_opos_token(raw_text)
    if tpos_id is not None:
        visible = bool(decode_target_pos(tpos_id)["visible"])
    elif opos_id is not None:
        visible = opos_id > 0
    else:
        # posxy checkpoints state the same thing in the other spelling: an <opos> point
        # means visible, <opos><novis> means not. Without this branch a posxy ckpt
        # reports visible: null on EVERY step — the same defect the grid branch above
        # was added to fix, one encoding later.
        if "opos" in posxy_channels(raw_text):
            visible = True
        elif posxy_sentinels(raw_text).get("opos") == "novis":
            visible = False
        else:
            visible = None
    if is_rvq:
        if waypoints is None:
            raise ValueError("decode_prediction_signals(is_rvq=True) needs waypoints for the stop signal")
        stop = bool(np.allclose(waypoints, 0.0))
        stop_reason = None
        if stop:
            if rvq_stop_l0 is None:
                # Older/custom bundles may not declare the dedicated stop code.
                stop_reason = "rvq_zero_waypoints"
            else:
                codes = parse_rvq_action_tokens(raw_text)
                stop_reason = (
                    "rvq_stop_code"
                    if codes[0] == int(rvq_stop_l0)
                    else "rvq_near_zero_waypoints"
                )
        return PredictionSignals(
            traj_id=None,
            stop=stop,
            stop_reason=stop_reason,
            visible=visible,
            tpos_id=tpos_id,
            apos_id=apos_id,
            opos_id=opos_id,
        )
    traj_id = parse_traj_token(raw_text)
    if vocab_size is not None and not (0 <= traj_id < int(vocab_size)):
        raise ValueError(f"traj id {traj_id} out of vocab range [0, {int(vocab_size)})")
    return PredictionSignals(
        traj_id=traj_id,
        stop=(traj_id == 0),
        stop_reason="flat_stop_token" if traj_id == 0 else None,
        visible=visible,
        tpos_id=tpos_id,
        apos_id=apos_id,
        opos_id=opos_id,
    )


# What a channel means when it has no pixel. Wire-neutral names, not token ids: the same
# four states exist in both formats, spelled differently (grid ids 1297-1299 / 0 versus the
# posxy <rotl>/<rotr>/<stop>/<novis> sentinels), and the client should not have to learn
# either spelling. Without these, every non-positional value collapses into the same null
# pixel and "rotate left", "arrived, stop" and "target not visible" become the same
# message — which is not something the client can recover from anywhere else.
_POSXY_STATE: dict[str, str] = {
    "rotl": "rot_left",
    "rotr": "rot_right",
    "stop": "stop",
    "novis": "not_visible",
}


def _grid_apos_state(apos_id: int | None) -> str:
    if apos_id is None or int(apos_id) == 0:
        return "none"
    return {
        APOS_ROTL_ID: "rot_left",
        APOS_ROTR_ID: "rot_right",
        APOS_STOP_ID: "stop",
    }.get(int(apos_id), "point")


def _grid_opos_state(opos_id: int | None) -> str:
    if opos_id is None:
        return "none"
    return "not_visible" if int(opos_id) == 0 else "point"


def pointing_payload(
    raw_text: str, *, width: int, height: int,
) -> dict[str, object] | None:
    """Pointing PIXELS in the frame the client just sent, for either token format.

    The client gets converted pixels, never token ids: the encoding is the server's
    business, and both formats in circulation decode to the same thing.

    * ``posxy`` — ``<apos><pos_x><pos_y>``, two shared axis tokens at 1000 bins each
      (~0.5 px on a 480-wide frame). Checked first: its bare ``<apos>`` marker cannot
      be confused with the grid's ``<apos_K>``, which requires ``_<digits>``.
    * ``grid`` — ``<apos_K>``, one id over a 48x27 cell grid (10 px cells at the
      480x270 label resolution).

    Returns None when the text carries neither, so a tpos-only or legacy checkpoint
    sends nothing rather than a bag of nulls.

    ``*_state`` says WHY a pixel is null, which the pixel cannot: ``point`` (it is not
    null), ``rot_left`` / ``rot_right`` / ``stop`` for an apos directive (grid ids
    1297-1299, posxy ``<rotl>`` / ``<rotr>`` / ``<stop>``), ``not_visible`` for an opos
    that saw nothing (grid id 0, posxy ``<novis>``), or ``none`` when the checkpoint did
    not emit that channel at all. Without it all four collapse into the same null and
    "rotate left", "arrived, stop" and "target not visible" become the same message —
    and only the last of them is recoverable elsewhere (from ``visible``). Note that an
    apos ``stop`` directive does NOT imply the response's top-level ``stop``, which
    reports the decoded action (zero waypoints), not the pointing channel.

    ``frame_size`` travels with the pixels because a pixel without the resolution it
    refers to is not a location: the client may render at a different size than it
    submitted, and the server resizes to yet another one internally.

    ``*_clamped`` marks a boundary cell or edge bin. Both encoders clamp u/v into range
    before quantising, so such a value means "at that edge OR beyond it" and its pixel
    is only the edge cell's centre. This is the one thing a client cannot recover from
    the pixel alone, and drawing a confident marker without it asserts more than the
    model said.
    """
    w, h = float(width), float(height)

    def _rounded(u: float, v: float) -> list[float]:
        return [round(u, 2), round(v, 2)]

    axis = posxy_channels(raw_text)
    sentinels = posxy_sentinels(raw_text)
    if axis or sentinels:
        payload: dict[str, object] = {
            "mode": "posxy",
            "frame_size": [int(width), int(height)],
        }
        for channel in ("apos", "opos"):
            bins = axis.get(channel)
            payload[f"{channel}_px"] = (
                None if bins is None else _rounded(*decode_posxy_center(bins[0], bins[1], w, h))
            )
            payload[f"{channel}_clamped"] = bins is not None and posxy_is_clamped(*bins)
            payload[f"{channel}_state"] = _POSXY_STATE.get(sentinels.get(channel, ""), "none") \
                if bins is None else "point"
        return payload

    apos_id = safe_parse_apos_token(raw_text)
    opos_id = safe_parse_opos_token(raw_text)
    if apos_id is None and opos_id is None:
        return None

    def _grid_pixel(point_id: int | None) -> list[float] | None:
        if not point_id:
            return None
        center = decode_point_pixel_center(point_id, w, h)
        return None if center is None else _rounded(float(center[0]), float(center[1]))

    return {
        "mode": "grid",
        "frame_size": [int(width), int(height)],
        "apos_px": _grid_pixel(apos_id),
        "opos_px": _grid_pixel(opos_id),
        "apos_clamped": bool(apos_id) and point_cell_is_clamped(int(apos_id)),
        "opos_clamped": bool(opos_id) and point_cell_is_clamped(int(opos_id)),
        "apos_state": _grid_apos_state(apos_id),
        "opos_state": _grid_opos_state(opos_id),
    }


def actions_payload(waypoints: np.ndarray, step: int) -> dict[str, object]:
    """Format waypoints for the WebSocket client: ``{"step": S, "actions": [[f, l, yaw], ...]}``."""
    return {
        "step": int(step),
        "actions": np.asarray(waypoints, dtype=np.float32).tolist(),
    }


def parse_actions_payload(actions_data: Any) -> list[list[float]]:
    """Parse new dict, legacy wrapped-list, or flat-list action payloads."""
    if actions_data is None:
        return []

    if isinstance(actions_data, dict):
        trajectories = actions_data.get("actions", [])
    else:
        trajectories = actions_data

    if not trajectories:
        return []

    if (
        isinstance(trajectories, list)
        and len(trajectories) == 1
        and isinstance(trajectories[0], list)
        and trajectories[0]
        and isinstance(trajectories[0][0], (list, tuple))
    ):
        trajectories = trajectories[0]

    if isinstance(trajectories, list) and trajectories and not isinstance(trajectories[0], (list, tuple)):
        if len(trajectories) % 3 != 0:
            raise ValueError("flat actions length must be divisible by 3")
        trajectories = [trajectories[i : i + 3] for i in range(0, len(trajectories), 3)]

    return [[float(wp[0]), float(wp[1]), float(wp[2])] for wp in trajectories]
