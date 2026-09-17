# WebSocket wire protocol

`lightnav-serve` speaks a small JSON-over-WebSocket protocol. Every frame is one
JSON text message; every request gets exactly one response. The reference client
is `lightnav-ws-client` (`src/lightnav/cli/ws_client.py`); the EVT-Bench
client in `evt_bench/trackvla_client_agent.py` speaks the same protocol.

The server accepts request frames up to 64 MiB (`max_size=64*1024*1024`), so a
full-resolution JPEG per message is fine. Responses are small (a few KiB), so the
client's own inbound limit rarely matters; raise it if your library complains.

## Session model

* One connection = one session: a frame buffer (the last `num_history_frames`
  frames, or the whole episode for slow-fast checkpoints) plus a per-session
  ViT cache. The session is created lazily on the first message of any type.
* Nothing persists across connections. `clientId` is log metadata, not
  authentication, and does not resume state.
* `reset` clears the frame buffer, the frame ids and the ViT cache. Send it at
  every episode boundary.
* All server-side inference for concurrent connections is micro-batched onto one
  shared engine; a client never sees this except through `latency_ms`.

## Requests

```json
{"action": "login", "data": {"clientId": "<string, optional>"}}
{"action": "reset", "data": {}}
{"action": "next",  "data": {"seq": <int>, "image": "<base64 JPEG>", "instruction": "<string or null>", "prompt_pointing": {"apos_id": <int>, "opos_id": <int>}}}
{"action": "next",  "data": {"seq": <int>, "image": "<base64 JPEG>", "instruction": "<string or null>", "prompt_pointing": {"opos_id": <int>}}}
```

* `clientId`, when present, must be a string or `null`.
* `seq` must be an integer (booleans are rejected; a float is accepted only if it
  is finite and integral). It is echoed back and otherwise unused.
* `image` must be a non-empty base64 string of a JPEG (or PNG) frame. The server
  decodes it to RGB and resizes to the checkpoint's `video_size` internally; send
  the camera frame at its native resolution.
* `instruction` may be `""` or `null`. **Buffer-only semantics:** the frame is
  always appended to the session buffer first; if the instruction is empty the
  server acknowledges the frame and does not run the model. This lets a client
  pre-fill the history window before the first prediction.
* `prompt_pointing` is optional. Supplying both fields keeps the legacy one-shot path:
  the server appends `<apos_K><opos_K>` before generation. Supplying only `opos_id`
  opts into staged decoding for grid dual-pointing checkpoints: the model first
  generates one APOS token, then the server appends that APOS plus the supplied OPOS
  and generates the action token(s). `apos_id` is in `[0, 1300)` and `opos_id` is in
  `[0, 1297)`. Omitting the object keeps the original prompt unchanged. Staged mode
  reuses the ViT result but performs two LLM calls; it does not reuse LLM KV cache.

## Responses

### login / reset

```json
{"action": "login", "data": {"rc": 0, "msg": "ok"}}
{"action": "reset", "data": {"rc": 0, "msg": "ok"}}
```

### next, buffer-only (instruction empty or null)

```json
{"action": "next", "data": {"rc": 0, "seq": 17, "msg": "image received"}}
```

### next, prediction

```json
{
  "action": "next",
  "data": {
    "rc": 0,
    "seq": 17,
    "actions": {
      "step": 12,
      "actions": [[0.31, 0.02, 0.05], [0.62, 0.04, 0.09], "... H rows"]
    },
    "latency_ms": 143.2,
    "stop": false,
    "stop_reason": null,
    "visible": true,
    "timings_ms": {"batch_size": 1.0, "queue_wait_ms": 0.4, "vit_ms": 61.0, "llm_ms": 70.1, "...": 0.0},
    "raw_text": "<tpos_12><traj_57>",
    "pointing": {"...": "only for pointing checkpoints, see below"}
  }
}
```

| field | type | meaning |
|---|---|---|
| `actions.step` | int | number of frames in the session buffer after this frame was appended (`min(frames since reset, num_history_frames)`; total frames for slow-fast checkpoints). Informational. |
| `actions.actions` | `[[float, float, float] x H]` | the predicted waypoint chunk, see *Waypoint convention*. Values are float32 converted to JSON numbers (e.g. `0.10000000149011612`). |
| `latency_ms` | float | wall time of the prediction on the server, including micro-batch queue wait, ViT, LLM decode and waypoint decode. |
| `stop` | bool | the model predicted the stop action (the decoded waypoint chunk is all zeros). |
| `stop_reason` | string or null | why `stop` is true: `rvq_stop_code` for an explicit RVQ stop code, `rvq_near_zero_waypoints` when RVQ decoding lands inside the near-zero threshold, `rvq_zero_waypoints` when an older/custom RVQ bundle has no declared stop code, or `flat_stop_token` for `<traj_0>`. `null` while moving. |
| `visible` | bool or null | whether the target is visible, when the checkpoint emits a grounding token: `<tpos_k>` decodes to a visibility bit; grid pointing uses `opos_id > 0`; `posxy` pointing uses `<opos>` present (true) / `<novis>` (false). `null` for checkpoints without grounding tokens. |
| `timings_ms` | object | server stage timings (`batch_size`, `queue_wait_ms`, `build_sample_ms`, `vit_ms`, `llm_ms`, `decode_waypoints_ms`, `batch_total_ms`, plus ViT cache counters when available). Optional for clients; may change. |
| `raw_text` | string | the model's raw output tokens for this step, truncated to 256 characters (with `...`) if longer. For debugging. |
| `pointing` | object | present only when the output carries `<apos*>`/`<opos*>` pointing tokens. |

Clients that only need to drive a robot read `actions.actions[0]` (the first
waypoint), `stop`, and optionally `visible`.

### pointing payload

Pointing checkpoints emit a pixel location for the action target (`apos`) and/or
the tracked object (`opos`). The server converts token ids to pixels in the frame
**the client sent** (`frame_size = [width, height]` of the decoded request image,
before the server's internal resize). Pixels are rounded to 2 decimals.

Grid encoding (`<apos_K>` / `<opos_K>`, one id over a 48x27 cell grid):

```json
{"mode": "grid", "frame_size": [640, 480],
 "apos_px": [412.5, 262.5], "opos_px": null,
 "apos_clamped": false, "opos_clamped": false,
 "apos_state": "point", "opos_state": "not_visible"}
```

Axis encoding (`<apos><pos_x><pos_y>`, two 1000-bin axis tokens per channel):

```json
{"mode": "posxy", "frame_size": [640, 480],
 "apos_px": [411.84, 261.6], "apos_clamped": false, "apos_state": "point",
 "opos_px": null, "opos_clamped": false, "opos_state": "none"}
```

| field | values |
|---|---|
| `*_px` | `[u, v]` pixel in the client frame, or `null` when the channel carries no pixel |
| `*_clamped` | `true` when the encoded value sits on a boundary cell/bin, i.e. "at that edge or beyond it" |
| `apos_state` | `none` (channel absent), `point`, `rot_left`, `rot_right`, `stop` |
| `opos_state` | `none`, `point`, `not_visible` |

An `apos_state` of `stop` is a pointing directive; the top-level `stop` reports
the decoded action and is the field to act on.

## Errors

| rc | when | shape |
|---|---|---|
| 400 | malformed request | `{"action": "<action or 'error'>", "data": {"rc": 400, "msg": "<reason>", "seq": N}}` (`seq` only if it was parsed) |
| 500 | the model output could not be decoded, or any other server-side error (while predicting, or while creating the session on `login`/`reset` — then without `seq`) | `{"action": "next", "data": {"rc": 500, "seq": N, "msg": "<str(exc)>"}}` |

400 messages: `bad json: ...`, `payload must be an object`, `data must be an
object`, `missing seq`, `seq must be an integer`, `missing image`, `instruction
must be a string`, `bad image: ...`, `clientId must be a string`, `unknown
action: '<x>'`. The `action` echoed in an error response is the request's
`action` if it was a non-empty string, else `"error"`.

The connection stays open after a 400 or 500; the client decides whether to
reuse its last action or stop. A response is never sent twice: if the server
fails to send a response the connection is closed. Model warnings (a
trajectory id out of the vocabulary range, a missing RVQ level) surface as rc
500 with the parser's message, never as silently wrong waypoints.

## Waypoint convention

Each row of `actions.actions` is a robot-local displacement

```
[forward_m, lateral_m, yaw_rad]      +lateral = left, +yaw = counter-clockwise
```

expressed relative to the robot pose at the current frame, one row per future
step (`H` rows, `H = --horizon`). Rows are cumulative poses along the predicted
chunk, so `actions.actions[0]` is the pose one step ahead. A stop is an all-zero
chunk with `stop: true`.

To turn the first waypoint into a normalised `[vx, vy, vyaw]` command in
`[-1, 1]` the reference clients divide by per-step maxima and clip:
`vx = clip(fwd / 0.375)`, `vy = clip(lat / 0.25)`, `vyaw = clip(yaw / (pi/20))`.
For a Habitat `velocity_control` command use
`lightnav.velocity.first_waypoint_to_velocity_cmd`.

## Minimal client (Python)

```python
import base64, json
from websockets.sync.client import connect

with connect("ws://localhost:8050", max_size=64 * 1024 * 1024) as ws:
    ws.send(json.dumps({"action": "login", "data": {"clientId": "demo"}}))
    assert json.loads(ws.recv())["data"]["rc"] == 0
    ws.send(json.dumps({"action": "reset", "data": {}}))
    ws.recv()
    for seq, jpeg_bytes in enumerate(frames):
        ws.send(json.dumps({"action": "next", "data": {
            "seq": seq,
            "image": base64.b64encode(jpeg_bytes).decode(),
            "instruction": "follow the person in the red shirt",
        }}))
        data = json.loads(ws.recv())["data"]
        if data["rc"] == 0 and "actions" in data:
            fwd, lat, yaw = data["actions"]["actions"][0]
```
