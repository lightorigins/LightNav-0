"""Wire contract of the WebSocket handler, driven with a fake socket and a fake service.

No engine, no GPU: ``make_handler`` is exercised end to end (JSON parsing, request
validation, JPEG decode, frame buffering, prediction responses, error codes) and the
argparse / engine-config plumbing is checked with ``build_engine`` monkeypatched.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from websockets.exceptions import ConnectionClosedOK

from lightnav.serving import ws_server


class FakeWebSocket:
    """Drives the handler's recv() loop: yields queued messages, then signals a
    normal close (websockets raises ConnectionClosed from recv())."""

    remote_address = ("127.0.0.1", 12345)

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.closed = False

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        raise ConnectionClosedOK(None, None)

    async def send(self, msg: str) -> None:
        self.sent.append(json.loads(msg))

    async def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, client_id=None):
        self.instruction = ""
        self.client_id = client_id
        self.observed: list[tuple[int, ...]] = []
        self.resets = 0
        self._buffer_len = 0

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.resets += 1
        self.observed = []
        self._buffer_len = 0

    def observe(self, rgb) -> None:
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        self.observed.append(tuple(rgb.shape))
        self._buffer_len = len(self.observed)


class FakeService:
    """``predictions`` are returned (or raised, for exceptions) in order by ``predict``."""

    def __init__(self, predictions=()):
        self._predictions = list(predictions)
        self.sessions: list[FakeSession] = []
        self.client_ids: list = []
        self.predict_calls = 0

    def make_session(self, client_id=None):
        self.client_ids.append(client_id)
        session = FakeSession(client_id)
        self.sessions.append(session)
        return session

    async def predict(self, session):
        self.predict_calls += 1
        item = self._predictions.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _pred(**overrides) -> SimpleNamespace:
    fields = dict(
        waypoints=np.zeros((10, 3), dtype=np.float32),
        stop=False,
        visible=None,
        traj_id=7,
        tpos_id=None,
        timings_ms={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _jpeg_b64(size=(4, 4)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _next(seq, instruction="go", size=(4, 4)) -> str:
    return json.dumps(
        {"action": "next", "data": {"seq": seq, "image": _jpeg_b64(size), "instruction": instruction}}
    )


async def _run(messages, service) -> FakeWebSocket:
    websocket = FakeWebSocket(messages)
    handler = ws_server.make_handler(service)
    await handler(websocket)
    return websocket


# -- engine / argparse plumbing --------------------------------------------------------------


def _engine_args(**overrides) -> Namespace:
    fields = dict(
        task="tracking",
        model_path="/path/to/model",
        backend="vllm_local",
        max_new_tokens=8,
        device="cuda",
        gpu_memory_utilization=0.85,
        max_batch_size=16,
        pool_spatial=None,
        num_history_frames=128,
        quantization=None,
        vit_cache_entries=None,
    )
    fields.update(overrides)
    return Namespace(**fields)


def test_build_engine_maps_tracking_task_to_trackvla(monkeypatch):
    seen = {}

    def fake_build_engine(cfg, task_type, max_new_tokens):
        seen.update(cfg=cfg, task_type=task_type, max_new_tokens=max_new_tokens)
        return object(), object()

    monkeypatch.setattr(ws_server, "build_engine", fake_build_engine)

    ws_server._build_engine(_engine_args())

    assert seen["task_type"] == "trackvla"
    assert seen["max_new_tokens"] == 8
    cfg = seen["cfg"]
    assert cfg.model_path == "/path/to/model"
    assert cfg.backend == "vllm_local"
    assert cfg.num_history_frames == 128
    assert cfg.max_num_seqs == 16  # --max_batch_size feeds the vLLM batch width
    assert cfg.gpu_memory_utilization == 0.85
    assert cfg.pool_spatial is None


def test_build_engine_maps_vln_task_to_vlnce_and_allows_config_history(monkeypatch):
    seen = {}

    def fake_build_engine(cfg, task_type, max_new_tokens):
        seen.update(cfg=cfg, task_type=task_type)
        return object(), object()

    monkeypatch.setattr(ws_server, "build_engine", fake_build_engine)

    ws_server._build_engine(_engine_args(task="vln", num_history_frames=None, pool_spatial=2))

    assert seen["task_type"] == "vlnce"
    assert seen["cfg"].num_history_frames is None
    assert seen["cfg"].pool_spatial == 2


def test_build_engine_plumbs_quantization_and_vit_cache(monkeypatch):
    seen = {}

    def fake_build_engine(cfg, task_type, max_new_tokens):
        seen["cfg"] = cfg
        return object(), object()

    monkeypatch.setattr(ws_server, "build_engine", fake_build_engine)
    ws_server._build_engine(
        _engine_args(quantization="fp8_llm_only", vit_cache_entries=1024)
    )
    assert seen["cfg"].quantization == "fp8_llm_only"
    assert seen["cfg"].vit_cache_entries == 1024


def test_quantization_flag_rejects_full_fp8(monkeypatch):
    monkeypatch.delenv("VLLM_QUANT", raising=False)
    parser = ws_server._build_parser()
    # only fp8_llm_only is a valid choice; plain "fp8" must be rejected
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model_path", "/m", "--traj_vocab_path", "/v", "--quantization", "fp8"]
        )
    args = parser.parse_args(
        ["--model_path", "/m", "--traj_vocab_path", "/v",
         "--quantization", "fp8_llm_only"]
    )
    assert args.quantization == "fp8_llm_only"


def test_quantization_and_cache_env_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_QUANT", "fp8_llm_only")
    monkeypatch.setenv("VLN_VIT_CACHE_ENTRIES", "1024")
    args = ws_server._build_parser().parse_args(
        ["--model_path", "/m", "--traj_vocab_path", "/v"]
    )
    assert args.quantization == "fp8_llm_only"
    assert args.vit_cache_entries == 1024

    monkeypatch.setenv("VLN_VIT_CACHE_ENTRIES", "not-an-int")
    with pytest.raises(SystemExit):
        ws_server._build_parser().parse_args(
            ["--model_path", "/m", "--traj_vocab_path", "/v"]
        )


def test_engine_task_mapping_rejects_unknown_tasks():
    assert ws_server._engine_task_for_server_task("tracking") == "trackvla"
    assert ws_server._engine_task_for_server_task("vln") == "vlnce"
    with pytest.raises(ValueError):
        ws_server._engine_task_for_server_task("objectnav")


def test_parser_defaults(monkeypatch):
    for key in ("TASK", "HOST", "PORT", "MAX_BATCH_SIZE", "MAX_WAIT_MS", "ACTION_TOKENIZER_BUNDLE"):
        monkeypatch.delenv(key, raising=False)

    args = ws_server._build_parser().parse_args(
        ["--model_path", "/path/to/ckpt", "--traj_vocab_path", "/path/to/vocab"]
    )

    assert args.K == 256 and args.horizon == 10
    assert args.task == "tracking" and args.backend == "vllm_local"
    assert args.host == "0.0.0.0" and args.port == 8050
    assert args.max_batch_size == 8 and args.max_wait_ms == 8.0
    assert args.max_new_tokens == 8 and args.gpu_memory_utilization == 0.85
    assert args.num_history_frames is None and args.pool_spatial is None
    assert args.ready_file is None and args.action_tokenizer_bundle is None
    assert args.no_warmup is False


def test_parser_reads_env_defaults(monkeypatch):
    monkeypatch.setenv("TASK", "vln")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("MAX_BATCH_SIZE", "2")
    monkeypatch.setenv("MAX_WAIT_MS", "4")
    monkeypatch.setenv("ACTION_TOKENIZER_BUNDLE", "/path/to/bundle")

    args = ws_server._build_parser().parse_args(["--model_path", "/path/to/ckpt"])

    assert args.task == "vln" and args.port == 9000 and args.host == "127.0.0.1"
    assert args.max_batch_size == 2 and args.max_wait_ms == 4.0
    assert args.action_tokenizer_bundle == "/path/to/bundle"


@pytest.mark.parametrize(
    "argv",
    [
        ["--model_path", "/path/to/ckpt"],
        [
            "--model_path",
            "/path/to/ckpt",
            "--traj_vocab_path",
            "/path/to/vocab",
            "--action_tokenizer_bundle",
            "/path/to/bundle",
        ],
    ],
)
def test_main_requires_exactly_one_action_decoder(monkeypatch, argv):
    monkeypatch.delenv("ACTION_TOKENIZER_BUNDLE", raising=False)
    monkeypatch.setattr(sys, "argv", ["lightnav-serve", *argv])
    with pytest.raises(SystemExit) as exc:
        ws_server.main()
    assert exc.value.code == 2


# -- login / reset -----------------------------------------------------------------------------


async def test_login_client_id_is_passed_to_session_creation():
    service = FakeService()
    ws = await _run([json.dumps({"action": "login", "data": {"clientId": "robot-client-7"}})], service)

    assert service.client_ids == ["robot-client-7"]
    assert ws.sent == [{"action": "login", "data": {"rc": 0, "msg": "ok"}}]
    assert service.sessions[0].instruction == ""


async def test_login_without_client_id_creates_an_anonymous_session():
    service = FakeService()
    ws = await _run([json.dumps({"action": "login", "data": {}})], service)

    assert ws.sent == [{"action": "login", "data": {"rc": 0, "msg": "ok"}}]
    assert service.client_ids == [None]


async def test_login_rejects_non_string_client_id_without_creating_session():
    class Service(FakeService):
        def make_session(self, client_id=None):
            raise AssertionError("session should not be created for invalid clientId")

    ws = await _run([json.dumps({"action": "login", "data": {"clientId": 7}})], Service())

    assert ws.sent == [{"action": "login", "data": {"rc": 400, "msg": "clientId must be a string"}}]


async def test_reset_clears_the_session_and_acks():
    service = FakeService()
    ws = await _run(
        [_next(1, ""), json.dumps({"action": "reset", "data": {}}), _next(2, "")], service
    )

    assert ws.sent[1] == {"action": "reset", "data": {"rc": 0, "msg": "ok"}}
    session = service.sessions[0]
    assert session.resets >= 2
    # The frame buffer restarted after reset: one frame, so step (buffer length) is 1.
    assert len(session.observed) == 1
    assert ws.sent[2] == {"action": "next", "data": {"rc": 0, "seq": 2, "msg": "image received"}}


# -- next --------------------------------------------------------------------------------------


@pytest.mark.parametrize("instruction", ["", None])
async def test_next_without_instruction_only_buffers_the_frame(instruction):
    service = FakeService()
    ws = await _run([_next(3, instruction, size=(8, 6))], service)

    assert ws.sent == [{"action": "next", "data": {"rc": 0, "seq": 3, "msg": "image received"}}]
    assert service.predict_calls == 0
    assert service.sessions[0].observed == [(6, 8, 3)]  # HWC RGB, decoded from the JPEG


async def test_next_accepts_an_integral_float_seq():
    service = FakeService()
    ws = await _run([_next(3.0, "")], service)
    assert ws.sent == [{"action": "next", "data": {"rc": 0, "seq": 3, "msg": "image received"}}]


async def test_success_response_matches_wire_contract():
    pred = _pred(
        waypoints=np.array([[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]], dtype=np.float32),
        stop=True,
        visible=False,
        traj_id=0,
        tpos_id=0,
        timings_ms={"llm_ms": 3.0},
        raw_text="<tpos_0><traj_0>",
    )
    service = FakeService([pred])

    ws = await _run([_next(5, "go to the chair")], service)

    assert ws.sent[0]["action"] == "next"
    data = ws.sent[0]["data"]
    assert data["rc"] == 0
    assert data["seq"] == 5
    assert data["actions"] == {
        "step": 1,  # frames in the buffer after observe()
        "actions": [[1.0, 0.25, 0.10000000149011612], [2.0, -0.5, -0.20000000298023224]],
    }
    assert data["stop"] is True
    assert data["visible"] is False
    assert data["raw_text"] == "<tpos_0><traj_0>"
    assert data["latency_ms"] >= 0.0
    assert data["timings_ms"] == {"llm_ms": 3.0}
    assert "pointing" not in data
    assert set(data) == {"rc", "seq", "actions", "latency_ms", "stop", "visible", "timings_ms", "raw_text"}
    assert service.sessions[0].instruction == "go to the chair"


async def test_step_counts_every_buffered_frame_since_reset():
    service = FakeService([_pred()])
    ws = await _run([_next(1, ""), _next(2, ""), _next(3, "go")], service)

    assert ws.sent[2]["data"]["actions"]["step"] == 3
    assert ws.sent[2]["data"]["visible"] is None
    assert ws.sent[2]["data"]["stop"] is False
    assert ws.sent[2]["data"]["raw_text"] == ""


async def test_raw_text_is_bounded_on_the_wire():
    long_text = "<traj_1>" * 100  # 800 chars
    service = FakeService([_pred(raw_text=long_text)])
    ws = await _run([_next(1)], service)

    raw = ws.sent[0]["data"]["raw_text"]
    assert raw.startswith("<traj_1>")
    assert len(raw) < 300


async def test_response_carries_pointing_pixels_in_the_clients_own_frame_size():
    """The client sent 480x270, so the pixels must be in 480x270 -- not in the label
    grid's resolution and not in whatever the model was fed."""
    service = FakeService([_pred(visible=True, raw_text="<apos_650><opos_114>")])
    ws = await _run([_next(1, "follow the man in the black shirt", size=(480, 270))], service)

    data = ws.sent[0]["data"]
    assert data["rc"] == 0
    assert data["pointing"]["mode"] == "grid"
    assert data["pointing"]["frame_size"] == [480, 270]
    assert 0.0 <= data["pointing"]["apos_px"][0] < 480.0
    assert 0.0 <= data["pointing"]["apos_px"][1] < 270.0
    # Pixels only: the token ids are the server's business.
    assert not any(key.endswith("_id") for key in data["pointing"])


async def test_response_carries_posxy_pointing_pixels_too():
    service = FakeService([_pred(raw_text="<apos><pos_412><pos_650><act_l0_91>")])
    ws = await _run([_next(1, "follow", size=(848, 480))], service)

    pointing = ws.sent[0]["data"]["pointing"]
    assert pointing["mode"] == "posxy"
    assert pointing["frame_size"] == [848, 480]
    assert pointing["apos_px"] == [pytest.approx(349.8, abs=0.01), pytest.approx(312.24, abs=0.01)]
    assert pointing["opos_px"] is None


async def test_response_omits_pointing_for_a_checkpoint_that_emits_none():
    service = FakeService([_pred(raw_text="<tpos_37><traj_7>")])
    ws = await _run([_next(1, "follow", size=(480, 270))], service)

    data = ws.sent[0]["data"]
    assert "pointing" not in data
    assert data["rc"] == 0, "a non-pointing checkpoint is still a normal response"


# -- errors ------------------------------------------------------------------------------------


async def test_predict_error_is_a_single_500_and_the_connection_continues():
    service = FakeService([ValueError("traj id 999 out of vocab range [0, 4)"), _pred()])
    ws = await _run([_next(1), _next(2)], service)

    assert ws.sent[0] == {
        "action": "next",
        "data": {"rc": 500, "seq": 1, "msg": "traj id 999 out of vocab range [0, 4)"},
    }
    assert len(ws.sent) == 2
    second = ws.sent[1]["data"]
    assert second["rc"] == 0 and second["seq"] == 2
    assert second["actions"]["step"] == 2  # the failed step's frame stays buffered


async def test_observe_exception_is_nonfatal_and_sends_one_500():
    class ObserveBoomSession(FakeSession):
        def observe(self, rgb) -> None:
            raise RuntimeError("observe exploded")

    class Service(FakeService):
        def make_session(self, client_id=None):
            return ObserveBoomSession(client_id)

    ws = await _run([_next(101), json.dumps({"action": "login", "data": {}})], Service())

    assert ws.sent == [
        {"action": "next", "data": {"rc": 500, "seq": 101, "msg": "observe exploded"}},
        {"action": "login", "data": {"rc": 0, "msg": "ok"}},
    ]


@pytest.mark.parametrize(
    ("message", "expected_action", "expected_msg"),
    [
        ("{not json", "error", "bad json"),
        (json.dumps([]), "error", "payload must be an object"),
        ("null", "error", "payload must be an object"),
        (json.dumps("payload"), "error", "payload must be an object"),
        (json.dumps({"action": "login", "data": None}), "login", "data must be an object"),
        (json.dumps({"action": "next", "data": []}), "next", "data must be an object"),
        (json.dumps({"action": "next", "data": 1}), "next", "data must be an object"),
        (json.dumps({"action": "next", "data": "data"}), "next", "data must be an object"),
        (json.dumps({"action": "next", "data": {"image": _jpeg_b64(), "instruction": "go"}}), "next", "missing seq"),
        (
            json.dumps({"action": "next", "data": {"seq": "bad", "image": _jpeg_b64(), "instruction": "go"}}),
            "next",
            "seq must be an integer",
        ),
        (
            json.dumps({"action": "next", "data": {"seq": True, "image": _jpeg_b64(), "instruction": "go"}}),
            "next",
            "seq must be an integer",
        ),
        (
            json.dumps({"action": "next", "data": {"seq": 1.5, "image": _jpeg_b64(), "instruction": "go"}}),
            "next",
            "seq must be an integer",
        ),
        (json.dumps({"action": "next", "data": {"seq": 1, "instruction": "go"}}), "next", "missing image"),
        (json.dumps({"action": "next", "data": {"seq": 1, "image": "", "instruction": "go"}}), "next", "missing image"),
        (
            json.dumps({"action": "next", "data": {"seq": 1, "image": _jpeg_b64(), "instruction": 5}}),
            "next",
            "instruction must be a string",
        ),
        (
            json.dumps({"action": "next", "data": {"seq": 1, "image": "not-base64!", "instruction": "go"}}),
            "next",
            "bad image",
        ),
        (
            json.dumps(
                {
                    "action": "next",
                    "data": {
                        "seq": 1,
                        "image": base64.b64encode(b"not a jpeg").decode("ascii"),
                        "instruction": "go",
                    },
                }
            ),
            "next",
            "bad image",
        ),
        (json.dumps({"action": "dance", "data": {}}), "dance", "unknown action: 'dance'"),
        (json.dumps({"data": {}}), "error", "unknown action"),
    ],
)
async def test_malformed_request_is_nonfatal_and_sends_one_400(message, expected_action, expected_msg):
    service = FakeService([_pred()])
    ws = await _run([message, _next(9)], service)

    assert len(ws.sent) == 2
    assert ws.sent[0]["action"] == expected_action
    assert ws.sent[0]["data"]["rc"] == 400
    assert ws.sent[0]["data"]["msg"].startswith(expected_msg)
    # The connection survives and the next well-formed request succeeds.
    assert ws.sent[1]["data"]["rc"] == 0 and ws.sent[1]["data"]["seq"] == 9


async def test_400_echoes_seq_only_when_it_was_parsed():
    service = FakeService()
    ws = await _run(
        [
            json.dumps({"action": "next", "data": {"seq": 4, "instruction": "go"}}),
            json.dumps({"action": "next", "data": {"image": _jpeg_b64(), "instruction": "go"}}),
        ],
        service,
    )
    assert ws.sent[0]["data"] == {"rc": 400, "msg": "missing image", "seq": 4}
    assert "seq" not in ws.sent[1]["data"]


async def test_bad_image_is_rejected_before_it_reaches_the_buffer():
    service = FakeService()
    await _run(
        [json.dumps({"action": "next", "data": {"seq": 1, "image": "not-base64!", "instruction": "go"}})],
        service,
    )
    assert service.sessions[0].observed == []


@pytest.mark.parametrize("send_error", [RuntimeError("send exploded"), ConnectionClosedOK(None, None)])
async def test_send_failure_ends_handler_without_retry(send_error):
    class SendBoomWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__([_next(102), _next(103)])
            self.send_attempts = 0

        async def send(self, msg: str) -> None:
            self.send_attempts += 1
            raise send_error

    service = FakeService([_pred(), _pred()])
    websocket = SendBoomWebSocket()
    handler = ws_server.make_handler(service)

    await handler(websocket)  # must not raise

    assert websocket.send_attempts == 1
    assert service.predict_calls == 1  # the second request was never processed


async def test_error_response_send_failure_ends_handler_too():
    class SendBoomWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__(["{not json", _next(1)])
            self.send_attempts = 0

        async def send(self, msg: str) -> None:
            self.send_attempts += 1
            raise RuntimeError("send exploded")

    service = FakeService([_pred()])
    websocket = SendBoomWebSocket()
    await ws_server.make_handler(service)(websocket)

    assert websocket.send_attempts == 1
    assert service.predict_calls == 0


# -- episode recording -------------------------------------------------------------------------


class FakeConnectionRecorder:
    def __init__(self, label, *, fail=None):
        self.label = label
        self.fail = fail
        self.begins = 0
        self.steps: list[dict] = []
        self.closed = False

    def begin_episode(self) -> None:
        if self.fail == "begin_episode":
            raise RuntimeError("begin_episode exploded")
        self.begins += 1

    def record_step(self, **kwargs) -> None:
        if self.fail == "record_step":
            raise OSError("disk full")
        self.steps.append(kwargs)

    def close(self) -> None:
        if self.fail == "close":
            raise RuntimeError("close exploded")
        self.closed = True


class FakeRecorder:
    """``EpisodeRecorder``-like: hands out one FakeConnectionRecorder per connection."""

    def __init__(self, fail=None):
        self.fail = fail
        self.conns: list[FakeConnectionRecorder] = []

    def begin_connection(self, label=None):
        if self.fail == "begin_connection":
            raise RuntimeError("begin_connection exploded")
        conn = FakeConnectionRecorder(label, fail=self.fail)
        self.conns.append(conn)
        return conn

    @property
    def steps(self) -> list[dict]:
        return [step for conn in self.conns for step in conn.steps]


async def _run_recorded(messages, service, recorder) -> FakeWebSocket:
    websocket = FakeWebSocket(messages)
    await ws_server.make_handler(service, recorder=recorder)(websocket)
    return websocket


def _login(client_id) -> str:
    return json.dumps({"action": "login", "data": {"clientId": client_id}})


_RESET = json.dumps({"action": "reset", "data": {}})


def _without_latency(sent: list[dict]) -> list[dict]:
    out = []
    for msg in sent:
        data = {k: v for k, v in msg["data"].items() if k != "latency_ms"}
        out.append({**msg, "data": data})
    return out


def test_decode_jpeg_b64_with_bytes_returns_the_clients_own_bytes():
    b64 = _jpeg_b64(size=(8, 6))
    rgb, raw = ws_server._decode_jpeg_b64_with_bytes(b64)
    assert raw == base64.b64decode(b64)
    assert rgb.shape == (6, 8, 3) and rgb.dtype == np.uint8
    assert np.array_equal(ws_server._decode_jpeg_b64(b64), rgb)


@pytest.mark.parametrize(
    ("client_id", "expected"),
    [
        ("robot-7", "robot-7"),
        ("cam_01.left", "cam_01.left"),
        ("a" * 64, "a" * 64),
        ("a" * 65, None),
        ("../evil", None),
        ("robot 7", None),
        (".hidden", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_recorder_label_accepts_only_plainly_safe_client_ids(client_id, expected):
    assert ws_server._recorder_label(client_id) == expected


async def test_predicted_next_is_recorded_with_the_raw_jpeg_bytes_step_seq_and_pointing():
    b64 = _jpeg_b64(size=(480, 270))
    predicted = json.dumps(
        {"action": "next", "data": {"seq": 9, "image": b64, "instruction": "follow the man"}}
    )
    pred = _pred(
        waypoints=np.array([[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]], dtype=np.float32),
        visible=True,
        raw_text="<apos_650><opos_114>",
    )
    recorder = FakeRecorder()

    ws = await _run_recorded(
        [_login("robot-7"), _RESET, _next(1, ""), _next(2, ""), predicted],
        FakeService([pred]),
        recorder,
    )

    (conn,) = recorder.conns
    assert conn.label == "robot-7"
    assert conn.begins == 1  # the reset opened the episode; the prediction reused it
    (step,) = conn.steps
    assert step["image"] == base64.b64decode(b64)  # the client's bytes, not a re-encode
    assert step["step"] == 3  # frames buffered since reset, as in the response
    assert step["seq"] == 9
    assert step["instruction"] == "follow the man"
    assert step["waypoints"] == pred.waypoints.tolist()
    assert step["stop"] is False and step["visible"] is True
    assert step["raw_text"] == "<apos_650><opos_114>"
    assert isinstance(step["latency_ms"], float) and step["latency_ms"] >= 0.0
    response = ws.sent[-1]["data"]
    assert response["actions"]["step"] == 3
    assert step["pointing"] == response["pointing"]
    assert step["pointing"]["frame_size"] == [480, 270]
    assert conn.closed is True


async def test_buffer_only_next_is_not_recorded():
    recorder = FakeRecorder()
    ws = await _run_recorded([_next(1, ""), _next(2, None)], FakeService(), recorder)

    assert recorder.steps == []
    assert recorder.conns == []  # nothing to record, so no connection recorder either
    assert [m["data"]["msg"] for m in ws.sent] == ["image received", "image received"]


async def test_first_prediction_opens_an_episode_when_none_is_open_and_reset_opens_another():
    recorder = FakeRecorder()
    await _run_recorded(
        [_next(1, "go"), _next(2, "go"), _RESET, _next(3, "go")],
        FakeService([_pred(), _pred(), _pred()]),
        recorder,
    )

    (conn,) = recorder.conns
    assert conn.label is None  # no login: the recorder picks its default label
    assert conn.begins == 2
    assert [s["seq"] for s in conn.steps] == [1, 2, 3]
    assert [s["step"] for s in conn.steps] == [1, 2, 1]


async def test_unsafe_client_id_is_not_used_as_a_directory_name():
    recorder = FakeRecorder()
    await _run_recorded([_login("../evil id"), _next(1, "go")], FakeService([_pred()]), recorder)
    assert recorder.conns[0].label is None


async def test_failed_prediction_is_not_recorded():
    recorder = FakeRecorder()
    ws = await _run_recorded(
        [_next(1, "go"), _next(2, "go")], FakeService([ValueError("boom"), _pred()]), recorder
    )
    assert ws.sent[0]["data"]["rc"] == 500
    assert [s["seq"] for s in recorder.steps] == [2]


@pytest.mark.parametrize("fail", ["begin_connection", "begin_episode", "record_step", "close"])
async def test_a_raising_recorder_leaves_every_response_unchanged(fail):
    messages = [_login("robot-7"), _RESET, _next(1, ""), _next(2, "go"), "{not json", _next(3, "go")]
    predictions = [_pred(raw_text="<apos_650><opos_114>"), _pred(stop=True)]

    baseline = await _run(list(messages), FakeService(list(predictions)))
    recorded = await _run_recorded(list(messages), FakeService(list(predictions)), FakeRecorder(fail=fail))

    assert _without_latency(recorded.sent) == _without_latency(baseline.sent)
    assert [m["data"]["rc"] for m in recorded.sent] == [0, 0, 0, 0, 400, 0]


def _recording_args(tmp_path, **overrides) -> Namespace:
    fields = dict(
        task="tracking",
        model_path="/path/to/model",
        record_dir=str(tmp_path / "rec"),
        record_fps=12,
        record_timeline="per_step",
        record_images=True,
        cam_hfov_deg=112.0,
        cam_height=0.4,
        traj_forward_offset=None,
        waypoint_dt_s=0.2,
    )
    fields.update(overrides)
    return Namespace(**fields)


def test_parser_recording_defaults(monkeypatch):
    for key in ("RECORD_DIR", "RECORD_FPS", "RECORD_TIMELINE", "RECORD_IMAGES", "CAM_HFOV_DEG",
                "CAM_HEIGHT", "TRAJ_FORWARD_OFFSET", "WAYPOINT_DT_S"):
        monkeypatch.delenv(key, raising=False)

    args = ws_server._build_parser().parse_args(
        ["--model_path", "/path/to/ckpt", "--traj_vocab_path", "/path/to/vocab"]
    )

    assert args.record_dir == ""
    assert args.record_fps == 10 and args.record_timeline == "realtime"
    assert args.record_images is True
    assert args.cam_hfov_deg == 90.0 and args.cam_height == 0.5
    assert args.traj_forward_offset is None and args.waypoint_dt_s == 0.1
    assert ws_server._build_recorder(args) is None  # off by default


def test_parser_reads_recording_env_defaults(monkeypatch):
    monkeypatch.setenv("RECORD_DIR", "/path/to/records")
    monkeypatch.setenv("RECORD_FPS", "15")
    monkeypatch.setenv("RECORD_TIMELINE", "per_step")
    monkeypatch.setenv("RECORD_IMAGES", "0")
    monkeypatch.setenv("CAM_HFOV_DEG", "112")
    monkeypatch.setenv("CAM_HEIGHT", "0.4")
    monkeypatch.setenv("TRAJ_FORWARD_OFFSET", "0.3")
    monkeypatch.setenv("WAYPOINT_DT_S", "0.25")

    args = ws_server._build_parser().parse_args(["--model_path", "/path/to/ckpt"])

    assert args.record_dir == "/path/to/records"
    assert args.record_fps == 15 and args.record_timeline == "per_step"
    assert args.record_images is False
    assert args.cam_hfov_deg == 112.0 and args.cam_height == 0.4
    assert args.traj_forward_offset == 0.3 and args.waypoint_dt_s == 0.25


def test_parser_recording_flags_override_env(monkeypatch):
    monkeypatch.setenv("RECORD_DIR", "/env/records")
    monkeypatch.setenv("RECORD_IMAGES", "1")
    monkeypatch.setenv("TRAJ_FORWARD_OFFSET", "   ")  # blank means unset

    args = ws_server._build_parser().parse_args(
        ["--model_path", "/path/to/ckpt", "--record_dir", "/cli/records", "--no_record_images",
         "--record_timeline", "per_step", "--traj_forward_offset", "0.6"]
    )
    assert args.record_dir == "/cli/records" and args.record_images is False
    assert args.record_timeline == "per_step" and args.traj_forward_offset == 0.6

    with pytest.raises(SystemExit):
        ws_server._build_parser().parse_args(
            ["--model_path", "/path/to/ckpt", "--record_timeline", "wallclock"]
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("", True), ("  ", True), ("0", False), ("false", False), ("No", False),
     ("off", False), ("1", True), ("true", True), ("anything", True)],
)
def test_env_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("RECORD_IMAGES", raising=False)
    else:
        monkeypatch.setenv("RECORD_IMAGES", value)
    assert ws_server._env_flag("RECORD_IMAGES", True) is expected


def test_build_recorder_configures_an_episode_recorder_from_the_args(tmp_path):
    from lightnav.viz import EpisodeRecorder

    recorder = ws_server._build_recorder(_recording_args(tmp_path, traj_forward_offset=0.3))

    assert isinstance(recorder, EpisodeRecorder)
    assert recorder.run_dir.parent == tmp_path / "rec" and recorder.run_dir.is_dir()
    assert recorder.task == "tracking" and recorder.model_path == "/path/to/model"
    assert recorder.hfov_deg == 112.0 and recorder.cam_height == 0.4
    assert recorder.forward_offset == 0.3 and recorder.waypoint_dt_s == 0.2
    assert recorder.video_fps == 12 and recorder.timeline == "per_step"
    assert recorder.save_images is True
    recorder.close()

    assert ws_server._build_recorder(_recording_args(tmp_path, record_dir="")) is None
    off = ws_server._build_recorder(_recording_args(tmp_path, record_images=False))
    assert off.save_images is False and off.forward_offset is None
    off.close()


async def test_record_dir_end_to_end_writes_an_episode_lightnav_render_can_find(tmp_path):
    from lightnav.viz.render_episode import find_episode_dirs, load_manifest, load_records

    recorder = ws_server._build_recorder(_recording_args(tmp_path))
    b64 = _jpeg_b64(size=(64, 36))
    predicted = json.dumps(
        {"action": "next", "data": {"seq": 5, "image": b64, "instruction": "follow the man"}}
    )
    pred = _pred(visible=True, raw_text="<apos_650><opos_114>")

    ws = await _run_recorded([_login("robot-7"), _RESET, _next(1, ""), predicted], FakeService([pred]), recorder)
    recorder.close()

    assert ws.sent[-1]["data"]["rc"] == 0
    (ep,) = find_episode_dirs([tmp_path / "rec"])
    assert ep.parent.name == "robot-7" and ep.name == "episode_000"
    assert (ep / "image_000002.jpg").read_bytes() == base64.b64decode(b64)
    assert not (ep / "actions.jsonl").exists()
    (record,) = load_records(ep)
    assert record["step"] == 2 and record["seq"] == 5
    assert record["pointing"] == ws.sent[-1]["data"]["pointing"]
    assert record["frame_size"] == [64, 36]
    manifest = load_manifest(ep)
    assert manifest["conn"] == "robot-7" and manifest["task"] == "tracking"
    assert manifest["overlay_hfov_deg"] == 112.0 and manifest["video_timeline"] == "per_step"
    assert manifest["instruction"] == "follow the man" and manifest["frame_size"] == [64, 36]


def test_build_recorder_fails_at_startup_when_record_dir_is_unusable(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with pytest.raises(OSError):
        ws_server._build_recorder(_recording_args(tmp_path, record_dir=str(blocker / "rec")))
