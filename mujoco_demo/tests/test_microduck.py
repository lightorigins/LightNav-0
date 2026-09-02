from pathlib import Path

import mujoco
import numpy as np
import pytest
from vln_mujoco.robots.microduck import (
    BASE_JOINT_NAME,
    CAMERA_NAME,
    EXPECTED_COMMAND_NAMES,
    EXPECTED_OBSERVATION_NAMES,
    IMU_SENSOR_NAME,
    MODEL_PREFIX,
    TRUNK_BODY_NAME,
    MicroDuckBackend,
    MicroDuckPolicy,
    build_model,
    shape_microduck_command,
)

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
DEFAULT_POSE = np.asarray(
    [
        0.0,
        -0.087,
        -0.458,
        -0.005,
        0.453,
        0.349,
        0.349,
        0.0,
        0.0,
        0.0,
        0.087,
        0.458,
        0.005,
        -0.453,
    ],
    dtype=np.float32,
)


class FakeTensorInfo:
    def __init__(self, name: str, shape: list[int], tensor_type: str) -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class FakeMetadata:
    def __init__(self, values: dict[str, str]) -> None:
        self.custom_metadata_map = values


class FakeSession:
    def __init__(
        self,
        *,
        joint_names: tuple[str, ...] = JOINT_NAMES,
        action: np.ndarray | None = None,
    ) -> None:
        self._inputs = [FakeTensorInfo("obs", [1, 61], "tensor(float)")]
        self._outputs = [FakeTensorInfo("actions", [1, 14], "tensor(float)")]
        self._metadata = FakeMetadata(
            {
                "joint_names": ",".join(joint_names),
                "default_joint_pos": ",".join(str(value) for value in DEFAULT_POSE),
                "action_scale": "0.5",
                "observation_names": ",".join(EXPECTED_OBSERVATION_NAMES),
                "command_names": ",".join(EXPECTED_COMMAND_NAMES),
            }
        )
        self.action = (
            action
            if action is not None
            else np.arange(14, dtype=np.float32).reshape(1, 14) / 100.0
        )
        self.last_observation: np.ndarray | None = None

    def get_inputs(self) -> list[FakeTensorInfo]:
        return self._inputs

    def get_outputs(self) -> list[FakeTensorInfo]:
        return self._outputs

    def get_modelmeta(self) -> FakeMetadata:
        return self._metadata

    def run(
        self,
        _output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.last_observation = input_feed["obs"].copy()
        return [self.action.copy()]


def write_test_models(tmp_path: Path) -> tuple[Path, Path]:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
        <mujoco model="test_scene">
          <worldbody>
            <geom name="floor" type="plane" size="20 20 0.1" />
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    bodies = "\n".join(
        f"""
        <body name="{name}_body" pos="0 0 0.01">
          <joint name="{name}" type="hinge" axis="0 1 0" />
          <geom type="capsule" size="0.005 0.01" mass="0.01" />
        </body>
        """
        for name in JOINT_NAMES
    )
    actuators = "\n".join(
        f'<position name="{name}" joint="{name}" kp="1" />'
        for name in JOINT_NAMES
    )
    robot = tmp_path / "robot.xml"
    robot.write_text(
        f"""
        <mujoco model="microduck_test">
          <worldbody>
            <body name="trunk_base" pos="0 0 0.12">
              <freejoint name="trunk_base_freejoint" />
              <geom type="box" size="0.03 0.02 0.02" mass="1" />
              <site name="imu" size="0.005" />
              <camera name="head_camera" pos="0.03 0 0" />
              {bodies}
            </body>
          </worldbody>
          <sensor><gyro name="imu_ang_vel" site="imu" /></sensor>
          <actuator>{actuators}</actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return scene, robot


def test_microduck_command_stops_inside_deadband() -> None:
    assert shape_microduck_command(0.05, 0.01) == (0.0, 0.0)
    assert shape_microduck_command(float("nan"), 0.4) == (0.0, 0.0)


def test_microduck_command_enters_stable_gait_range() -> None:
    linear, angular = shape_microduck_command(0.15, 0.4)
    assert linear == pytest.approx(0.28)
    assert angular == pytest.approx(0.4)


def test_microduck_command_is_clamped_to_policy_range() -> None:
    assert shape_microduck_command(2.0, -4.0) == pytest.approx((0.3, -1.5))


def test_microduck_model_is_namespaced(tmp_path: Path) -> None:
    scene, robot = write_test_models(tmp_path)
    model = build_model(robot, scene)

    assert model.nu == 14
    assert model.body(TRUNK_BODY_NAME).id >= 0
    assert model.joint(BASE_JOINT_NAME).id >= 0
    assert model.camera(CAMERA_NAME).id >= 0
    assert model.sensor(IMU_SENSOR_NAME).id >= 0
    assert all(
        model.actuator(index).name == f"{MODEL_PREFIX}{name}"
        for index, name in enumerate(JOINT_NAMES)
    )


def test_policy_uses_metadata_for_observation_and_action(tmp_path: Path) -> None:
    scene, robot = write_test_models(tmp_path)
    model = build_model(robot, scene)
    data = mujoco.MjData(model)
    session = FakeSession()
    policy = MicroDuckPolicy(model, data, tmp_path / "unused.onnx", session)
    data.qpos[policy.joint_qpos] = policy.default_pose
    mujoco.mj_forward(model, data)

    assert policy.set_velocity(0.15, 0.4) == pytest.approx((0.28, 0.4))
    observation = policy.observation()
    assert observation.shape == (61,)
    assert observation[-13] == pytest.approx(0.28)
    assert observation[-11] == pytest.approx(0.4)

    policy.step()

    assert session.last_observation is not None
    assert session.last_observation.shape == (1, 61)
    assert data.ctrl == pytest.approx(DEFAULT_POSE + session.action[0] * 0.5)


def test_policy_rejects_mismatched_joint_order(tmp_path: Path) -> None:
    scene, robot = write_test_models(tmp_path)
    model = build_model(robot, scene)
    data = mujoco.MjData(model)
    wrong_order = tuple(reversed(JOINT_NAMES))

    with pytest.raises(ValueError, match="joint_names"):
        MicroDuckPolicy(
            model,
            data,
            tmp_path / "unused.onnx",
            FakeSession(joint_names=wrong_order),
        )


def test_microduck_backend_implements_shared_runtime_contract(tmp_path: Path) -> None:
    scene, robot = write_test_models(tmp_path)
    backend = MicroDuckBackend(
        robot,
        tmp_path / "unused.onnx",
        policy_session=FakeSession(),
        scene_source=scene,
    )

    backend.step((0.15, 0.4))
    state = backend.state()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    assert state.telemetry["policy_command"] == pytest.approx(
        {"linear": 0.28, "angular": 0.4}
    )
    assert backend.third_person_camera(camera) is camera
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
