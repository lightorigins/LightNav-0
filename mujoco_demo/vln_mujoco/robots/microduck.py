from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from ..model import SPAWN, load_scene_xml
from .base import RenderCamera, RobotState, Twist2D

MODEL_PREFIX = "microduck_"
CAMERA_NAME = f"{MODEL_PREFIX}rgb"
TRUNK_BODY_NAME = f"{MODEL_PREFIX}trunk_base"
BASE_JOINT_NAME = f"{MODEL_PREFIX}trunk_base_freejoint"
IMU_SENSOR_NAME = f"{MODEL_PREFIX}imu_ang_vel"

CONTROL_DECIMATION = 4
MAX_LINEAR_COMMAND = 0.30
MIN_LINEAR_COMMAND = 0.28
LINEAR_DEADBAND = 0.08
MAX_ANGULAR_COMMAND = 1.50
ANGULAR_DEADBAND = 0.03
TORQUE_LIMIT_NM = 0.6405236195572268

EXPECTED_OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "head_command",
    "body_command",
)
EXPECTED_COMMAND_NAMES = ("twist", "head_pose", "body_pose")


class TensorInfo(Protocol):
    name: str
    shape: list[int | str | None]
    type: str


class ModelMetadata(Protocol):
    custom_metadata_map: dict[str, str]


class PolicySession(Protocol):
    def get_inputs(self) -> list[TensorInfo]: ...

    def get_outputs(self) -> list[TensorInfo]: ...

    def get_modelmeta(self) -> ModelMetadata: ...

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


def _load_policy_session(policy_path: Path) -> PolicySession:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "MicroDuck requires the optional dependency: "
            "uv sync --extra microduck"
        ) from exc
    return ort.InferenceSession(
        str(policy_path.resolve()),
        providers=["CPUExecutionProvider"],
    )


def _metadata_names(metadata: dict[str, str], key: str) -> tuple[str, ...]:
    value = metadata.get(key, "")
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError(f"MicroDuck walking policy is missing {key!r} metadata")
    return names


def _metadata_floats(metadata: dict[str, str], key: str) -> np.ndarray:
    value = metadata.get(key, "")
    try:
        values = np.asarray(
            [float(part.strip()) for part in value.split(",") if part.strip()],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise ValueError(f"MicroDuck walking policy has invalid {key!r} metadata") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"MicroDuck walking policy has invalid {key!r} metadata")
    return values


def shape_microduck_command(linear: float, angular: float) -> Twist2D:
    if not math.isfinite(linear) or not math.isfinite(angular):
        return (0.0, 0.0)
    linear = float(np.clip(linear, -MAX_LINEAR_COMMAND, MAX_LINEAR_COMMAND))
    angular = float(np.clip(angular, -MAX_ANGULAR_COMMAND, MAX_ANGULAR_COMMAND))
    if abs(linear) < LINEAR_DEADBAND:
        linear = 0.0
    elif abs(linear) < MIN_LINEAR_COMMAND:
        linear = math.copysign(MIN_LINEAR_COMMAND, linear)
    if abs(angular) < ANGULAR_DEADBAND:
        angular = 0.0
    return linear, angular


def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    scalar = quaternion[0]
    xyz = quaternion[1:4]
    cross = np.cross(xyz, vector) * 2.0
    return vector - scalar * cross + np.cross(xyz, cross)


def build_model(
    robot_source: Path,
    scene_source: Path | None = None,
) -> mujoco.MjModel:
    robot_source = robot_source.resolve()
    if not robot_source.is_file():
        raise FileNotFoundError(f"MicroDuck MJCF is missing: {robot_source}")

    scene_xml = ET.tostring(load_scene_xml(scene_source), encoding="unicode")
    scene = mujoco.MjSpec.from_string(scene_xml)
    robot = mujoco.MjSpec.from_file(str(robot_source))
    trunk = robot.body("trunk_base")
    camera = robot.camera("head_camera")
    if trunk is None or camera is None:
        raise ValueError("MicroDuck MJCF requires trunk_base and head_camera")

    trunk.pos = [SPAWN[0], SPAWN[1], 0.125]
    camera.name = "rgb"
    camera.fovy = 79.865
    camera.quat = [0.70710678, 0.0, 0.0, -0.70710678]

    anchor = scene.worldbody.add_frame()
    scene.attach(robot, prefix=MODEL_PREFIX, frame=anchor)
    return scene.compile()


class MicroDuckPolicy:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: Path,
        session: PolicySession | None = None,
    ) -> None:
        policy_path = policy_path.resolve()
        if session is None and not policy_path.is_file():
            raise FileNotFoundError(f"MicroDuck walking policy is missing: {policy_path}")
        self.model = model
        self.data = data
        self.session = session or _load_policy_session(policy_path)

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 61]:
            raise ValueError("MicroDuck walking policy must accept [1, 61] observations")
        if inputs[0].type != "tensor(float)":
            raise ValueError("MicroDuck walking policy input must be float32")
        if len(outputs) != 1 or outputs[0].shape != [1, 14]:
            raise ValueError("MicroDuck walking policy must return [1, 14] actions")
        if outputs[0].type != "tensor(float)":
            raise ValueError("MicroDuck walking policy output must be float32")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        metadata = self.session.get_modelmeta().custom_metadata_map
        joint_names = _metadata_names(metadata, "joint_names")
        observation_names = _metadata_names(metadata, "observation_names")
        command_names = _metadata_names(metadata, "command_names")
        if observation_names != EXPECTED_OBSERVATION_NAMES:
            raise ValueError("MicroDuck walking policy observation layout is unsupported")
        if command_names != EXPECTED_COMMAND_NAMES:
            raise ValueError("MicroDuck walking policy command layout is unsupported")

        self.default_pose = _metadata_floats(metadata, "default_joint_pos")
        if self.default_pose.shape != (14,):
            raise ValueError("MicroDuck walking policy must define 14 default joints")
        try:
            self.action_scale = float(metadata["action_scale"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "MicroDuck walking policy has invalid 'action_scale' metadata"
            ) from exc
        if not math.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError(
                "MicroDuck walking policy has invalid 'action_scale' metadata"
            )

        actuator_joint_names: list[str] = []
        joint_qpos: list[int] = []
        joint_qvel: list[int] = []
        for index in range(model.nu):
            joint_id = int(model.actuator_trnid[index, 0])
            if joint_id < 0:
                raise ValueError("MicroDuck actuators must target joints")
            model_name = model.joint(joint_id).name
            if not model_name.startswith(MODEL_PREFIX):
                raise ValueError("MicroDuck actuator joint is missing its model prefix")
            actuator_joint_names.append(model_name.removeprefix(MODEL_PREFIX))
            joint_qpos.append(int(model.jnt_qposadr[joint_id]))
            joint_qvel.append(int(model.jnt_dofadr[joint_id]))
        if tuple(actuator_joint_names) != joint_names:
            raise ValueError(
                "MicroDuck policy joint_names do not match the MJCF actuator order"
            )
        self.joint_qpos = np.asarray(joint_qpos)
        self.joint_qvel = np.asarray(joint_qvel)

        self.trunk_body = model.body(TRUNK_BODY_NAME).id
        self.imu_sensor = model.sensor(IMU_SENSOR_NAME).id
        if int(model.sensor_dim[self.imu_sensor]) != 3:
            raise ValueError("MicroDuck IMU angular-velocity sensor must have 3 values")
        self.last_action = np.zeros(14, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.command.fill(0.0)

    def set_velocity(self, linear: float, angular: float) -> Twist2D:
        linear, angular = shape_microduck_command(linear, angular)
        self.command.fill(0.0)
        self.command[0] = linear
        self.command[2] = angular
        return linear, angular

    def observation(self) -> np.ndarray:
        sensor_address = int(self.model.sensor_adr[self.imu_sensor])
        angular_velocity = self.data.sensordata[
            sensor_address : sensor_address + 3
        ].astype(np.float32)
        quaternion = self.data.xquat[self.trunk_body].astype(np.float32)
        gravity = _quat_rotate_inverse(
            quaternion,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        joint_position = self.data.qpos[self.joint_qpos].astype(np.float32)
        joint_velocity = self.data.qvel[self.joint_qvel].astype(np.float32)
        observation = np.concatenate(
            (
                angular_velocity,
                gravity,
                joint_position - self.default_pose,
                joint_velocity,
                self.last_action,
                self.command,
            )
        ).astype(np.float32)
        if observation.shape != (61,) or not np.isfinite(observation).all():
            raise RuntimeError("MicroDuck policy observation is invalid")
        return observation

    def step(self) -> None:
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: self.observation().reshape(1, 61)},
        )
        action = np.asarray(outputs[0], dtype=np.float32)
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise RuntimeError("MicroDuck walking policy returned invalid actions")
        self.last_action[:] = action[0]
        self.data.ctrl[:] = self.default_pose + self.last_action * self.action_scale


class MicroDuckBackend:
    name = "MicroDuck"
    camera_name = CAMERA_NAME

    def __init__(
        self,
        robot_model: Path,
        walking_policy: Path,
        policy_session: PolicySession | None = None,
        scene_source: Path | None = None,
    ) -> None:
        self.model = build_model(robot_model, scene_source)
        if self.model.nu != 14:
            raise ValueError("MicroDuck MJCF must define 14 actuators")
        self.model.actuator_forcerange[:, 0] = -TORQUE_LIMIT_NM
        self.model.actuator_forcerange[:, 1] = TORQUE_LIMIT_NM
        self.model.actuator_forcelimited[:] = 1
        self.data = mujoco.MjData(self.model)
        self.policy = MicroDuckPolicy(
            self.model,
            self.data,
            walking_policy,
            session=policy_session,
        )
        self._base_joint = self.model.joint(BASE_JOINT_NAME).id
        self._base_qpos = int(self.model.jnt_qposadr[self._base_joint])
        self._base_dof = int(self.model.jnt_dofadr[self._base_joint])
        self._policy_command = (0.0, 0.0)
        self._physics_steps = 0
        self._initialize_pose()

    def _initialize_pose(self) -> None:
        self.data.qpos[self._base_qpos : self._base_qpos + 3] = [
            SPAWN[0],
            SPAWN[1],
            0.125,
        ]
        self.data.qpos[self._base_qpos + 3 : self._base_qpos + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qpos[self.policy.joint_qpos] = self.policy.default_pose
        self.data.ctrl[:] = self.policy.default_pose
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.policy.reset()
        self._policy_command = (0.0, 0.0)
        self._physics_steps = 0
        self._initialize_pose()

    def step(self, command: Twist2D) -> None:
        if self._physics_steps % CONTROL_DECIMATION == 0:
            self._policy_command = self.policy.set_velocity(*command)
            self.policy.step()
        mujoco.mj_step(self.model, self.data)
        self._physics_steps += 1

    @staticmethod
    def _yaw(quaternion: np.ndarray) -> float:
        qw, qx, qy, qz = (float(value) for value in quaternion)
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def state(self) -> RobotState:
        qpos = self.data.qpos
        x = float(qpos[self._base_qpos])
        y = float(qpos[self._base_qpos + 1])
        z = float(qpos[self._base_qpos + 2])
        quaternion = qpos[self._base_qpos + 3 : self._base_qpos + 7].astype(
            np.float32
        )
        yaw = self._yaw(quaternion)
        velocity_world = self.data.qvel[
            self._base_dof : self._base_dof + 3
        ].astype(np.float32)
        velocity_body = _quat_rotate_inverse(quaternion, velocity_world)
        angular = float(self.data.qvel[self._base_dof + 5])
        return RobotState(
            pose=(x, y, z, yaw),
            velocity=(float(velocity_body[0]), angular),
            telemetry={
                "policy_command": {
                    "linear": self._policy_command[0],
                    "angular": self._policy_command[1],
                }
            },
        )

    def third_person_camera(self, camera: mujoco.MjvCamera) -> RenderCamera:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.55
        camera.azimuth = 135.0
        camera.elevation = -20.0
        camera.lookat[:] = self.data.xpos[self.policy.trunk_body]
        return camera
