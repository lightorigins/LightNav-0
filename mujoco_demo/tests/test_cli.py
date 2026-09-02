import pytest
from vln_mujoco.__main__ import parse_args, simulation_from_args


def test_turtlebot_is_the_default_robot() -> None:
    args = parse_args([])
    assert args.robot == "turtlebot"
    assert args.robot_model is None
    assert args.walking_policy is None


def test_microduck_requires_model_and_policy_paths() -> None:
    args = parse_args(["--robot", "microduck"])
    with pytest.raises(SystemExit, match="requires --robot-model"):
        simulation_from_args(args)


def test_turtlebot_rejects_microduck_only_paths() -> None:
    args = parse_args(["--robot-model", "robot.xml"])
    with pytest.raises(SystemExit, match="require --robot microduck"):
        simulation_from_args(args)
