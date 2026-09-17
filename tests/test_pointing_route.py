"""世界坐标路径推进和 apos 投影的 CPU 单测。"""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.pointing_route import AposRoute, CameraIntrinsics, project_world_point
from lightnav.vln_utils import APOS_STOP_ID


@pytest.fixture
def camera() -> CameraIntrinsics:
    return CameraIntrinsics(width=640, height=360, fx=320.0, fy=320.0, cx=320.0, cy=180.0)


def test_isaac_projection_uses_minus_z_forward(camera):
    projected = project_world_point([0.0, 0.0, -2.0], np.eye(4), camera)
    assert projected is not None
    assert projected.depth == pytest.approx(2.0)
    assert projected.u == pytest.approx(320.0)
    assert projected.v == pytest.approx(180.0)


def test_route_requires_central_consecutive_frames_and_distance(camera):
    route = AposRoute(
        [[0.0, 0.0, -2.0], [0.0, 0.0, -4.0]],
        camera,
        confirm_frames=2,
        max_distance_m=2.1,
    )
    # 第一个点可见且位于中央，但机器人距离还没有达到门限。
    state = route.update(np.eye(4), robot_world=[0.0, 0.0, 1.0])
    assert not state.advanced and state.point_index == 0
    # 连续两帧满足条件后，才推进到第二个点。
    for _ in range(2):
        state = route.update(np.eye(4), robot_world=[0.0, 0.0, -1.0])
    assert state.advanced and state.point_index == 1
    assert state.apos_id is not None


def test_route_resets_streak_when_point_leaves_center(camera):
    route = AposRoute([[0.0, 0.0, -2.0]], camera, confirm_frames=2, max_distance_m=None)
    first = route.update(np.eye(4))
    assert not first.advanced
    transform = np.eye(4)
    transform[0, 3] = 10.0
    outside = route.update(transform)
    assert not outside.in_center
    final = route.update(np.eye(4))
    assert not final.advanced


def test_last_point_emits_stop_sentinel(camera):
    route = AposRoute([[0.0, 0.0, -2.0]], camera, confirm_frames=1, max_distance_m=None)
    state = route.update(np.eye(4))
    assert state.done and state.advanced
    assert state.apos_id == APOS_STOP_ID
    assert state.apos_token == f"<apos_{APOS_STOP_ID}>"


def test_point_outside_image_does_not_emit_clamped_apos(camera):
    route = AposRoute([[10.0, 0.0, -2.0]], camera, max_distance_m=None)
    state = route.update(np.eye(4))
    assert state.visible is False
    assert state.apos_id is None
    assert state.apos_token is None


def test_route_can_advance_by_distance_when_point_is_outside_image(camera):
    route = AposRoute(
        [[10.0, 0.0, -2.0], [0.0, 0.0, -2.0]],
        camera,
        confirm_frames=1,
        max_distance_m=3.0,
        require_visible=False,
    )

    state = route.update(np.eye(4), robot_world=[10.0, 0.0, -2.0])

    assert state.advanced
    assert state.point_index == 1


def test_route_advances_when_current_point_was_passed_toward_next(camera):
    route = AposRoute(
        [[0.0, 0.0, -2.0], [4.0, 0.0, -2.0]],
        camera,
        confirm_frames=1,
        max_distance_m=0.5,
        require_visible=False,
        advance_when_passed=True,
        pass_hysteresis_m=1.0,
    )

    route.update(np.eye(4), robot_world=[-4.0, 0.0, -2.0])
    nearest = route.update(np.eye(4), robot_world=[-1.0, 0.0, -2.0])
    assert not nearest.advanced
    passed = route.update(np.eye(4), robot_world=[2.2, 0.0, -2.0])

    assert passed.advanced
    assert passed.advance_reason == "passed"
    assert passed.point_index == 1
    assert passed.next_distance_m == pytest.approx(1.8)


def test_route_does_not_skip_before_crossing_current_point(camera):
    route = AposRoute(
        [[0.0, 0.0, -2.0], [4.0, 0.0, -2.0]],
        camera,
        max_distance_m=0.5,
        require_visible=False,
        advance_when_passed=True,
        pass_hysteresis_m=0.5,
    )

    route.update(np.eye(4), robot_world=[-1.0, 0.0, -2.0])
    state = route.update(np.eye(4), robot_world=[-2.0, 0.0, -2.0])

    assert not state.advanced
    assert state.point_index == 0


def test_route_never_pass_skips_the_final_point(camera):
    route = AposRoute(
        [[0.0, 0.0, -2.0]],
        camera,
        max_distance_m=0.5,
        require_visible=False,
        advance_when_passed=True,
        pass_hysteresis_m=0.5,
    )

    route.update(np.eye(4), robot_world=[-1.0, 0.0, -2.0])
    state = route.update(np.eye(4), robot_world=[2.0, 0.0, -2.0])

    assert not state.advanced
    assert not state.done
