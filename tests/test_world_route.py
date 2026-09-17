"""世界坐标路线跟随器的 CPU 单测。"""

from __future__ import annotations

import json
import math

import pytest

from lightnav.world_route import (
    POINTING_MODES,
    SavedPointingRoute,
    WorldRouteFollower,
    initial_route_yaw_degrees,
    load_saved_pointing_route,
    load_world_route,
    prompt_pointing_for_mode,
    route_targets_for_pointing_mode,
)


def test_picker_route_executes_apos_before_opos(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(json.dumps({"opos": [10, 0, 2], "apos": [[2, 0, 0], [5, 0, 0]]}), encoding="utf-8")
    assert load_world_route(path) == ((5.0, 0.0, 0.0), (2.0, 0.0, 0.0), (10.0, 0.0, 2.0))


def test_picker_new_points_route_reverses_click_order(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps({"points": [[10, 0, 0], [8, 0, 0], [4, 0, 0], [0, 0, 0]]}),
        encoding="utf-8",
    )
    assert load_world_route(path) == (
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    )


def test_follower_switches_only_after_reaching_point():
    follower = WorldRouteFollower([(2.0, 0.0, 0.0), (4.0, 0.0, 0.0)], reach_distance_m=0.5)
    first = follower.update(0.0, 0.0, 0.0)
    assert first.point_index == 0 and not first.advanced
    assert first.command[0] > 0.0
    second = follower.update(1.8, 0.0, 0.0)
    assert second.point_index == 1 and second.advanced
    assert second.target_world == (4.0, 0.0, 0.0)


def test_follower_stops_at_final_goal():
    follower = WorldRouteFollower([(1.0, 0.0, 0.0)], reach_distance_m=0.5)
    result = follower.update(0.8, 0.0, 0.0)
    assert result.done and result.command == (0.0, 0.0, 0.0)


def test_follower_turns_in_place_for_behind_target():
    follower = WorldRouteFollower([(0.0, -2.0, 0.0)], reach_distance_m=0.2)
    result = follower.update(0.0, 0.0, 0.0)
    assert result.command[0] == pytest.approx(0.0)
    assert result.command[2] < 0.0


def test_saved_pointing_route_skips_last_apos_start(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps({"opos": [10, 0, 0], "apos": [[8, 0, 0], [4, 0, 0], [0, 0, 0]]}),
        encoding="utf-8",
    )
    route = load_saved_pointing_route(path)
    assert route.start_world == (0.0, 0.0, 0.0)
    assert route.apos_targets == ((4.0, 0.0, 0.0), (8.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert route.opos_world == (10.0, 0.0, 0.0)


def test_saved_pointing_route_accepts_explicit_start_without_apos(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps({"start": [0, 0, 0], "opos": [10, 0, 0], "apos": []}),
        encoding="utf-8",
    )
    route = load_saved_pointing_route(path)
    assert route.start_world == (0.0, 0.0, 0.0)
    assert route.apos_targets == ((10.0, 0.0, 0.0),)
    assert route.opos_world == (10.0, 0.0, 0.0)


def test_saved_pointing_route_keeps_apos_when_start_is_explicit(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps({"start": [0, 0, 0], "opos": [10, 0, 0], "apos": [[4, 0, 0], [8, 0, 0]]}),
        encoding="utf-8",
    )
    route = load_saved_pointing_route(path)
    assert route.start_world == (0.0, 0.0, 0.0)
    assert route.apos_targets == ((8.0, 0.0, 0.0), (4.0, 0.0, 0.0), (10.0, 0.0, 0.0))


def test_saved_pointing_route_reads_ordered_points_format(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "points": [
                    [10, 0, 0],  # OPOS, recorded first
                    [8, 0, 0],
                    [4, 0, 0],
                    [0, 0, 0],  # start, recorded last
                ]
            }
        ),
        encoding="utf-8",
    )
    route = load_saved_pointing_route(path)
    assert route.start_world == (0.0, 0.0, 0.0)
    assert route.apos_targets == ((4.0, 0.0, 0.0), (8.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert route.opos_world == (10.0, 0.0, 0.0)


def test_saved_pointing_route_reads_early_points_records(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "points": [
                    {"role": "opos", "world": [10, 0, 0]},
                    {"role": "apos", "world": [4, 0, 0]},
                    {"role": "apos", "world": [0, 0, 0]},
                ]
            }
        ),
        encoding="utf-8",
    )
    route = load_saved_pointing_route(path)
    assert route.start_world == (0.0, 0.0, 0.0)
    assert route.apos_targets == ((4.0, 0.0, 0.0), (10.0, 0.0, 0.0))


def test_multi_opos_uses_every_point_except_start_as_sequential_opos():
    route = SavedPointingRoute(
        start_world=(0.0, 0.0, 0.0),
        apos_targets=((4.0, 0.0, 0.0), (8.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        opos_world=(10.0, 0.0, 0.0),
    )

    assert POINTING_MODES == ("pair", "opos_only", "multi_opos")
    assert route_targets_for_pointing_mode(route, "multi_opos") == (
        (4.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    )
    assert prompt_pointing_for_mode("multi_opos", apos_id=123, opos_id=456) == {
        "opos_id": 456
    }


def test_existing_pointing_modes_keep_their_target_and_payload_behavior():
    route = SavedPointingRoute(
        start_world=(0.0, 0.0, 0.0),
        apos_targets=((4.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        opos_world=(10.0, 0.0, 0.0),
    )

    assert route_targets_for_pointing_mode(route, "pair") == route.apos_targets
    assert route_targets_for_pointing_mode(route, "opos_only") == (route.opos_world,)
    assert prompt_pointing_for_mode("pair", apos_id=123, opos_id=456) == {
        "apos_id": 123,
        "opos_id": 456,
    }
    assert prompt_pointing_for_mode("opos_only", apos_id=123, opos_id=456) == {
        "opos_id": 456
    }


def test_initial_yaw_faces_first_target_for_each_pointing_mode():
    route = SavedPointingRoute(
        start_world=(0.0, 0.0, 0.0),
        apos_targets=((0.0, 2.0, 0.0), (4.0, 2.0, 0.0)),
        opos_world=(4.0, 2.0, 0.0),
    )

    assert initial_route_yaw_degrees(route, "pair") == pytest.approx(90.0)
    assert initial_route_yaw_degrees(route, "multi_opos") == pytest.approx(90.0)
    assert initial_route_yaw_degrees(route, "opos_only") == pytest.approx(
        math.degrees(math.atan2(2.0, 4.0))
    )


def test_initial_yaw_is_absent_without_distinct_start_and_target():
    no_start = SavedPointingRoute(None, ((1.0, 0.0, 0.0),), (1.0, 0.0, 0.0))
    same_point = SavedPointingRoute(
        (1.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0),),
        (1.0, 0.0, 0.0),
    )

    assert initial_route_yaw_degrees(no_start, "multi_opos") is None
    assert initial_route_yaw_degrees(same_point, "multi_opos") is None


@pytest.mark.parametrize("helper", [route_targets_for_pointing_mode, prompt_pointing_for_mode])
def test_pointing_mode_helpers_reject_unknown_mode(helper):
    route = SavedPointingRoute(None, ((1.0, 0.0, 0.0),), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="pointing mode"):
        if helper is route_targets_for_pointing_mode:
            helper(route, "unknown")
        else:
            helper("unknown", apos_id=1, opos_id=2)
