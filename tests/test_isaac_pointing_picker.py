"""Isaac 依赖之外的场景发现逻辑单测。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.isaac_pointing_picker import OccupancyMap, _build_parser, discover_scenes, resolve_scene


def test_parser_supports_start_and_opos_mode():
    args = _build_parser().parse_args(["--start-and-opos"])
    assert args.start_and_opos is True


def test_discover_scenes_prefers_directory_named_usd(tmp_path: Path):
    three = tmp_path / "three"
    three.mkdir()
    (three / "other.usda").touch()
    (three / "three.usd").touch()
    park = tmp_path / "park"
    park.mkdir()
    (park / "park.usda").touch()

    assert discover_scenes(tmp_path) == [
        ("park", (park / "park.usda").resolve()),
        ("three", (three / "three.usd").resolve()),
    ]


def test_discover_scenes_ignores_hidden_directories(tmp_path: Path):
    hidden = tmp_path / ".generated"
    hidden.mkdir()
    (hidden / "route.usda").touch()
    assert discover_scenes(tmp_path) == []


def test_resolve_scene_accepts_name_and_file_path(tmp_path: Path):
    scene_dir = tmp_path / "company"
    scene_dir.mkdir()
    scene_file = scene_dir / "company.usd"
    scene_file.touch()

    assert resolve_scene("company", tmp_path) == ("company", scene_file.resolve())
    assert resolve_scene(str(scene_file), tmp_path) == ("company", scene_file.resolve())


def test_resolve_scene_reports_available_names(tmp_path: Path):
    scene_dir = tmp_path / "three"
    scene_dir.mkdir()
    (scene_dir / "three.usd").touch()

    with pytest.raises(ValueError, match="three"):
        resolve_scene("missing", tmp_path)


def test_occupancy_map_converts_canvas_and_marks_obstacle(tmp_path: Path):
    occupancy = OccupancyMap((0.0, 4.0, 0.0, 4.0), ground_z=0.0, resolution=1.0)
    occupancy.mark_occupied(1.0, 2.0, 1.0, 2.0)

    world_x, world_y, column, row = occupancy.world_from_canvas(200.0, 320.0, canvas_size=640)
    assert (column, row) == (1, 2)
    assert (world_x, world_y) == (1.5, 1.5)
    assert occupancy.occupied[row][column]

    svg_path = tmp_path / "map.svg"
    occupancy.to_svg(svg_path)
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
