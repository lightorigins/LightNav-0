#!/usr/bin/env python3
"""在 Isaac Sim 中生成二维占据图并点击记录 OPOS/APOS 世界坐标。

用法示例（使用 Isaac Sim 自带 Python）：

    /home/ubuntu/miniconda3/envs/isaacsim6/bin/python \
        tools/isaac_pointing_picker.py --scene three

程序会根据场景几何体包围盒生成俯视占据图。点击点按顺序写入 JSON 的 ``points``：
第一个点是最终目标，最后一个点是起点，中间点是路径点。点击位置会同步绘制到
3D 场景，保存为 JSON，不会覆盖原始 USD 文件。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


DEFAULT_SCENE_ROOT = Path("/home/ubuntu/scene")
MAP_IMAGE_SIZE = 640


def _scene_file_candidates(scene_dir: Path) -> list[Path]:
    """返回一个场景目录中常见的 USD 文件，优先使用与目录同名的文件。"""
    preferred = [scene_dir / f"{scene_dir.name}{suffix}" for suffix in (".usd", ".usda", ".usdc")]
    found = [path for path in preferred if path.is_file()]
    if found:
        return found
    return sorted(
        path
        for path in scene_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )


def discover_scenes(scene_root: Path) -> list[tuple[str, Path]]:
    """扫描场景根目录，返回 ``(名称, USD 路径)``。"""
    if not scene_root.is_dir():
        raise FileNotFoundError(f"场景根目录不存在: {scene_root}")
    scenes: list[tuple[str, Path]] = []
    for child in sorted(scene_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        candidates = _scene_file_candidates(child)
        if candidates:
            scenes.append((child.name, candidates[0].resolve()))
    return scenes


def resolve_scene(scene_value: str | None, scene_root: Path) -> tuple[str, Path]:
    """把场景名或 USD 路径解析成可打开的文件。"""
    scenes = discover_scenes(scene_root)
    if scene_value:
        supplied = Path(scene_value).expanduser()
        if supplied.is_file():
            return supplied.stem, supplied.resolve()
        for name, path in scenes:
            if name == scene_value:
                return name, path
        raise ValueError(
            f"找不到场景 {scene_value!r}。可用场景: "
            + (", ".join(name for name, _ in scenes) or "无")
        )
    if not scenes:
        raise ValueError(f"{scene_root} 下没有找到 USD 场景")
    print("请选择要打开的场景:")
    for index, (name, path) in enumerate(scenes, start=1):
        print(f"  {index}. {name}  ({path})")
    while True:
        answer = input(f"输入编号 [1-{len(scenes)}]: ").strip()
        try:
            index = int(answer)
        except ValueError:
            index = 0
        if 1 <= index <= len(scenes):
            return scenes[index - 1]
        print("编号无效，请重新输入。", flush=True)


class OccupancyMap:
    """根据场景几何体包围盒生成俯视二维占据图。"""

    def __init__(self, bounds: tuple[float, float, float, float], ground_z: float, resolution: float) -> None:
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.ground_z = ground_z
        # 地图过大时自动降低分辨率，避免 UI 生成过多 SVG 小方格。
        extent = max(self.max_x - self.min_x, self.max_y - self.min_y)
        self.resolution = max(float(resolution), extent / 600.0 if extent > 0 else float(resolution))
        self.width = max(1, int(math.ceil((self.max_x - self.min_x) / self.resolution)))
        self.height = max(1, int(math.ceil((self.max_y - self.min_y) / self.resolution)))
        self.occupied = [[False] * self.width for _ in range(self.height)]

    @classmethod
    def from_stage(cls, stage, resolution: float = 0.2) -> "OccupancyMap":
        """读取碰撞 Mesh 的三角面，并把高于地面的面栅格化。"""
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        scene_boxes: list[tuple[float, float, float, float, float, float, str]] = []
        obstacle_boxes: list[tuple[float, float, float, float, float, float, str]] = []
        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if not prim.IsA(UsdGeom.Boundable) or type_name in {"Camera", "DistantLight", "SphereLight", "RectLight"}:
                continue
            # Gaussian splat 点云只有一个覆盖全场景的包围盒，不能直接当作障碍物。
            if type_name.startswith("ParticleField3D"):
                continue
            try:
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points = mesh.GetPointsAttr().Get() or []
                    counts = mesh.GetFaceVertexCountsAttr().Get() or []
                    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
                    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    world_points = [transform.Transform(point) for point in points]
                    if world_points:
                        scene_boxes.append((
                            min(float(point[0]) for point in world_points),
                            max(float(point[0]) for point in world_points),
                            min(float(point[1]) for point in world_points),
                            max(float(point[1]) for point in world_points),
                            min(float(point[2]) for point in world_points),
                            max(float(point[2]) for point in world_points),
                            str(prim.GetPath()),
                        ))
                    cursor = 0
                    face_stride = max(1, len(counts) // 12000)
                    for face_number, count in enumerate(counts):
                        face_indices = indices[cursor : cursor + int(count)]
                        cursor += int(count)
                        if face_number % face_stride != 0:
                            continue
                        face_points = [world_points[int(index)] for index in face_indices if 0 <= int(index) < len(world_points)]
                        if len(face_points) < 3:
                            continue
                        z_values = [float(point[2]) for point in face_points]
                        # 水平地面三角面不占据栅格；墙体和立体物体的侧面会被标记。
                        if max(z_values) - min(z_values) >= 0.15:
                            obstacle_boxes.append((
                                min(float(point[0]) for point in face_points),
                                max(float(point[0]) for point in face_points),
                                min(float(point[1]) for point in face_points),
                                max(float(point[1]) for point in face_points),
                                min(z_values),
                                max(z_values),
                                str(prim.GetPath()),
                            ))
                    continue
                value = cache.ComputeWorldBound(prim).GetRange()
                minimum, maximum = value.GetMin(), value.GetMax()
                if value.IsEmpty():
                    continue
                box = (float(minimum[0]), float(maximum[0]), float(minimum[1]), float(maximum[1]), float(minimum[2]), float(maximum[2]), str(prim.GetPath()))
                scene_boxes.append(box)
                obstacle_boxes.append(box)
            except Exception:
                # 个别带坏引用的 Prim 不应阻止整张地图生成。
                continue
        if not scene_boxes:
            raise RuntimeError("场景中没有可用于生成占据图的几何体")

        min_x = min(item[0] for item in scene_boxes)
        max_x = max(item[1] for item in scene_boxes)
        min_y = min(item[2] for item in scene_boxes)
        max_y = max(item[3] for item in scene_boxes)
        ground_z = min(item[4] for item in scene_boxes)
        margin = max(1.0, resolution * 3.0)
        result = cls((min_x - margin, max_x + margin, min_y - margin, max_y + margin), ground_z, resolution)

        ignored_words = ("ground", "floor", "terrain", "navmesh", "lightnavpointpicker")
        for x0, x1, y0, y1, z0, z1, path in obstacle_boxes:
            name = path.lower()
            height = z1 - z0
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            # 薄而巨大的地板/平面不是障碍物；其余高于地面的几何体标记为占据。
            if any(word in name for word in ignored_words) or height < 0.15 or (height < 0.35 and area > 25.0):
                continue
            result.mark_occupied(x0, x1, y0, y1)
        return result

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """把世界 XY 坐标转换为地图列、行。"""
        column = int(math.floor((x - self.min_x) / self.resolution))
        row = int(math.floor((self.max_y - y) / self.resolution))
        if 0 <= column < self.width and 0 <= row < self.height:
            return column, row
        return None

    def cell_to_world(self, column: int, row: int) -> tuple[float, float]:
        """把地图列、行转换为对应栅格中心的世界 XY 坐标。"""
        return (
            self.min_x + (column + 0.5) * self.resolution,
            self.max_y - (row + 0.5) * self.resolution,
        )

    def mark_occupied(self, x0: float, x1: float, y0: float, y1: float) -> None:
        """把一个世界包围盒覆盖的栅格标成占据。"""
        c0 = int(math.floor((min(x0, x1) - self.min_x) / self.resolution))
        c1 = int(math.floor((max(x0, x1) - self.min_x) / self.resolution))
        r0 = int(math.floor((self.max_y - max(y0, y1)) / self.resolution))
        r1 = int(math.floor((self.max_y - min(y0, y1)) / self.resolution))
        c0, c1 = max(0, c0), min(self.width - 1, c1)
        r0, r1 = max(0, r0), min(self.height - 1, r1)
        if c0 > c1 or r0 > r1:
            return
        for row in range(r0, r1 + 1):
            for column in range(c0, c1 + 1):
                self.occupied[row][column] = True

    def world_from_canvas(
        self,
        x: float,
        y: float,
        canvas_size: float = MAP_IMAGE_SIZE,
        canvas_height: float | None = None,
    ) -> tuple[float, float, int, int] | None:
        """把地图控件中的像素位置转换为世界坐标和栅格索引。"""
        canvas_width = float(canvas_size)
        canvas_height = canvas_width if canvas_height is None else float(canvas_height)
        if not (0 <= x <= canvas_width and 0 <= y <= canvas_height):
            return None
        column = min(self.width - 1, max(0, int(x / canvas_width * self.width)))
        row = min(self.height - 1, max(0, int(y / canvas_height * self.height)))
        world_x, world_y = self.cell_to_world(column, row)
        return world_x, world_y, column, row

    def to_svg(self, path: Path, points: list[dict[str, object]] | None = None, size: int = MAP_IMAGE_SIZE) -> None:
        """保存一张可在 omni.ui.Image 中显示和点击的 SVG 占据图。"""
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
            f'<rect width="{size}" height="{size}" fill="#f4f7f5"/>',
        ]
        cell_width = size / self.width
        cell_height = size / self.height
        for row, values in enumerate(self.occupied):
            for column, occupied in enumerate(values):
                if occupied:
                    parts.append(
                        f'<rect x="{column * cell_width:.2f}" y="{row * cell_height:.2f}" '
                        f'width="{cell_width + 0.3:.2f}" height="{cell_height + 0.3:.2f}" fill="#263238"/>'
                    )
        parts.append(f'<rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="#78909c" stroke-width="2"/>')
        for index, point in enumerate(points or []):
            world = point["world"]
            cell = self.world_to_cell(float(world[0]), float(world[1]))
            if cell is None:
                continue
            column, row = cell
            colour = {
                "start": "#1976d2",
                "opos": "#e91e63",
                "apos": "#00a878",
            }.get(str(point["role"]), "#00a878")
            parts.append(
                f'<circle cx="{(column + 0.5) * cell_width:.2f}" cy="{(row + 0.5) * cell_height:.2f}" '
                f'r="{10 if index == 0 else 7}" fill="{colour}" stroke="white" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{(column + 0.5) * cell_width + 9:.2f}" y="{(row + 0.5) * cell_height - 9:.2f}" '
                'font-family="sans-serif" font-size="14" fill="#111">'
                f'{index + 1}</text>'
            )
        parts.append("</svg>\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(parts), encoding="utf-8")

    def to_png(self, path: Path, points: list[dict[str, object]] | None = None, size: int = MAP_IMAGE_SIZE) -> None:
        """保存 PNG 占据图，避免某些 Isaac 版本无法加载 SVG。"""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (size, size), (244, 247, 245))
        draw = ImageDraw.Draw(image)
        cell_width = size / self.width
        cell_height = size / self.height
        for row, values in enumerate(self.occupied):
            for column, occupied in enumerate(values):
                if occupied:
                    left, top = column * cell_width, row * cell_height
                    right, bottom = (column + 1) * cell_width, (row + 1) * cell_height
                    draw.rectangle((left, top, right, bottom), fill=(38, 50, 56))
        draw.rectangle((0, 0, size - 1, size - 1), outline=(120, 144, 156), width=2)
        for index, point in enumerate(points or []):
            world = point["world"]
            cell = self.world_to_cell(float(world[0]), float(world[1]))
            if cell is None:
                continue
            column, row = cell
            cx = (column + 0.5) * cell_width
            cy = (row + 0.5) * cell_height
            radius = 10 if index == 0 else 7
            colour = {
                "start": (25, 118, 210),
                "opos": (233, 30, 99),
                "apos": (0, 168, 120),
            }.get(str(point["role"]), (0, 168, 120))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline=(255, 255, 255), width=2)
            draw.text((cx + radius + 2, cy - radius - 2), str(index + 1), fill=(17, 17, 17))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 Isaac Sim 中点击记录有序 pointing 路线坐标")
    parser.add_argument(
        "--scene",
        help="场景目录名（例如 three、park）或 USD 文件路径；省略时启动前交互选择",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=DEFAULT_SCENE_ROOT,
        help=f"场景根目录（默认: {DEFAULT_SCENE_ROOT}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="路线 JSON 输出路径；默认写入 scene/pointing_routes/<场景名>_route.json",
    )
    parser.add_argument("--list-scenes", action="store_true", help="只列出可用场景，不启动 Isaac Sim")
    parser.add_argument("--confirm-frames", type=int, default=3, help="保留给下游路由器的确认帧建议值")
    parser.add_argument("--map-resolution", type=float, default=0.2, help="占据图栅格分辨率（米，默认 0.2）")
    parser.add_argument("--allow-occupied", action="store_true", help="允许在占据栅格上选点（默认只允许空闲区域）")
    parser.add_argument(
        "--start-and-opos",
        action="store_true",
        help="只记录两个点：第一次点击为 OPOS，第二次点击为起点，不记录 APOS",
    )
    return parser


class PointPicker:
    """管理二维占据图点击、3D 标记绘制和 JSON 保存。"""

    def __init__(
        self,
        stage,
        occupancy_map: OccupancyMap,
        output_path: Path,
        scene_name: str,
        confirm_frames: int,
        allow_occupied: bool,
        start_and_opos: bool = False,
    ) -> None:
        import omni.ui as ui
        from pxr import Gf, Sdf, Usd, UsdGeom

        self.stage = stage
        self.occupancy_map = occupancy_map
        self.output_path = output_path
        self.scene_name = scene_name
        self.confirm_frames = int(confirm_frames)
        self.allow_occupied = allow_occupied
        self.start_and_opos = bool(start_and_opos)
        self.points: list[dict[str, object]] = []
        self.finished = False
        self.closed = False
        self._ui = ui
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._Sdf = Sdf
        self._Gf = Gf

        # 标记放在 SessionLayer 中，关闭工具时不会污染原始 USD。
        self._marker_root = "/World/LightNavPointPicker"
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            UsdGeom.Xform.Define(stage, Sdf.Path(self._marker_root))

        self.map_image_path = output_path.with_suffix(".png")
        self.occupancy_map.to_png(self.map_image_path)
        self.window = ui.Window("LightNav Occupancy Map Picker", width=700, height=790, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=7, style={"margin": 12}):
                ui.Label("Occupancy map: dark = obstacle, light = free space")
                ui.Label(
                    "First click = OPOS; second click = start"
                    if self.start_and_opos
                    else "Last click = start; other clicks = ordered route targets"
                )
                self.map_image = ui.Image(str(self.map_image_path), width=MAP_IMAGE_SIZE, height=MAP_IMAGE_SIZE)
                self.map_image.set_mouse_pressed_fn(self.on_map_click)
                self.status = ui.Label("Waiting for a click on the map")
                self.points_label = ui.Label("Recorded points: 0")
                with ui.HStack(spacing=6):
                    ui.Button("Undo last", clicked_fn=self.undo)
                    ui.Button("Clear", clicked_fn=self.clear)
                with ui.HStack(spacing=6):
                    ui.Button("Save", clicked_fn=self.save)
                    ui.Button("Save and exit", clicked_fn=self.finish)

    def _set_status(self, text: str) -> None:
        self.status.text = text
        print(text, flush=True)

    def _refresh_map(self) -> None:
        """重绘二维地图，使新点在地图上立即可见。"""
        self.occupancy_map.to_png(self.map_image_path, self.points)
        # 先清空 source 再重新设置，兼容旧版 omni.ui 的资源缓存行为。
        self.map_image.source = ""
        self.map_image.source = str(self.map_image_path)

    def _map_local_position(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """兼容不同 Kit 版本返回的本地坐标/屏幕坐标两种回调格式。"""
        width = float(getattr(self.map_image, "computed_width", MAP_IMAGE_SIZE) or MAP_IMAGE_SIZE)
        height = float(getattr(self.map_image, "computed_height", MAP_IMAGE_SIZE) or MAP_IMAGE_SIZE)
        screen_x = float(getattr(self.map_image, "screen_position_x", 0.0) or 0.0)
        screen_y = float(getattr(self.map_image, "screen_position_y", 0.0) or 0.0)
        candidates = [(float(x), float(y)), (float(x) - screen_x, float(y) - screen_y)]
        for local_x, local_y in candidates:
            if -2.0 <= local_x <= width + 2.0 and -2.0 <= local_y <= height + 2.0:
                return max(0.0, min(width, local_x)), max(0.0, min(height, local_y)), width, height
        return None

    def on_map_click(self, x: float, y: float, button: int, _modifier: int) -> None:
        """处理二维地图窗口中的点击并换算成世界坐标。"""
        if button != 0 or self.finished:
            return
        if self.start_and_opos and len(self.points) >= 2:
            self._set_status("Start and OPOS are already recorded; no APOS is accepted in this mode")
            return
        local_position = self._map_local_position(x, y)
        if local_position is None:
            self._set_status(f"Click coordinates are invalid: ({x:.1f}, {y:.1f})")
            return
        local_x, local_y, width, height = local_position
        result = self.occupancy_map.world_from_canvas(local_x, local_y, width, height)
        if result is None:
            self._set_status("Click is outside the map")
            return
        world_x, world_y, column, row = result
        if self.occupancy_map.occupied[row][column] and not self.allow_occupied:
            self._set_status("This cell is occupied; choose a light free-space cell")
            return
        if self.start_and_opos:
            role = "opos" if not self.points else "start"
        else:
            role = "opos" if not self.points else "apos"
        xyz = [world_x, world_y, self.occupancy_map.ground_z + 0.05]
        self.points.append({"role": role, "world": xyz, "map_cell": [column, row]})
        self._add_marker(len(self.points) - 1, role, xyz)
        if self.start_and_opos:
            opos_count = sum(point["role"] == "opos" for point in self.points)
            start_count = sum(point["role"] == "start" for point in self.points)
            self.points_label.text = f"Recorded points: {len(self.points)} (OPOS {opos_count}, START {start_count})"
        else:
            self.points_label.text = f"Recorded points: {len(self.points)} (OPOS 1, APOS {max(0, len(self.points) - 2)}, START 1)"
        self._set_status(f"Recorded {role.upper()}: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")
        self._refresh_map()
        self.save(silent=True)

    def _add_marker(self, index: int, role: str, xyz: list[float]) -> None:
        """在 SessionLayer 中绘制一个小球标记，便于核对点击位置。"""
        path = self._Sdf.Path(f"{self._marker_root}/{role.upper()}_{index:03d}")
        with self._Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            sphere = self._UsdGeom.Sphere.Define(self.stage, path)
            sphere.CreateRadiusAttr(0.20 if role == "opos" else 0.15)
            sphere.AddTranslateOp().Set(self._Gf.Vec3d(*xyz))
            colour = {
                "start": (0.10, 0.46, 0.82),
                "opos": (1.0, 0.15, 0.55),
                "apos": (0.15, 1.0, 0.65),
            }.get(role, (0.15, 1.0, 0.65))
            sphere.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*colour)])
            sphere.CreateVisibilityAttr().Set(self._UsdGeom.Tokens.inherited)
            sphere.CreatePurposeAttr().Set(self._UsdGeom.Tokens.default_)

    def _remove_markers(self) -> None:
        with self._Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            root = self.stage.GetPrimAtPath(self._Sdf.Path(self._marker_root))
            if root and root.IsValid():
                for child in list(root.GetChildren()):
                    self.stage.RemovePrim(child.GetPath())

    def undo(self) -> None:
        if not self.points:
            return
        removed = self.points.pop()
        self._remove_markers()
        for index, point in enumerate(self.points):
            self._add_marker(index, str(point["role"]), list(point["world"]))
        self.points_label.text = f"Recorded points: {len(self.points)}"
        self._set_status(f"Undid {str(removed['role']).upper()}; total points: {len(self.points)}")
        self._refresh_map()
        self.save(silent=True)

    def clear(self) -> None:
        self.points.clear()
        self._remove_markers()
        self.points_label.text = "Recorded points: 0"
        self._set_status("Cleared all points")
        self._refresh_map()
        self.save(silent=True)

    def save(self, *_args, silent: bool = False) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        root_layer = self.stage.GetRootLayer()
        scene_file = getattr(root_layer, "realPath", "") or root_layer.identifier
        self.occupancy_map.to_png(self.map_image_path, self.points)
        payload = {
            "scene": self.scene_name,
            "scene_file": str(scene_file),
            "confirm_frames": self.confirm_frames,
            "occupancy_map": {
                "file": str(self.map_image_path),
                "bounds_xy": [self.occupancy_map.min_x, self.occupancy_map.max_x, self.occupancy_map.min_y, self.occupancy_map.max_y],
                "ground_z": self.occupancy_map.ground_z,
                "resolution_m": self.occupancy_map.resolution,
                "width": self.occupancy_map.width,
                "height": self.occupancy_map.height,
            },
            # Keep only the ordered world coordinates. The last point is the
            # start; the selected runtime mode interprets all preceding points.
            "points": [point["world"] for point in self.points],
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not silent:
            self._set_status(f"Saved {len(self.points)} points: {self.output_path}")

    def finish(self, *_args) -> None:
        self.save()
        self.finished = True
        self._set_status("Saved; closing Isaac Sim")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.save(silent=True)
        self.window.visible = False


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    try:
        scenes = discover_scenes(scene_root)
        if args.list_scenes:
            for name, path in scenes:
                print(f"{name}\t{path}")
            return 0
        scene_name, scene_path = resolve_scene(args.scene, scene_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.confirm_frames < 1:
        parser.error("--confirm-frames 必须 >= 1")
    if args.map_resolution <= 0:
        parser.error("--map-resolution 必须 > 0")
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else scene_root / "pointing_routes" / f"{scene_name}_route.json"
    )

    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        print(f"无法导入 Isaac Sim，请使用 Isaac Sim Python 运行此脚本: {exc}", file=sys.stderr)
        return 1

    simulation_app = SimulationApp({
        "headless": False,
        "renderer": "RaytracedLighting",
        "open_usd": str(scene_path),
        "create_new_stage": False,
    })
    picker = None
    try:
        for _ in range(20):
            simulation_app.update()
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        occupancy_map = OccupancyMap.from_stage(stage, resolution=args.map_resolution)
        picker = PointPicker(
            stage,
            occupancy_map,
            output_path,
            scene_name,
            args.confirm_frames,
            args.allow_occupied,
            args.start_and_opos,
        )
        print(f"已打开场景: {scene_path}", flush=True)
        occupied_cells = sum(sum(row) for row in occupancy_map.occupied)
        total_cells = occupancy_map.width * occupancy_map.height
        print(
            f"Occupancy map: {occupancy_map.width}x{occupancy_map.height}, "
            f"resolution={occupancy_map.resolution:.3f} m, occupied={occupied_cells / total_cells:.1%}",
            flush=True,
        )
        print(
            "占据图已生成，请在二维地图中左键点击：第一次是 OPOS，第二次是起点。"
            if args.start_and_opos
            else "占据图已生成，请按远端到起点的顺序点击；最后一次点击作为起点。",
            flush=True,
        )
        print("可在 LightNav Point Picker 窗口中撤销、清空、保存或退出。", flush=True)
        while simulation_app.is_running() and not picker.finished:
            simulation_app.update()
        if picker is not None:
            picker.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
