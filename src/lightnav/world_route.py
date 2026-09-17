"""按世界坐标路线生成 Go2 的归一化运动指令。

路线选择器保存的是 Isaac Sim 世界坐标，而 Go2 控制器需要机器人坐标系
下的 ``[forward, lateral, yaw]`` 指令。本模块只做 CPU 数学和 JSON 校验，
因此可以在没有 Isaac Sim 或模型权重的环境中测试。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POINTING_MODES = ("pair", "opos_only", "multi_opos")


def _point(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{name} must contain at least x and y")
    try:
        result = tuple(float(value[index]) if index < len(value) else 0.0 for index in range(3))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def load_world_route(path: str | Path) -> tuple[tuple[float, float, float], ...]:
    """读取 picker 生成的路线，并返回 ``起点 -> OPOS`` 的执行顺序。

    新格式的 ``points`` 按点击顺序为 ``OPOS -> APOS... -> 起点``，所以执行时
    需要反转整个列表。旧格式的 APOS 同样按目标方向到起点记录。
    """
    route_path = Path(path).expanduser()
    with route_path.open("r", encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict):
        raise ValueError("route JSON must be an object")
    raw_points = document.get("points")
    if isinstance(raw_points, list) and raw_points:
        points = []
        for index, value in enumerate(raw_points):
            if isinstance(value, dict):
                value = value.get("world")
            points.append(_point(value, f"points[{index}]"))
        return tuple(reversed(points))
    goal = _point(document.get("opos"), "opos")
    raw_apos = document.get("apos", [])
    if not isinstance(raw_apos, list):
        raise ValueError("apos must be a list")
    apos = tuple(_point(value, f"apos[{index}]") for index, value in enumerate(raw_apos))
    return tuple(reversed(apos)) + (goal,)


@dataclass(frozen=True)
class RouteCommand:
    """一次路线更新的结果。"""

    command: tuple[float, float, float]
    point_index: int
    target_world: tuple[float, float, float] | None
    distance_m: float | None
    advanced: bool
    done: bool


@dataclass(frozen=True)
class SavedPointingRoute:
    """picker 路线中用于模型 pointing 的起点、中间目标和最终目标。"""

    start_world: tuple[float, float, float] | None
    apos_targets: tuple[tuple[float, float, float], ...]
    opos_world: tuple[float, float, float]


def route_targets_for_pointing_mode(
    route: SavedPointingRoute,
    mode: str,
) -> tuple[tuple[float, float, float], ...]:
    """返回某种 pointing 模式实际参与距离推进的目标序列。

    ``pair`` 把中间点作为 APOS、首个记录点作为最终 OPOS；``opos_only``
    只跟踪最终 OPOS；``multi_opos`` 则把起点之外的所有记录点都依次解释为
    OPOS。后两种模式在协议层都只发送 ``opos_id``。
    """
    normalized = str(mode).strip().lower()
    if normalized not in POINTING_MODES:
        choices = ", ".join(repr(value) for value in POINTING_MODES)
        raise ValueError(f"pointing mode must be one of: {choices}")
    if normalized == "opos_only":
        return (route.opos_world,)
    return route.apos_targets


def initial_route_yaw_degrees(
    route: SavedPointingRoute,
    mode: str,
) -> float | None:
    """返回从路线起点朝向第一个执行目标的世界坐标 yaw。"""
    if route.start_world is None:
        return None
    targets = route_targets_for_pointing_mode(route, mode)
    if not targets:
        return None
    dx = targets[0][0] - route.start_world[0]
    dy = targets[0][1] - route.start_world[1]
    if math.hypot(dx, dy) <= 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def prompt_pointing_for_mode(
    mode: str,
    *,
    apos_id: int,
    opos_id: int,
) -> dict[str, int]:
    """按模式构造 bridge 发给服务端的语义 pointing 字段。"""
    normalized = str(mode).strip().lower()
    if normalized not in POINTING_MODES:
        choices = ", ".join(repr(value) for value in POINTING_MODES)
        raise ValueError(f"pointing mode must be one of: {choices}")
    if normalized in ("opos_only", "multi_opos"):
        return {"opos_id": int(opos_id)}
    return {"apos_id": int(apos_id), "opos_id": int(opos_id)}


def load_saved_pointing_route(path: str | Path) -> SavedPointingRoute:
    """读取路线文件。

    新格式只保存按点击顺序排列的 ``points``：最后一个点是起点，其余点的
    pointing 语义由运行模式决定。``pair`` 把中间点解释为 APOS、首点解释为
    最终 OPOS；``multi_opos`` 把起点之外的点全部解释为 OPOS。旧的
    ``start``/``opos``/``apos`` 格式仍兼容，方便已有路线文件平滑迁移。
    """
    route_path = Path(path).expanduser()
    with route_path.open("r", encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict):
        raise ValueError("route JSON must be an object")
    raw_points = document.get("points")
    if isinstance(raw_points, list) and raw_points:
        points: list[tuple[float, float, float]] = []
        for index, value in enumerate(raw_points):
            # 兼容 picker 早期带 role/map_cell 的 points 记录。
            if isinstance(value, dict):
                value = value.get("world")
            points.append(_point(value, f"points[{index}]"))
        goal = points[0]
        start = points[-1]
        middle = points[1:-1]
        targets = tuple(reversed(middle)) + (goal,)
        return SavedPointingRoute(start, targets, goal)

    goal = _point(document.get("opos"), "opos")
    raw_apos = document.get("apos", [])
    if not isinstance(raw_apos, list):
        raise ValueError("apos must be a list")
    apos = tuple(_point(value, f"apos[{index}]") for index, value in enumerate(raw_apos))
    raw_start = document.get("start")
    if raw_start is not None:
        start = _point(raw_start, "start")
        targets = tuple(reversed(apos)) + (goal,)
    else:
        start = apos[-1] if apos else None
        # 旧格式的最后一个 APOS 是机器人起点；其余 APOS 从后往前走，最终把
        # OPOS 作为最后一个可行走点，确保 APOS_STOP 只在真正到达终点后发出。
        targets = tuple(reversed(apos[:-1])) + (goal,)
    return SavedPointingRoute(start, targets, goal)


class WorldRouteFollower:
    """将世界坐标路线转换成归一化 Go2 指令。

    ``reach_distance_m`` 控制何时切换下一个点。指令分量都限制在 ``[-1, 1]``；
    距离越近，线速度会按 ``slow_radius_m`` 自动减小，避免在目标附近冲过头。
    """

    def __init__(
        self,
        points_world: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]],
        *,
        reach_distance_m: float = 0.7,
        slow_radius_m: float = 2.0,
        max_heading_rad: float = math.pi / 2.0,
    ) -> None:
        if not points_world:
            raise ValueError("points_world must contain at least one point")
        if reach_distance_m <= 0.0 or slow_radius_m <= 0.0 or max_heading_rad <= 0.0:
            raise ValueError("route thresholds must be positive")
        self.points_world = tuple(_point(point, f"points_world[{index}]") for index, point in enumerate(points_world))
        self.reach_distance_m = float(reach_distance_m)
        self.slow_radius_m = float(slow_radius_m)
        self.max_heading_rad = float(max_heading_rad)
        self.point_index = 0
        self.done = False

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "WorldRouteFollower":
        return cls(load_world_route(path), **kwargs)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def update(self, robot_x: float, robot_y: float, robot_yaw: float) -> RouteCommand:
        """根据机器人当前世界位姿计算一帧指令。"""
        if not all(math.isfinite(float(value)) for value in (robot_x, robot_y, robot_yaw)):
            raise ValueError("robot pose must be finite")
        if self.done:
            return RouteCommand((0.0, 0.0, 0.0), len(self.points_world), None, None, False, True)

        advanced = False
        target = self.points_world[self.point_index]
        dx = target[0] - float(robot_x)
        dy = target[1] - float(robot_y)
        distance = math.hypot(dx, dy)
        if distance <= self.reach_distance_m:
            self.point_index += 1
            advanced = True
            if self.point_index >= len(self.points_world):
                self.done = True
                return RouteCommand((0.0, 0.0, 0.0), self.point_index, None, distance, True, True)
            target = self.points_world[self.point_index]
            dx = target[0] - float(robot_x)
            dy = target[1] - float(robot_y)
            distance = math.hypot(dx, dy)

        yaw = float(robot_yaw)
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading = math.atan2(lateral, forward)
        # 目标在身后时先转向，减少横向走位和目标点附近的摆动。
        forward_scale = 0.0 if abs(heading) > self.max_heading_rad else 1.0
        speed_scale = min(1.0, distance / self.slow_radius_m)
        command = (
            self._clamp(forward / self.slow_radius_m * speed_scale * forward_scale),
            self._clamp(lateral / self.slow_radius_m * speed_scale),
            self._clamp(heading / self.max_heading_rad),
        )
        return RouteCommand(command, self.point_index, target, distance, advanced, False)
