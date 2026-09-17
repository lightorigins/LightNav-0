"""把世界坐标路径点转换为相对于图像的 ``apos`` 指引。

LightNav 的 pointing 词表描述的是图像位置，而 Isaac Sim 中的路径通常
使用世界坐标编写。本模块负责连接这两个坐标系，但不让模型或 WebSocket
服务负责路径进度：

* 通过相机投影当前世界坐标路径点；
* 要求该点连续若干帧处于图像中央区域；
* 可选地要求机器人已经足够接近该点；
* 路径结束后输出下一个点，或 ``apos`` 停止哨兵。

``camera_from_world`` 是世界坐标到相机坐标的齐次变换矩阵。
``camera_convention`` 支持 OpenCV 坐标约定（``+Z`` 向前、``+Y`` 向下）
和 Isaac USD 坐标约定（``-Z`` 向前、``+Y`` 向上，默认值）。从 Isaac Sim
获取相机位姿的适配代码仍由调用方负责，因此本模块可以在 CPU 单测中运行。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from lightnav.vln_utils import APOS_STOP_ID, encode_point_pixel


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参和图像像素尺寸。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
    ) -> "CameraIntrinsics":
        """根据水平视场角构造内参。

        如果 Isaac 相机能提供单独的垂直焦距，调用方可以直接构造
        ``CameraIntrinsics``。这里只能拿到水平视场角时，沿用 LightNav
        当前的近似：``fx == fy``。
        """
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("camera width and height must be positive")
        if not math.isfinite(float(hfov_deg)) or not 0.0 < float(hfov_deg) < 180.0:
            raise ValueError("hfov_deg must be finite and in (0, 180)")
        fx = (float(width) / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
        fy = fx
        return cls(int(width), int(height), fx, fy, float(width) / 2.0, float(height) / 2.0)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class ProjectedPoint:
    """一个世界坐标路径点的投影结果。"""

    u: float
    v: float
    depth: float

    @property
    def in_image(self) -> bool:
        return self.depth > 0.0 and math.isfinite(self.u) and math.isfinite(self.v)


@dataclass(frozen=True)
class AposRouteState:
    """由 :meth:`AposRoute.update` 返回、供调用方观察的路径状态。"""

    point_index: int
    done: bool
    advanced: bool
    visible: bool
    in_center: bool
    distance_m: float | None
    projection: ProjectedPoint | None
    apos_id: int | None
    apos_token: str | None
    advance_reason: str | None = None
    next_distance_m: float | None = None
    best_distance_m: float | None = None


def project_world_point(
    point_world: np.ndarray | list[float] | tuple[float, float, float],
    camera_from_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    camera_convention: str = "isaac_usd",
) -> ProjectedPoint | None:
    """把一个世界坐标点投影到相机图像。

    ``camera_from_world`` 必须是 4x4 的世界到相机变换矩阵。在 Isaac USD
    约定下，相机局部 ``-Z`` 轴向前、``+Y`` 轴向上；返回的 ``v`` 会转换为
    图像坐标系中的向下为正。输入形状错误或包含非有限值时返回 ``None``。
    """
    if camera_convention not in ("isaac_usd", "opencv"):
        raise ValueError("camera_convention must be 'isaac_usd' or 'opencv'")
    transform = np.asarray(camera_from_world, dtype=np.float64)
    point = np.asarray(point_world, dtype=np.float64)
    if transform.shape != (4, 4) or point.shape != (3,):
        raise ValueError("camera_from_world must be (4, 4) and point_world must be (3,)")
    if not np.all(np.isfinite(transform)) or not np.all(np.isfinite(point)):
        return None

    # 齐次坐标变换把地图点带到相机坐标系，后续透视除法才有意义。
    camera_point = transform @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    if camera_convention == "isaac_usd":
        # USD 相机沿局部 -Z 看向场景；图像 v 轴向下，所以 Y 需要反号。
        depth = -float(camera_point[2])
        if depth <= 0.0:
            return ProjectedPoint(float("nan"), float("nan"), depth)
        u = intrinsics.cx + intrinsics.fx * float(camera_point[0]) / depth
        v = intrinsics.cy - intrinsics.fy * float(camera_point[1]) / depth
    else:
        # OpenCV 约定使用局部 +Z 向前，并且图像 v 轴与相机 Y 同向向下。
        depth = float(camera_point[2])
        if depth <= 0.0:
            return ProjectedPoint(float("nan"), float("nan"), depth)
        u = intrinsics.cx + intrinsics.fx * float(camera_point[0]) / depth
        v = intrinsics.cy + intrinsics.fy * float(camera_point[1]) / depth
    return ProjectedPoint(u, v, depth)


class AposRoute:
    """根据稳定的图像证据依次推进世界坐标路径点。

    默认只有当路径点投影位于图像中央矩形内，并且（设置
    ``max_distance_m`` 时）机器人距离不超过该阈值，才具备切换资格。
    ``require_visible=False`` 时，路线推进只使用距离条件，但图像外的点仍然
    不会生成被夹到边缘的 token。资格可要求连续 ``confirm_frames`` 次保持。
    ``advance_when_passed=True`` 还允许在机器人明确越过当前点并接近下一点时
    推进，用于避免模型轨迹绕过中间航点后永久卡住。
    """

    def __init__(
        self,
        points_world: list[np.ndarray | list[float] | tuple[float, float, float]],
        intrinsics: CameraIntrinsics,
        *,
        center_fraction: float | tuple[float, float] = 0.5,
        confirm_frames: int = 3,
        max_distance_m: float | None = 2.0,
        require_visible: bool = True,
        advance_when_passed: bool = False,
        pass_hysteresis_m: float = 0.5,
        camera_convention: str = "isaac_usd",
    ) -> None:
        if not points_world:
            raise ValueError("points_world must contain at least one point")
        points = tuple(np.asarray(point, dtype=np.float64) for point in points_world)
        if any(point.shape != (3,) or not np.all(np.isfinite(point)) for point in points):
            raise ValueError("every route point must be a finite 3-vector")
        if isinstance(center_fraction, tuple):
            if len(center_fraction) != 2:
                raise ValueError("center_fraction tuple must be (width_fraction, height_fraction)")
            center_w, center_h = (float(value) for value in center_fraction)
        else:
            center_w = center_h = float(center_fraction)
        if not (0.0 < center_w <= 1.0 and 0.0 < center_h <= 1.0):
            raise ValueError("center_fraction values must be in (0, 1]")
        if int(confirm_frames) < 1:
            raise ValueError("confirm_frames must be >= 1")
        if max_distance_m is not None and (
            not math.isfinite(float(max_distance_m)) or float(max_distance_m) <= 0.0
        ):
            raise ValueError("max_distance_m must be positive and finite, or None")
        if not math.isfinite(float(pass_hysteresis_m)) or float(pass_hysteresis_m) < 0.0:
            raise ValueError("pass_hysteresis_m must be non-negative and finite")
        if camera_convention not in ("isaac_usd", "opencv"):
            raise ValueError("camera_convention must be 'isaac_usd' or 'opencv'")

        self.points_world = points
        self.intrinsics = intrinsics
        self.center_fraction = (center_w, center_h)
        self.confirm_frames = int(confirm_frames)
        self.max_distance_m = None if max_distance_m is None else float(max_distance_m)
        self.require_visible = bool(require_visible)
        self.advance_when_passed = bool(advance_when_passed)
        self.pass_hysteresis_m = float(pass_hysteresis_m)
        self.camera_convention = camera_convention
        self.point_index = 0
        self._eligible_streak = 0
        self._best_distance_m: float | None = None
        self.done = False

    @property
    def current_point_world(self) -> np.ndarray | None:
        if self.done:
            return None
        return self.points_world[self.point_index].copy()

    def reset(self) -> None:
        self.point_index = 0
        self._eligible_streak = 0
        self._best_distance_m = None
        self.done = False

    def update(
        self,
        camera_from_world: np.ndarray,
        *,
        robot_world: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    ) -> AposRouteState:
        """处理一帧相机位姿，并返回当前 ``apos`` 状态。"""
        if self.done:
            return AposRouteState(
                len(self.points_world), True, False, False, False, None, None,
                APOS_STOP_ID, f"<apos_{APOS_STOP_ID}>",
            )

        robot = None if robot_world is None else np.asarray(robot_world, dtype=np.float64)
        if robot is not None and (robot.shape != (3,) or not np.all(np.isfinite(robot))):
            raise ValueError("robot_world must be a finite 3-vector")
        target = self.points_world[self.point_index]
        projection = project_world_point(
            target,
            camera_from_world,
            self.intrinsics,
            camera_convention=self.camera_convention,
        )
        visible = bool(
            projection is not None
            and projection.in_image
            and 0.0 <= projection.u < self.intrinsics.width
            and 0.0 <= projection.v < self.intrinsics.height
        )
        center_w, center_h = self.center_fraction
        in_center = bool(
            visible
            and abs(projection.u - self.intrinsics.cx) <= self.intrinsics.width * center_w / 2.0
            and abs(projection.v - self.intrinsics.cy) <= self.intrinsics.height * center_h / 2.0
        )
        # 距离门限是可选的；没有机器人位姿时，只有关闭门限（None）才会放行。
        distance = None if robot is None else float(np.linalg.norm(target - robot))
        if distance is not None and (
            self._best_distance_m is None or distance < self._best_distance_m
        ):
            self._best_distance_m = distance
        next_distance = None
        passed = False
        if (
            self.advance_when_passed
            and robot is not None
            and distance is not None
            and self._best_distance_m is not None
            and self.point_index + 1 < len(self.points_world)
        ):
            next_target = self.points_world[self.point_index + 1]
            next_distance = float(np.linalg.norm(next_target - robot))
            segment_xy = next_target[:2] - target[:2]
            beyond_current = bool(
                np.linalg.norm(segment_xy) > 1e-6
                and np.dot(robot[:2] - target[:2], segment_xy) > 0.0
            )
            moved_away = distance >= self._best_distance_m + self.pass_hysteresis_m
            passed = beyond_current and moved_away and next_distance < distance
        distance_ok = self.max_distance_m is None or (distance is not None and distance <= self.max_distance_m)
        image_ok = in_center if self.require_visible else True
        reached = image_ok and distance_ok
        eligible = reached or passed
        # 任何一帧不满足启用的门槛都会清零，避免抖动误触发切换。
        self._eligible_streak = self._eligible_streak + 1 if eligible else 0
        advanced = False
        advance_reason = None
        if self._eligible_streak >= self.confirm_frames:
            self._eligible_streak = 0
            self.point_index += 1
            advanced = True
            advance_reason = "reached" if reached else "passed"
            if self.point_index >= len(self.points_world):
                # 所有地图点都确认完成，使用 1299 告知上层进入停止状态。
                self.done = True
                self._best_distance_m = None
            elif robot is not None:
                self._best_distance_m = float(
                    np.linalg.norm(self.points_world[self.point_index] - robot)
                )
            else:
                self._best_distance_m = None

        if self.done:
            return AposRouteState(
                len(self.points_world), True, advanced, visible, in_center, distance,
                projection, APOS_STOP_ID, f"<apos_{APOS_STOP_ID}>",
                advance_reason, next_distance, self._best_distance_m,
            )

        # 切换后立即重新投影，保证本次返回的 token 描述的是新目标点。
        if advanced:
            projection = project_world_point(
                self.points_world[self.point_index],
                camera_from_world,
                self.intrinsics,
                camera_convention=self.camera_convention,
            )
            visible = bool(
                projection is not None
                and projection.in_image
                and 0.0 <= projection.u < self.intrinsics.width
                and 0.0 <= projection.v < self.intrinsics.height
            )
            in_center = bool(
                visible
                and abs(projection.u - self.intrinsics.cx) <= self.intrinsics.width * center_w / 2.0
                and abs(projection.v - self.intrinsics.cy) <= self.intrinsics.height * center_h / 2.0
            )

        apos_id = None
        apos_token = None
        if visible and projection is not None:
            apos_id = encode_point_pixel(
                projection.u,
                projection.v,
                self.intrinsics.width,
                self.intrinsics.height,
            )
            apos_token = f"<apos_{apos_id}>"
        return AposRouteState(
            self.point_index, False, advanced, visible, in_center, distance,
            projection, apos_id, apos_token,
            advance_reason, next_distance, self._best_distance_m,
        )
