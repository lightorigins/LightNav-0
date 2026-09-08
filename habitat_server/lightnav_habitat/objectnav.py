"""ObjectNav (HM3D v1 / MP3D v1 / HM3D-OVON) Habitat environment wrapper.

Inherits the velocity-control, early-stop and termination logic from ``VLNCEEnv``.
Differences: the instruction is synthesized from ``episode.object_category``
("Find the chair."), success is measured to the nearest goal viewpoint
(``success_distance`` 0.1 m for HM3D v1/v2, 0.25 m for OVON), there is no NDTW, and the
``velocity_control`` action is injected programmatically (the ObjectNav task config
does not declare one).

``navmesh_cell_height`` re-bakes each scene's navmesh at load time. HM3D ships
``<scene>.basis.navmesh`` baked with habitat defaults (``cell_height=0.20``); episode sets
generated on a ``cell_height=0.05`` navmesh (HM3D ObjectNav **v2**: start positions and goal
view points alike) then sit a constant 0.05-0.15 m *below* the shipped walkable surface, and
``geodesic_distance`` picks up that vertical residual as a hard floor on
``distance_to_goal`` - at the official 0.1 m radius the metric measures data alignment, not
the policy. Re-baking at 0.05 removes the artifact. HM3D v1 and MP3D align with the shipped
navmesh (audited: zero offset) and must be run WITHOUT this option.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .vlnce import (
    VLNCEEnv,
    _apply_image_size,
    _disable_episode_shuffle,
    _inject_nav_measurements,
)

logger = logging.getLogger(__name__)


class ObjectNavEnv(VLNCEEnv):
    """Habitat ObjectNav task wrapper (see module docstring).

    Metrics in ``info``: success, spl, soft_spl, distance_to_goal, path_length,
    oracle_success, steps_taken, plus ``object_category`` and ``goal_positions``.
    """

    # Human-readable instruction templates per object category.
    _CATEGORY_TEMPLATES: Dict[str, str] = {
        "chair": "Find the chair.",
        "bed": "Find the bed.",
        "sofa": "Find the sofa.",
        "plant": "Find the potted plant.",
        "toilet": "Find the toilet.",
        "tv_monitor": "Find the TV monitor.",
    }

    def __init__(
        self,
        config_path: str,
        data_path: Optional[str] = None,
        scenes_dir: Optional[str] = None,
        split: Optional[str] = None,
        gpu_id: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        max_steps: int = 500,
        success_distance: float = 0.1,  # HM3D ObjectNav standard: 0.1 m to a viewpoint
        split_id: Optional[int] = None,
        split_num: Optional[int] = None,
        early_stop_rotation: int = 0,
        early_stop_steps: int = 0,
        navmesh_cell_height: Optional[float] = None,
    ):
        # ``split=None`` keeps the yaml split (``val`` for HM3D / MP3D v1, ``val_unseen`` for OVON).
        self._object_category: str = ""
        # None = keep the shipped per-scene navmesh; a float (0.05 for HM3D ObjectNav v2)
        # re-bakes every scene's navmesh at that cell_height (see the module docstring).
        self._navmesh_cell_height = navmesh_cell_height
        self._navmesh_agent_radius = 0.18
        self._navmesh_agent_height = 0.88
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            scenes_dir=scenes_dir,
            split=split,
            gpu_id=gpu_id,
            image_size=image_size,
            max_steps=max_steps,
            success_distance=success_distance,
            split_id=split_id,
            split_num=split_num,
            early_stop_rotation=early_stop_rotation,
            early_stop_steps=early_stop_steps,
        )

    # -- config assembly --------------------------------------------------------

    def _create_habitat_env(self) -> Any:
        try:
            import habitat
            from habitat.config.read_write import read_write
        except ImportError as e:
            raise ImportError(f"habitat-lab not installed: {e}")
        from habitat.config.default_structured_configs import VelocityControlActionConfig
        from omegaconf import OmegaConf

        # Tolerant "ObjectNav-v1" loader (OVON episode fields / empty category maps) and the
        # PathLength/OracleSuccess/StepsTaken measures; must precede habitat.Env.
        from . import objectnav_extensions, vlnce_extensions  # noqa: F401

        logger.info(f"Loading ObjectNav config from: {self.config_path}")
        config = habitat.get_config(self.config_path)

        with read_write(config):
            self._apply_dataset_overrides(config)
            _disable_episode_shuffle(config)
            _apply_image_size(config, self.image_size)

            # Keep the ObjectNav built-ins (distance_to_goal to VIEW_POINTS, success, spl,
            # soft_spl); add path_length / oracle_success / steps_taken. No NDTW.
            _inject_nav_measurements(config, self.success_distance)

            # Inject velocity_control as a structured config so OmegaConf validates it.
            actions = config.habitat.task.actions
            OmegaConf.set_struct(actions, False)
            actions.velocity_control = OmegaConf.structured(
                VelocityControlActionConfig(
                    lin_vel_range=[0.0, 0.25],  # m/s   -> 0.25 m per step at dt=1 s
                    ang_vel_range=[-30.0, 30.0],  # deg/s -> 30 deg per step at dt=1 s
                    time_step=1.0,
                    min_abs_lin_speed=0.025,
                    min_abs_ang_speed=1.0,
                )
            )
            OmegaConf.set_struct(actions, True)

            self._cache_velocity_control_config(config)

        self.split = str(config.habitat.dataset.split)
        logger.info(
            f"ObjectNav config loaded - task: {config.habitat.task.type}, "
            f"dataset: {config.habitat.dataset.type}, split: {self.split}"
        )

        agent_cfg = config.habitat.simulator.agents.main_agent
        self._navmesh_agent_radius = float(agent_cfg.radius)
        self._navmesh_agent_height = float(agent_cfg.height)

        env = habitat.Env(config=config)

        if self._navmesh_cell_height is not None:
            self._install_navmesh_rebake(env, float(self._navmesh_cell_height))

        try:
            total_eps = len(env._dataset.episodes)
            categories: Dict[str, int] = {}
            for ep in env._dataset.episodes:
                cat = getattr(ep, "object_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            cat_str = ", ".join(f"{k}={v}" for k, v in sorted(categories.items()))
            logger.info(f"Dataset loaded: {total_eps} episodes. Categories: {cat_str}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not summarize dataset: {e}")

        self._apply_episode_split(env)
        return env

    # -- navmesh re-bake ----------------------------------------------------------

    def _install_navmesh_rebake(self, env: Any, cell_height: float) -> None:
        """Re-bake the navmesh once per scene, at ``cell_height``, via the public
        ``sim.recompute_navmesh`` API.

        The hook wraps ``sim.reconfigure`` - NOT ``env.reset()``: habitat's order is
        *instance scene -> sim.reconfigure -> task.reset() builds the measures*, so a
        ``reset()`` hook would score each scene's FIRST episode (including its
        ``_start_geodesic_distance``, the SPL denominator) on the stale navmesh.
        A failed re-bake logs and falls back to the shipped navmesh; it never aborts.
        """
        import habitat_sim

        sim = env.sim
        radius, height = self._navmesh_agent_radius, self._navmesh_agent_height
        original_reconfigure = sim.reconfigure
        state = {"scene": None}  # None, not the boot scene: the first reconfigure must re-bake

        def _scene_name() -> str:
            name = getattr(sim, "curr_scene_name", None)
            if not name:
                try:
                    name = sim.habitat_config.scene
                except Exception:  # noqa: BLE001
                    name = "?"
            return str(name)

        def reconfigure(*args: Any, **kwargs: Any) -> Any:
            result = original_reconfigure(*args, **kwargs)
            scene = _scene_name()
            if scene == state["scene"]:
                return result
            state["scene"] = scene
            try:
                settings = habitat_sim.nav.NavMeshSettings()
                settings.set_defaults()
                settings.agent_radius = radius
                settings.agent_height = height
                settings.cell_height = cell_height
                t0 = time.monotonic()
                ok = sim.recompute_navmesh(sim.pathfinder, settings)
                logger.info(
                    "[navmesh] rebaked %s at cell_height=%.2f (r=%.2f, h=%.2f): "
                    "ok=%s area=%.1fm2 in %dms",
                    scene, cell_height, radius, height, ok,
                    float(sim.pathfinder.navigable_area),
                    int((time.monotonic() - t0) * 1000),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[navmesh] rebake failed for %s (%s); keeping the shipped navmesh", scene, e)
            return result

        sim.reconfigure = reconfigure
        logger.info("[navmesh] rebake hook installed (cell_height=%.2f)", cell_height)

    # -- observations -----------------------------------------------------------

    def _compute_progress(self) -> float:
        """Progress against the viewpoint distance reported by the DistanceToGoal measure."""
        progress = 0.0
        env = self._habitat_env
        if env is None:
            return progress
        try:
            dtg = float(env.get_metrics().get("distance_to_goal", float("inf")))
            start = self._start_geodesic_distance
            if np.isfinite(dtg) and np.isfinite(start) and start > 0:
                progress = float(np.clip((start - dtg) / start, 0.0, 1.0))
        except Exception:
            pass
        return progress

    def _extract_task_obs(self, habitat_obs: Dict) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}

        # Instruction from the current episode's object_category (valid after reset()).
        category = ""
        if self._habitat_env is not None:
            episode = getattr(self._habitat_env, "current_episode", None)
            if episode is not None:
                category = getattr(episode, "object_category", "")

        if category:
            instruction_text = self._CATEGORY_TEMPLATES.get(
                category,
                f"Find the {category.replace('_', ' ')}.",
            )
            self._object_category = category
        else:
            instruction_text = self._current_instruction  # keep the previous one on error

        obs["instruction"] = {"text": instruction_text}
        self._current_instruction = instruction_text
        obs["goal_distance"] = np.array([float("inf")], dtype=np.float32)
        obs["progress"] = np.array([self._compute_progress()], dtype=np.float32)
        return obs

    # -- reset / info -------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset and re-baseline the progress start distance to the nearest viewpoint.

        ``VLNCEEnv.reset`` measures the start distance to ``goals[0].position`` (object
        centre). For ObjectNav the reference is the distance to the nearest viewpoint, which
        is what Habitat's DistanceToGoal reports in ``info["distance_to_goal"]``.
        """
        obs, info = super().reset(seed=seed, options=options)

        if "distance_to_goal" in info:
            dtg = float(info["distance_to_goal"])
            if np.isfinite(dtg) and dtg > 0.0:
                self._start_geodesic_distance = dtg

        if self._object_category:
            info["object_category"] = self._object_category

        return obs, info

    def _compute_info(self, habitat_obs: Dict) -> Dict[str, Any]:
        info = super()._compute_info(habitat_obs)
        if self._object_category:
            info["object_category"] = self._object_category
        # ObjectNav episodes target every instance of the category: one goal per instance.
        if hasattr(self._habitat_env, "current_episode"):
            episode = self._habitat_env.current_episode
            if getattr(episode, "goals", None):
                info["goal_positions"] = [[float(x) for x in g.position] for g in episode.goals]
        return info
