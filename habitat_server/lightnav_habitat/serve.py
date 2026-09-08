"""Command-line entry point: serve one Habitat benchmark environment over ZeroMQ.

Example::

    HABITAT_SIM_GPU_ID=0 python -m lightnav_habitat.serve \\
        --task vlnce --config habitat_server/configs/vlnce_r2r.yaml --port 5555
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

from .objectnav import ObjectNavEnv
from .remote_server import RemoteEnvServer
from .vlnce import VLNCEEnv

_ENV_CLASSES = {
    "vlnce": VLNCEEnv,
    "objectnav": ObjectNavEnv,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Habitat environment server (VLN-CE / ObjectNav) over ZeroMQ"
    )
    parser.add_argument("--task", default="vlnce", choices=sorted(_ENV_CLASSES))
    parser.add_argument("--config", required=True, help="Habitat config yaml path")
    parser.add_argument("--port", type=int, default=5555, help="TCP port to bind (tcp://*:PORT)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Steps after which the episode is reported truncated "
        "(must not exceed the yaml's environment.max_episode_steps)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional RGB/depth sensor height override (default: yaml value)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Optional RGB/depth sensor width override (default: yaml value)",
    )
    parser.add_argument("--split-id", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--split-num", type=int, default=None, help="Number of shards")
    parser.add_argument(
        "--early-stop-rotation",
        type=int,
        default=0,
        help="Force STOP after this many consecutive steps without distance_to_goal change "
        "(0 = disabled)",
    )
    parser.add_argument(
        "--early-stop-steps",
        type=int,
        default=0,
        help="Force STOP after this many steps (0 = disabled)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split override (default: val_unseen for vlnce, the yaml value for objectnav)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Dataset root override; becomes <root>/{split}/{split}.json.gz "
        "(cannot express RxR's {split}_guide.json.gz - edit the yaml instead)",
    )
    parser.add_argument("--scenes-dir", default=None, help="Scene datasets directory override")
    parser.add_argument(
        "--success-distance",
        type=float,
        default=None,
        help="Success radius in meters. Default: 3.0 (vlnce) / 0.1 (objectnav, HM3D v1). "
        "Use 0.25 for HM3D-OVON.",
    )
    parser.add_argument(
        "--navmesh-cell-height",
        type=float,
        default=None,
        help="objectnav only: re-bake each scene's navmesh at this cell_height (m). "
        "REQUIRED as 0.05 for HM3D ObjectNav v2 (its episodes were generated on a "
        "cell_height=0.05 navmesh; the shipped one is 0.20, which floors "
        "distance_to_goal). Leave unset for HM3D v1 / MP3D (they align with the "
        "shipped navmesh) and by default for OVON.",
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help="Path touched once the simulator is initialized and the server accepts requests",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be provided together")
    if (args.split_id is None) != (args.split_num is None):
        parser.error("--split-id and --split-num must be provided together")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("lightnav_habitat.serve")

    # habitat-sim selects both the CUDA and the EGL device by UUID from gpu_device_id and
    # must see every GPU to match them, so CUDA_VISIBLE_DEVICES is deliberately NOT set
    # here. Callers who want strict isolation set CUDA_VISIBLE_DEVICES=<gpu> together with
    # HABITAT_SIM_GPU_ID=0.
    gpu_id = int(os.environ.get("HABITAT_SIM_GPU_ID", "0"))

    kwargs: Dict[str, Any] = {
        "gpu_id": gpu_id,
        "max_steps": args.max_steps,
        "early_stop_rotation": args.early_stop_rotation,
        "early_stop_steps": args.early_stop_steps,
    }
    if args.image_height is not None:
        kwargs["image_size"] = (args.image_height, args.image_width)
    if args.split_id is not None:
        kwargs["split_id"] = args.split_id
        kwargs["split_num"] = args.split_num
    if args.split is not None:
        kwargs["split"] = args.split
    if args.data_path:
        kwargs["data_path"] = args.data_path
    if args.scenes_dir:
        kwargs["scenes_dir"] = args.scenes_dir
    if args.success_distance is not None:
        kwargs["success_distance"] = args.success_distance
    if args.navmesh_cell_height is not None:
        if args.task != "objectnav":
            parser.error("--navmesh-cell-height only applies to --task objectnav")
        kwargs["navmesh_cell_height"] = args.navmesh_cell_height

    logger.info("Starting Habitat server")
    logger.info("task=%s config=%s port=%d gpu=%d", args.task, args.config, args.port, gpu_id)
    if args.image_height is None:
        logger.info("image_size=yaml sensor size")
    else:
        logger.info("image_size=%dx%d", args.image_height, args.image_width)
    if args.split_id is not None:
        logger.info("split=%d/%d", args.split_id, args.split_num)

    # Register the dataset loaders and measures with habitat's registry before the
    # environment (and therefore habitat.Env) is built.
    from . import objectnav_extensions, vlnce_extensions  # noqa: F401

    env_cls = _ENV_CLASSES[args.task]
    env = env_cls(config_path=args.config, **kwargs)
    server = RemoteEnvServer(env, address=f"tcp://*:{args.port}")
    server.start(ready_file=args.ready_file)


if __name__ == "__main__":
    main()
