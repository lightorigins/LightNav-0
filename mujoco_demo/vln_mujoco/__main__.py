from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from aiohttp import web

from .server import VlnMujocoServer
from .simulation import Simulation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vln_mujoco simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--vln-server", default="")
    parser.add_argument(
        "--robot",
        choices=("turtlebot", "microduck"),
        default="turtlebot",
    )
    parser.add_argument("--robot-model", type=Path)
    parser.add_argument("--walking-policy", type=Path)
    return parser.parse_args(argv)


def simulation_from_args(args: argparse.Namespace) -> Simulation:
    if args.robot == "turtlebot":
        if args.robot_model is not None or args.walking_policy is not None:
            raise SystemExit(
                "--robot-model and --walking-policy require --robot microduck"
            )
        return Simulation()
    if args.robot_model is None or args.walking_policy is None:
        raise SystemExit(
            "--robot microduck requires --robot-model and --walking-policy"
        )
    from .robots.microduck import MicroDuckBackend

    return Simulation(MicroDuckBackend(args.robot_model, args.walking_policy))


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    server = VlnMujocoServer(
        default_vln_server=args.vln_server,
        simulation=simulation_from_args(args),
    )
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"vln_mujoco ready: http://{display_host}:{args.port}", flush=True)
    web.run_app(server.app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
