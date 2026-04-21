from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for Hydra scaling sweeps. Prefer running the printed command directly."
    )
    parser.add_argument("--config-name", default="scaling", help="Hydra config name, e.g. scaling or scaling_smoke.")
    parser.add_argument("--run", action="store_true", help="Execute the suggested Hydra multirun command.")
    return parser.parse_args()


def build_command(config_name: str) -> list[str]:
    if config_name == "scaling_smoke":
        return [
            sys.executable,
            "-m",
            "training.train",
            "--config-name",
            "scaling_smoke",
            "-m",
            "model_size=tiny",
            "scaling.target_tokens=1000,3000",
            "optimization.lr=0.0003",
            "runtime.seed=1337",
        ]
    return [
        sys.executable,
        "-m",
        "training.train",
        "--config-name",
        config_name,
        "-m",
        "model_size=tiny,small,base",
        "scaling.target_tokens=100000,300000,1000000,3000000,10000000",
        "optimization.lr=0.0001,0.0003,0.001",
        "runtime.seed=1337",
    ]


def main() -> None:
    args = parse_args()
    command = build_command(str(args.config_name))
    print("Hydra multirun command:")
    print(" ".join(command))
    print(f"cwd={REPO_ROOT}")
    if args.run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
