"""
Fixed-arguments launcher for experimentation.
"""

import subprocess
import sys
from pathlib import Path

# Abstraction to run -> the main script that builds and solves it.
MAIN_SCRIPTS = {
    "IMDP": "RunIMDP.py",
    "SVMDP": "RunSVMDP.py",
}

def run(abstraction: str, args: list[str]) -> None:
    if abstraction not in MAIN_SCRIPTS:
        raise ValueError(
            f"Unknown abstraction '{abstraction}'; expected one of {sorted(MAIN_SCRIPTS)}"
        )

    root = Path(__file__).resolve().parent
    runfile = root / MAIN_SCRIPTS[abstraction]

    cmd = [sys.executable, str(runfile), *args]
    subprocess.run(cmd, check=True, cwd=root)

def config_MountainCar() -> list[str]:
    return [
        "--model",
        "MountainCar",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_CartPole() -> list[str]:
    return [
        "--model",
        "CartPole",
        "--satprob",
        "0.99",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Dubins3D() -> list[str]:
    return [
        "--model",
        "Dubins3D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Dubins4D() -> list[str]:
    return [
        "--model",
        "Dubins4D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Drone4D() -> list[str]:
    return [
        "--model",
        "Drone4D",
        # "--batch_size",
        # "1000",
        # "--noise_distr",
        # "normal",
        "--solver",
        "jax",
    ]

def config_Drone6D_small() -> list[str]:
    return [
        "--model",
        "Drone6D_small",
        "--satprob",
        "0.95",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Drone6D() -> list[str]:
    return [
        "--model",
        "Drone6D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

# To run a particular benchmark, change the abstraction and the config in the calls below.
if __name__ == "__main__":
    # Which abstraction to build: "IMDP" or "SVMDP".
    ABSTRACTION = "IMDP"

    run(ABSTRACTION, config_MountainCar())
    # run(ABSTRACTION, config_CartPole())
    # run(ABSTRACTION, config_Dubins3D())
    # run(ABSTRACTION, config_Dubins4D())
    # run(ABSTRACTION, config_Drone4D())
    # run(ABSTRACTION, config_Drone6D_small())
    # run(ABSTRACTION, config_Drone6D())
