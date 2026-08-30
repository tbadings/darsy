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

# To run a particular benchmark, change the abstraction and uncomment one of the calls below.
if __name__ == "__main__":
    # Which abstraction to build: "IMDP" or "SVMDP".
    ABSTRACTION = "SVMDP"

    # Optional extra arguments: "--batch_size", "1000" / "--noise_distr", "normal".

    # run("IMDP", ["--model", "MountainCar", "--solver", "jax"])
    # run("SVMDP", ["--model", "MountainCar", "--solver", "jax"])

    # run("IMDP", ["--model", "Pendulum", "--solver", "jax"])
    # run("SVMDP", ["--model", "Pendulum", "--solver", "jax"])

    # run("IMDP", ["--model", "Dubins3D", "--solver", "jax"])
    # run("SVMDP", ["--model", "Dubins3D", "--solver", "jax"])

    # run("IMDP", ["--model", "Dubins4D", "--solver", "jax"])
    # run("SVMDP", ["--model", "Dubins4D", "--solver", "jax"])

    # run("IMDP", ["--model", "Drone4D", "--solver", "jax"])
    # run("SVMDP", ["--model", "Drone4D", "--solver", "jax"])

    # Still too big to run:
    # run("IMDP", ["--model", "Drone6D_small", "--satprob", "0.95", "--solver", "jax"])
    # run("SVMDP", ["--model", "Drone6D_small", "--satprob", "0.95", "--solver", "jax"])
    # run("IMDP", ["--model", "Drone6D", "--solver", "jax"])
    # run("SVMDP", ["--model", "Drone6D", "--solver", "jax"])
    # run("IMDP", ["--model", "CartPole", "--satprob", "0.99", "--solver", "jax"])
    # run("SVMDP", ["--model", "CartPole", "--satprob", "0.99", "--solver", "jax"])