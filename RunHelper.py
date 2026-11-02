"""
Fixed-arguments launcher for experimentation.
"""

import subprocess
import sys
from pathlib import Path

# Abstraction to run -> the main script that builds and solves it.
MAIN_SCRIPTS = {
    "IMDP": "RunIMDP.py",
    "2IMDP": "Run2IMDP.py",
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
    # Which abstraction to build: "IMDP" or "2IMDP".
    ABSTRACTION = "IMDP"

    # Optional extra arguments: "--batch_size", "1000" / "--noise_distr", "normal".

    # run(ABSTRACTION, ["--model", "MountainCar", "--solver", "jax"])

    # run(ABSTRACTION, ["--model", "Pendulum", "--solver", "jax"])

    # run(ABSTRACTION, ["--model", "Dubins3D", "--solver", "jax"])

    # run(ABSTRACTION, ["--model", "Dubins4D", "--solver", "jax"])

    run(ABSTRACTION, ["--model", "Drone4D", "--solver", "jax"])

    # Still too big to run:
    # run(ABSTRACTION, ["--model", "Drone6D_small", "--satprob", "0.95", "--solver", "jax"])
    # run(ABSTRACTION, ["--model", "Drone6D", "--solver", "jax"])
    # run(ABSTRACTION, ["--model", "CartPole", "--satprob", "0.99", "--solver", "jax"])
