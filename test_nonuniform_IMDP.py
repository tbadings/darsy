import os
import sys
import runpy
from pathlib import Path

if __name__ == "__main__":

    # Add Spot executables to PATH
    spot_bin = Path.home() / "miniconda3/envs/spot-env/bin"
    os.environ["PATH"] = f"{spot_bin}{os.pathsep}{os.environ['PATH']}"

    # Set working directory
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    # Command-line arguments
    sys.argv = [
        "RunIMDP.py",
        "--model", "Temperature4D",
        "--noise_distr", "gaussian",
        "--coarse_factor", "1",
        "--num_samples", "100",
        "--batch_size", "1",
    ]

    # Run script
    runpy.run_path(
        str(project_dir / "RunIMDP.py"),
        run_name="__main__"
    )