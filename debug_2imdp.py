# debug_2imdp.py

import runpy
import sys


def main():

    sys.argv = [
        "Run2IMDP.py",
        "--model", "Dubins3D",
        "--batch_size", "10",
        "--num_samples", "500",
        "--coarse_factor", "3"
    ]

    runpy.run_path(
        "Run2IMDP.py",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()