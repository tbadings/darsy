#!/usr/bin/env bash
# Run all benchmark experiments. Each line can also be copied into a terminal at the repository root.
#
# Usage (from anywhere):  bash experiments/run_all.sh
#
# Expected runtimes are wall-clock times per line (IMDP + 2IMDP, 1k/10k/100k samples, one seed),
# measured on the development Mac on 2026-11-03 ("measured") or extrapolated from runs of similar
# size ("estimate"). Abstraction time grows roughly 7-10x per 10x samples, so the 100k runs dominate.
# Expected total: roughly 10-14 hours.

cd "$(dirname "$0")/.."

# Expected runtime: ~2 min (estimate; 1000 states)
bash experiments/run_benchmark.sh Temperature1D_reach    --seeds "0" --noise_distr gaussian
# Expected runtime: ~1 min (estimate; 100 states)
bash experiments/run_benchmark.sh Temperature1D_safety   --seeds "0" --noise_distr gaussian
# Expected runtime: ~2 min (estimate; 144 states)
bash experiments/run_benchmark.sh Temperature2D          --seeds "0" # Uses truncated Gaussian noise
# Expected runtime: ~5 min (estimate; 1728 states x 8 actions)
bash experiments/run_benchmark.sh Temperature3D          --seeds "0" # Uses truncated Gaussian noise


# Expected runtime: ~2 min (measured)
bash experiments/run_benchmark.sh Dubins2D_scLTL         --seeds "0"
# Expected runtime: ~11 min (measured)
bash experiments/run_benchmark.sh DoubleIntegrator       --seeds "0"
# Expected runtime: ~12 min (estimate; same partition and actions as DoubleIntegrator)
bash experiments/run_benchmark.sh DoubleIntegrator_scLTL --seeds "0"
# Expected runtime: ~17 min (measured)
bash experiments/run_benchmark.sh Pendulum               --seeds "0"


# Expected runtime: ~4 h
bash experiments/run_benchmark.sh Temperature4D          --seeds "0" # Uses truncated Gaussian noise


# Expected runtime: ~4 h (IMDP runs measured at ~1 h; 2IMDP runs estimated at ~3 h)
bash experiments/run_benchmark.sh MountainCar            --seeds "0"


# Expected runtime: ~2-4 h (rough estimate; 125k states x 9 actions)
bash experiments/run_benchmark.sh Dubins3D               --seeds "0"


# Expected runtime: ~2-4 h (rough estimate; as Dubins3D, plus the product with the automaton)
bash experiments/run_benchmark.sh Dubins3D_scLTL         --seeds "0"