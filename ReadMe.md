# IMDP Synthesis

This repository implements abstraction-based methods for robust policy synthesis in discrete-time stochastic dynamical systems, based on interval Markov decision process (IMDP) and two-layer IMDP (2IMDP) abstractions.

# Installation

Follow the steps below to install the tool:

## 1. Python environment

The project is currently documented and tested with Python 3.13. We recommend creating a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Next, install the base Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. JAX installation

JAX should be installed differently depending on whether you wish to run the code on CPU or GPU.

For CPU-only execution:

```bash
pip install jax
```

For CUDA-enabled execution via conda:

```bash
pip install "jax[cuda13]"
```

Apple Silicon should use the CPU version of JAX.

# Project entrypoints

There is one entrypoint per abstraction:

```bash
python RunIMDP.py --model <model_name>     # IMDP abstraction
python Run2IMDP.py --model <model_name>    # Two-layer IMDP abstraction
```

Both share the run pipeline in `core/pipeline.py` and the command-line options. `Run2IMDP.py` builds a coarse layer controlled by `--coarse_factor` and uses JAX value iteration. `--solver storm` and policy iteration apply to `RunIMDP.py` only.

Both entrypoints use a full `DensePartition` with grid edges aligned to labeled regions.
`Run2IMDP.py` groups those fine cells into a `CoarsePartition` for its second layer.

`RunHelper.py` is a convenience launcher for local experimentation: set `ABSTRACTION` to `"IMDP"` or `"2IMDP"` and enable the desired `run(...)` line to replay a fixed configuration.

# Available benchmarks

The current benchmark registry includes:

- `Dubins3D`: unicycle with position and heading states; steering rate and velocity inputs.
- `Dubins4D`: 4D Dubins vehicle with 2D control input.
- `Drone4D`: 4D quadrotor model.
- `Pendulum`: inverted pendulum benchmark.
- `MountainCar`: mountain car benchmark.
- `DoubleIntegrator`: double integrator benchmark.
- `Test1D`: simple 1D test model.

Included but untested benchmarks are:

- `Drone6D`: 6D quadrotor model.
- `Drone6D_small`: reduced 6D quadrotor configuration for faster debugging.
- `CartPole`: cart-pole benchmark.

Example runs:

`Dubins3D` and `Dubins3D_scLTL` use a 1 s step, steering coefficient 0.85, and
heading wrapped to `[-pi, pi)`. Their nine controls are the Cartesian product of
steering rates `{-pi/2, 0, pi/2}` and velocities `{-1, 0, 1}`. Their step function
adds noise only to the heading, using zero-mean Gaussian noise with standard
deviation 0.05 by default, or triangular noise with halfwidth 0.05.
Other models still default to triangular noise; `--noise_distr` explicitly
overrides the default.

```bash
python RunIMDP.py --model Dubins3D --solver jax
python RunIMDP.py --model Drone4D --solver storm
python Run2IMDP.py --model MountainCar --solver jax --no-policy_iteration
```

# Running a benchmark

A typical command looks as follows:

```bash
python RunIMDP.py --model MountainCar --solver jax --noise_distr gaussian
```

This will:

1. create a timestamped output directory under `output/`,
2. construct the partition and IMDP abstraction,
3. solve the robust reachability problem,
4. run Monte Carlo validation,
5. save logs, plots, and a `checkpoint.pkl` file (the last only with `--save_checkpoint`).

Steps 4 and 5 are best effort: a failure while simulating or plotting is logged (with its traceback in `run_<timestamp>.log`) and the remaining steps still run, so a plotting bug cannot discard a finished synthesis. A closing `N post-processing step(s) failed: ...` line lists whatever did not produce output.

## Reusing a checkpoint

To skip abstraction generation and rerun the solver and plotting pipeline from a saved checkpoint:

```bash
python RunIMDP.py --model MountainCar --load_checkpoint output/<timestamp>_MountainCar/checkpoint.pkl
```

When `--load_checkpoint` is provided, the saved model, partition, and abstraction are loaded directly from the specified pickle file. Checkpoints are specific to the entrypoint that wrote them.

# Solver backends

Two solver backends are supported:

- `--solver jax`: robust dynamic programming implemented in JAX.
- `--solver storm`: robust value iteration through Storm.

Examples:

```bash
python RunIMDP.py --model Pendulum --solver jax
python RunIMDP.py --model Test1D --solver storm --no-policy_iteration
```

Note: the Storm backend currently supports robust value iteration, not robust policy iteration. If `--solver storm` is selected together with policy iteration, the code emits a warning and falls back to robust value iteration.

# Important command-line options

## Model and solver selection

- `--model`: benchmark name.
- `--solver {jax,storm}`: select the backend.
- `--noise_distr {gaussian,normal,triangular}`: select the noise model. `normal` is accepted as an alias of `gaussian`.
- `--policy_iteration` or `--no-policy_iteration`: robust policy iteration (default) or value iteration.
- `--satprob`: stop improving the policy once the value in the initial state reaches this bound, and only evaluate it to convergence. Policy iteration only.

## Performance and memory tuning

- `--batch_size`: number of states processed together when computing transition probability intervals.
- `--frs_batch_size`: number of regions processed per batch when computing forward reachable sets.
- `--coarse_factor`: number of fine cells per coarse block dimension for `Run2IMDP.py` (default: `3`).
- `--gpu` or `--no-gpu`: select the main JAX platform.
- `--gpu_rvi` or `--no-gpu_rvi`: select the device used for robust dynamic programming.

If you run into memory limits, reduce `--batch_size` first. On larger benchmarks, values such as `1000`, `100`, or even smaller may be necessary.

## Reproducibility and logging

- `--seed`: random seed for NumPy and JAX.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: logging verbosity.
- `--output_root`: root directory for timestamped run folders.

## Plotting controls

- `--plot_title` or `--no-plot_title`
- `--plot_grid` or `--no-plot_grid`
- `--plot_ticks` or `--no-plot_ticks`

# Output structure

Each run creates an output directory named like:

```text
output/2026-06-02_10-49-08_MountainCar/
```

Depending on the benchmark and selected options, that directory may contain:

- `checkpoint.pkl`: serialized model, partition, IMDP, and selected arguments.
- `run_<timestamp>.log`: full log output.
- `satprob_<timestamp>.{pdf,png}`: heatmap of the satisfaction probability per state.
- `policy_<timestamp>.{pdf,png}`: heatmap of the first control input selected per state.
- `traces_<timestamp>.{pdf,png}`: simulated trajectories over the model's `plot_dimensions` (1D models are drawn against the time step).
- `traces_3d_<timestamp>_view<i>.{pdf,png}`: 3D trajectories, for models defining `plot_dimensions_3d` (such as `Drone6D`).
- a trajectory GIF, for models defining `gif_dimensions` (`Pendulum`, `MountainCar`, `CartPole`).

# Repository structure

- `benchmarks/`: benchmark model definitions.
- `core/pipeline.py`: run pipeline shared by both entrypoints (setup, checkpointing, policy extraction, validation, plotting).
- `core/abstraction/`: partitioning, IMDP and two-layer IMDP construction, and solver implementations.
- `core/plotting/`: heatmaps and trajectory visualization.
- `core/validate/`: Monte Carlo simulation and validation utilities.
