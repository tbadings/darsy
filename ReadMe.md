# DARSY

DARSY is a tool for data-driven abstraction and policy synthesis for discrete-time stochastic systems with unknown dynamics. From sampled transitions of a system simulator, it builds an interval Markov decision process (IMDP) or a two-layer IMDP (2IMDP) abstraction whose probability intervals hold with a user-specified confidence, and synthesizes a robust policy for a reach-avoid, finite-horizon safety, or scLTL specification.

# Installation

Follow the steps below to install the tool:

## 1. Python environment

The project is tested with Python 3.13 (the experiments in the paper used Python 3.13.13). We recommend creating a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Next, install the base Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. JAX installation

JAX should be installed differently depending on whether you wish to run the code on CPU or GPU. The experiments in the paper used JAX 0.12.0.1 on CPU.

For CPU-only execution:

```bash
pip install jax
```

For CUDA-enabled execution on NVIDIA GPUs:

```bash
pip install "jax[cuda13]"
```

Apple Silicon should use the CPU version of JAX.

## 3. Spot (for scLTL specifications)

General scLTL specifications (`Dubins2D_scLTL`, `DoubleIntegrator_scLTL`, `Dubins3D_scLTL`) are checked and translated into a DFA with the `ltlfilt` and `ltl2tgba` command-line tools from [Spot](https://spot.lre.epita.fr). The Python bindings are not needed, but both tools must be on your `PATH`. Benchmarks with a plain reach-avoid (`(!unsafe) U goal`) or safety (`G(safe)`) specification do not use Spot.

Install Spot with one of the following options.

On macOS via Homebrew:

```bash
brew install spot
```

Inside a conda environment, via conda-forge:

```bash
conda install -c conda-forge spot
```

On Linux, use the Debian/Ubuntu packages or build from source, following the [Spot installation instructions](https://spot.lre.epita.fr/install.html). A source build that skips the Python bindings looks as follows:

```bash
tar xzf spot-<version>.tar.gz
cd spot-<version>
./configure --disable-python
make
sudo make install
```

If you built Storm from source with its shipped copy of Spot, you can instead add the bundled binaries to your `PATH`:

```bash
export PATH="<storm_dir>/build/resources/3rdparty/spot/bin:$PATH"
```

Verify the installation with:

```bash
ltl2tgba --version
ltlfilt --version
```

# Project entrypoints

There is one entrypoint per abstraction:

```bash
python RunIMDP.py --model <model_name>     # IMDP abstraction
python Run2IMDP.py --model <model_name>    # Two-layer IMDP abstraction
```

Both share the run pipeline in `core/pipeline.py` and the command-line options. The specification is defined by the benchmark and is one of:

- reach-avoid, `(!unsafe) U goal`, solved over an unbounded horizon unless the benchmark sets a horizon;
- finite-horizon safety, `G(safe)`, over the horizon set by the benchmark;
- a general scLTL formula, solved on the product of the abstraction with the formula's DFA (requires Spot).

`Run2IMDP.py` additionally builds a coarse layer controlled by `--coarse_factor`. Both entrypoints support robust value and policy iteration with `--solver jax` (the default). `--solver storm` applies only to `RunIMDP.py` with a reach-avoid specification.

Both entrypoints use a full `DensePartition` with grid edges aligned to labeled regions.
`Run2IMDP.py` groups those fine cells into a `CoarsePartition` for its second layer.

# Available benchmarks

The current benchmark registry (`benchmarks/__init__.py`) includes:

- `Temperature1D_reach`: single-room temperature with a binary heater (1D, linear, additive noise); reach-avoid.
- `Temperature1D_safety`: same system as `Temperature1D_reach`; safety `G(safe)` over 20 steps.
- `Temperature2D`, `Temperature3D`, `Temperature4D`: 2, 3, and 4 coupled rooms with one binary heater per room (linear, multiplicative noise); safety `G(safe)` over 15 steps.
- `DoubleIntegrator`: double integrator with position and velocity states and a scalar acceleration input (2D, linear); reach-avoid.
- `DoubleIntegrator_scLTL`: same system as `DoubleIntegrator`; scLTL `F (origin & F (left & F right))`.
- `MountainCar`: mountain car with position and velocity states (2D, nonlinear); reach-avoid (reach the top of the hill).
- `Pendulum`: pendulum swing-up with angle and angular velocity states and a torque input in `[-2, 2]` (2D, nonlinear); reach-avoid (reach the upright position).
- `PendulumHard`: as `Pendulum`, but with the torque limited to `[-1.2, 1.2]`.
- `Dubins2D_scLTL`: Dubins vehicle with position states and heading and speed inputs (2D, nonlinear); scLTL `((!unsafe) & (!wet | ((!goal) U carpet))) U (goal & (!unsafe))`.
- `Dubins3D`: unicycle with position and heading states and steering rate and velocity inputs (3D, nonlinear); reach-avoid.
- `Dubins3D_scLTL`: same system as `Dubins3D`, with the scLTL specification of `Dubins2D_scLTL`.
- `Test1D`: simple 1D test model.

All benchmarks use the noise distribution selected with `--noise_distr` (triangular by default), except `Temperature2D`, `Temperature3D`, and `Temperature4D`, which always use multiplicative Gaussian noise truncated to a ball.

Example runs:

`Dubins3D` and `Dubins3D_scLTL` use a 1 s step, steering coefficient 0.85, and
heading wrapped to `[-pi, pi)`. Their nine controls are the Cartesian product of
steering rates `{-pi/2, 0, pi/2}` and velocities `{-1, 0, 1}`. Their step function
adds noise only to the heading: triangular noise with halfwidth 0.05 by default, or
zero-mean Gaussian noise with standard deviation 0.05 with `--noise_distr gaussian`.

```bash
python RunIMDP.py --model Dubins3D --solver jax
python RunIMDP.py --model DoubleIntegrator --solver storm
python Run2IMDP.py --model MountainCar --solver jax --no-policy_iteration
```

# Running a benchmark

A typical command looks as follows:

```bash
python RunIMDP.py --model MountainCar --solver jax --noise_distr gaussian
```

This will:

1. create a timestamped output directory under `output/`,
2. construct the partition and the IMDP abstraction, estimating the probability intervals from `--num_samples` sampled transitions per state-action pair,
3. synthesize a robust policy for the benchmark's specification,
4. run Monte Carlo validation (1000 simulations of the closed-loop system),
5. save a `summary.csv` with the key results, logs, plots, and a `checkpoint.pkl` file (the last only with `--save_checkpoint`).

Steps 4 and 5 are best effort: a failure while simulating or plotting is logged (with its traceback in `run_<timestamp>.log`) and the remaining steps still run, so a plotting bug cannot discard a finished synthesis. A closing `N post-processing step(s) failed: ...` line lists whatever did not produce output.

## Reusing a checkpoint

To skip abstraction generation and rerun the solver and plotting pipeline from a checkpoint saved with `--save_checkpoint`:

```bash
python RunIMDP.py --model MountainCar --load_checkpoint output/<timestamp>_IMDP_MountainCar/checkpoint.pkl
```

When `--load_checkpoint` is provided, the saved model, partition, and abstraction are loaded directly from the specified pickle file. Checkpoints are specific to the entrypoint that wrote them.

# Reproducing the experiments

The main results table of the paper (Table 2) reports 13 benchmarks, each solved with an IMDP and a 2IMDP abstraction using 1 000, 10 000, and 100 000 sampled transitions per abstract state-action pair, with seed `0` and the default confidence parameter `--pac_beta 0.001`. The scripts in `experiments/` run these configurations and generate the table.

The benchmarks in the paper correspond to the following models:

| Paper benchmark | `--model` | Specification | Extra arguments |
| --- | --- | --- | --- |
| Temp1D | `Temperature1D_reach` | reach-avoid | `--noise_distr gaussian` |
| Temp1D | `Temperature1D_safety` | safety | `--noise_distr gaussian` |
| Temp2D | `Temperature2D` | safety | |
| Temp3D | `Temperature3D` | safety | |
| Temp4D | `Temperature4D` | safety | |
| DoubleInt | `DoubleIntegrator` | reach-avoid | |
| DoubleInt | `DoubleIntegrator_scLTL` | scLTL | |
| MountainCar | `MountainCar` | reach-avoid | |
| Pendulum-T | `Pendulum` | reach-avoid | |
| Pendulum-G | `PendulumHard` | reach-avoid | `--noise_distr gaussian` |
| Dubins2D | `Dubins2D_scLTL` | scLTL | |
| Dubins3D | `Dubins3D` | reach-avoid | |
| Dubins3D | `Dubins3D_scLTL` | scLTL | |

## 1. Run the experiments

To run all benchmarks with the settings from the paper, from the repository root:

```bash
bash experiments/run_all.sh
```

The scripts call `python`; to use another interpreter without activating the environment, set `PYTHON`, e.g. `PYTHON=.venv/bin/python bash experiments/run_all.sh`.

`run_all.sh` calls `experiments/run_benchmark.sh` once per benchmark, which runs both abstractions with all three numbers of samples and collects the six runs in `output/MultiRun_<timestamp>_<model>/`. To run a single benchmark, call it directly, with the extra arguments from the table above:

```bash
bash experiments/run_benchmark.sh Temperature1D_reach --seeds "0" --noise_distr gaussian
```

Passing several seeds, e.g. `--seeds "0 1 2"`, repeats every run for each seed.

The full sweep takes roughly 12 hours on the machine used for the paper (Apple MacBook Pro with an M5 chip and 24 GB of RAM). `run_all.sh` lists the expected runtime of each benchmark: most benchmarks take less than half an hour each, while `Temperature4D`, `MountainCar`, `Dubins3D`, and `Dubins3D_scLTL` take 2 to 4 hours each, dominated by the runs with 100 000 samples. Runtimes depend on the hardware. The sampled transitions are determined by `--seed` together with `--batch_size` and `--frs_batch_size`, so keep the batch sizes at their defaults to reproduce the paper's abstractions.

## 2. Generate the table

The table is generated from the `summary.csv` files of the runs:

```bash
python experiments/make_results_table.py output
```

This reads every `output/MultiRun_*/<run>/summary.csv` and writes two LaTeX tables:

- `output/table_main.tex`: number of states, actions, and transitions, abstraction and synthesis times, and the lower bound on the satisfaction probability.
- `output/table_main_extended.tex`: as above, plus the empirical satisfaction probability from the Monte Carlo simulations. This is Table 2 of the paper.

Per benchmark and number of samples, the best runtimes and lower bound are set in bold. Missing runs are shown as `--` (with a warning), and the script stops if the folder holds more than one run for the same benchmark, abstraction, and number of samples. If `output/` also contains earlier runs, first move the `MultiRun_*` folders to include into a separate folder and pass that folder instead:

```bash
mkdir results
mv output/MultiRun_* results/
python experiments/make_results_table.py results
```

The script only uses the Python standard library.

## Results from the paper

The runs reported in the paper, including their logs, `summary.csv` files, and plots, are stored in `output_HSCC/`. To regenerate the table from these runs without rerunning the experiments:

```bash
python experiments/make_results_table.py output_HSCC
```

# Solver backends

Two solver backends are supported:

- `--solver jax`: robust dynamic programming (value or policy iteration) implemented in JAX; supports both entrypoints and all specifications.
- `--solver storm`: robust value iteration through Storm (via `stormpy`); supports `RunIMDP.py` with reach-avoid specifications only.

All experiments in the paper were run with the JAX backend (`--solver jax`, the default) on the CPU.

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
- `--noise_distr {gaussian,normal,triangular}`: select the noise model (default: `triangular`). `normal` is accepted as an alias of `gaussian`.
- `--policy_iteration` or `--no-policy_iteration`: robust policy iteration (default) or value iteration.
- `--satprob`: stop improving the policy once the value in the initial state reaches this bound, and only evaluate it to convergence. Policy iteration only.

## Data-driven abstraction

- `--num_samples`: number of sampled transitions per state-action pair used to estimate the probability intervals (default: `5000`).
- `--frs_num_samples`: number of samples per state-action pair used to compute the forward reachable set, which determines the possible successor states (default: `100`).
- `--alpha`: the forward reachable set is the `[alpha/2, 100-alpha/2]` percentile envelope of the sampled next states (default: `0`, i.e. the full sampled range).
- `--pac_beta`: confidence parameter; all probability intervals of the abstraction hold jointly with probability at least `1 - pac_beta` (default: `0.001`).
- `--coarse_factor`: number of fine cells per coarse block dimension for `Run2IMDP.py` (default: `3`).

## Performance and memory tuning

- `--batch_size`: number of states processed together when computing transition probability intervals (default: `10`).
- `--frs_batch_size`: number of regions processed per batch when computing forward reachable sets (default: `100`).
- `--gpu` or `--no-gpu`: select the main JAX platform.
- `--gpu_rvi` or `--no-gpu_rvi`: select the device used for robust dynamic programming.

Larger batch sizes reduce the runtime but increase the memory usage, in particular for JIT compilation by JAX. If you run into memory limits, reduce `--batch_size` first. Note that the batch sizes also determine how the random samples are drawn, so changing them changes the abstraction.

## Reproducibility and logging

- `--seed`: random seed for NumPy and JAX (default: `0`).
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: logging verbosity.
- `--output_root`: root directory for timestamped run folders (default: `output`).
- `--save_checkpoint`: save the model, partition, and abstraction to `checkpoint.pkl`.
- `--load_checkpoint`: load a saved checkpoint instead of building the abstraction.

## Plotting controls

- `--plot_title` or `--no-plot_title`
- `--plot_grid` or `--no-plot_grid`
- `--plot_ticks` or `--no-plot_ticks`

# Output structure

Each run creates an output directory named like:

```text
output/2026-06-02_10-49-08_IMDP_MountainCar/
```

Runs started by `experiments/run_benchmark.sh` are grouped in `output/MultiRun_<timestamp>_<model>/`.

Depending on the benchmark and selected options, that directory may contain:

- `summary.csv`: settings and key results of the run, such as the abstraction and synthesis times (`time_abstraction`, `time_synthesis`), the abstraction size (`num_states`, `num_actions`, `num_intervals`), the bounds on the satisfaction probability in the initial state (`lower_bound`, `upper_bound`), and the empirical satisfaction probability (`empirical_satprob`).
- `checkpoint.pkl`: serialized model, partition, abstraction, and selected arguments.
- `run_<timestamp>.log`: full log output.
- `satprob_<timestamp>.{pdf,png}`: heatmap of the satisfaction probability per state (reach-avoid).
- `policy_<timestamp>.{pdf,png}`: heatmap of the first control input selected per state (reach-avoid).
- `scLTL_satprob_initial_<timestamp>.{pdf,png}` and `scLTL_policy_initial_<timestamp>.{pdf,png}`: the same heatmaps for scLTL specifications, for the DFA state reached after reading each cell's label from the initial DFA state.
- `product_synthesis.npz` and `scLTL_monte_carlo.npz`: product values, policy, and Monte Carlo outcomes (scLTL).
- `safety_lower_<timestamp>.{pdf,png}` and `safety_upper_<timestamp>.{pdf,png}`: heatmaps of the lower and upper bounds on the safety probability (safety).
- `safety_mc_outcomes_<timestamp>.{pdf,png}` and `safety_mc_trajectories_<timestamp>.{pdf,png}`: Monte Carlo outcomes and trajectories (safety).
- `safety_synthesis.npz` and `safety_monte_carlo.npz`: values, policy, and Monte Carlo outcomes (safety).
- `trajectories_<timestamp>.{pdf,png}`: simulated trajectories over the model's `plot_dimensions` (1D models are drawn against the time step).
- `trajectories_3d_*.{pdf,png}`: 3D trajectories, for models defining `plot_dimensions_3d`.
- `<model>_<timestamp>.gif`: a trajectory animation, for models defining `gif_dimensions` (`Pendulum`, `PendulumHard`, `MountainCar`).

# Repository structure

- `benchmarks/`: benchmark model definitions; `benchmarks/__init__.py` holds the registry used by `--model`, and `benchmarks/models.py` the shared dynamics.
- `core/options.py`: command-line options.
- `core/pipeline.py`: run pipeline shared by both entrypoints (setup, checkpointing, policy extraction, validation, plotting, run summary).
- `core/abstraction/`: partitioning, IMDP and two-layer IMDP construction (forward reachable sets and probability intervals), and solver implementations.
- `core/automaton/`: translation of scLTL formulas into DFAs with Spot.
- `core/plotting/`: heatmaps and trajectory visualization.
- `core/validate/`: Monte Carlo simulation and validation utilities.
- `experiments/`: scripts to run the experiments and generate the results table of the paper.
- `output_HSCC/`: the runs and tables reported in the paper.
