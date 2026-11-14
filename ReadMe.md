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

`Run2IMDP.py` additionally builds a coarse layer controlled by `--coarse_factor`. Both entrypoints use JAX robust value iteration for unbounded objectives and backward dynamic programming for finite horizons.

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
python RunIMDP.py --model Dubins3D
python RunIMDP.py --model DoubleIntegrator
python Run2IMDP.py --model MountainCar
```

# Running a benchmark

A typical command looks as follows:

```bash
python RunIMDP.py --model MountainCar --noise_distr gaussian
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

# Adding a new benchmark

Define the benchmark in `benchmarks/` and register its name in `benchmarks/__init__.py`; both entrypoints
then accept it through `--model`.

## 1. Choose or define the dynamics

Reuse a class from [benchmarks/models.py](benchmarks/models.py) when you only
want to change the initial state, partition, or specification. For a new system, add a
dynamics class there with the following interface:

| Attribute or method | Meaning |
| --- | --- |
| `n`, `p` | Number of state and control dimensions. |
| `step(state, action, noise)` | Return the next state with shape `(n,)`, given inputs of shapes `(n,)`, `(p,)`, and `(n,)`. Use JAX-compatible operations, such as `jax.numpy`, because the framework batches and JIT-compiles this method. |
| `noise` | A distribution providing `sample(size=..., rng=...)`, `sample_jax(key, shape=...)`, and metadata such as `noise['type']`. Samples have a trailing state dimension: requesting `(batch, steps)` produces `(batch, steps, n)`. |
| `lipschitz` | An `(n, n)` matrix bounding the componentwise change in the next state as the current state varies, for the same action and noise. The framework uses the supplied bound directly. |
| `wrap` | A Boolean array of length `n` marking periodic state dimensions. Implement the corresponding wrapping in `step`, as in the Dubins and pendulum models. |
| `state_variables` | One display name per state dimension. |

The existing Gaussian, triangular, and truncated Gaussian samplers are in
[benchmarks/dynamics/distributions.py](benchmarks/dynamics/distributions.py).
Keep random sampling in those samplers; `step` should use its supplied `noise`
argument. The type contracts are documented in
[core/interfaces.py](core/interfaces.py).

## 2. Define the benchmark settings

For example, create `benchmarks/MyBenchmark.py`. This example reuses the
one-dimensional integrator dynamics, with a smaller domain and a goal to reach
within ten transitions:

```python
import numpy as np

from benchmarks.models import Test1DDynamics


class MyBenchmark(Test1DDynamics):
    """Reach a target interval with a one-dimensional stochastic integrator."""

    def __init__(self, args):
        super().__init__(args)
        self.u_min = np.array([-1.0])
        self.u_max = np.array([1.0])
        self.num_actions = [3]  # Controls: -1, 0, 1.
        self.partition = {
            "boundary": np.array([[-5.0], [5.0]]),
            "number_per_dim": np.array([20]),
        }
        self.regions = {
            "r_goal": np.array([[[-0.5], [0.5]]]),
            "r_unsafe": np.empty((0, 2, self.n)),
        }
        self.goal = self.regions["r_goal"]
        self.critical = self.regions["r_unsafe"]
        self.atomic_propositions = ["goal", "unsafe"]
        self.labels = {"r_goal": {"goal"}, "r_unsafe": {"unsafe"}}
        self.specification = "(!unsafe) U goal"
        self.horizon = 10
        self.x0 = np.array([-3.0])
        self.plot_dimensions = [0]

    def labeling_function(self, region: str) -> set[str]:
        if region not in self.regions:
            raise ValueError(f"Unknown region: {region}")
        return self.labels[region]
```

The partition boundary has shape `(2, n)`: lower bounds followed by upper bounds.
`number_per_dim` sets the nominal resolution; grid edges are also aligned to
region boundaries. Each region is a union of boxes with shape `(number_of_boxes,
2, n)`. Use `np.empty((0, 2, n))` for an empty region. Choose an initial state
inside the partition and one or two state indices for `plot_dimensions`.
The control grid is the Cartesian product of evenly spaced values between
`u_min` and `u_max`, with `num_actions` values per control dimension.

To choose another task:

- **Reach-avoid:** use `(!unsafe) U goal`, expose `goal` and `critical` geometry,
  and define `r_goal`/`r_unsafe` with their corresponding proposition labels.
  Omit `horizon` for an unbounded task.
- **Safety:** use `G(safe)`, provide `safe_set` as an array of boxes, and set a
  nonnegative integer `horizon`. Include the safe boxes in `regions` with a
  `safe` label so the partition aligns to their boundaries. See
  [Temperature1D_safety.py](benchmarks/Temperature1D_safety.py).
- **General scLTL:** define the regions and proposition labels used by the
  formula, and omit `horizon`; the current pipeline solves these tasks over an
  unbounded horizon. Spot must be installed. See
  [Dubins2D_scLTL.py](benchmarks/Dubins2D_scLTL.py).

## 3. Register and run the benchmark

Add the import to `benchmarks/__init__.py`:

```python
from .MyBenchmark import MyBenchmark
```

Then add this entry to the existing `model_map` dictionary inside `create_model`:

```python
"MyBenchmark": MyBenchmark,
```

Run either abstraction from the repository root:

```bash
python RunIMDP.py --model MyBenchmark
python Run2IMDP.py --model MyBenchmark --coarse_factor 3
```

Start with a small partition while checking the model, then increase the
resolution and sample counts for your experiments. The example inherits
`--noise_distr` support from `Test1DDynamics`.

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

# Synthesis

Both entrypoints use JAX. Unbounded reach-avoid and scLTL objectives use robust value iteration;
finite-horizon reach-avoid and safety objectives use backward dynamic programming.
The solver returns a policy, its worst-case satisfaction probability, and its best-case probability.

# Important command-line options

## Model selection

- `--model`: benchmark name.
- `--noise_distr {gaussian,normal,triangular}`: select the noise model (default: `triangular`). `normal` is accepted as an alias of `gaussian`.

## Data-driven abstraction

- `--num_samples`: number of sampled transitions per state-action pair used to estimate the probability intervals (default: `5000`).
- `--frs_num_samples`: number of samples per state-action pair used to compute the forward reachable set, which determines the possible successor states (default: `100`).
- `--alpha`: total tail fraction in `[0, 1]`, split equally between both tails in each dimension. Before Lipschitz inflation, the sampled next-state envelope uses the `[100*alpha/2, 100*(1-alpha/2)]` percentiles. For example, `--alpha 0.1` selects the 5th and 95th percentiles. The default `0` uses the full sampled range; `1` uses the median.
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
