import argparse
from collections.abc import Sequence

import numpy as np


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError('Expected a value in [0, 1].')
    return parsed


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse run options, normalize the noise alias, and select float64 precision."""
    # Options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random number generators (Jax, Numpy)")
    parser.add_argument('--model', type=str, default='',
                        help="Benchmark model to run")
    parser.add_argument('--noise_distr', type=str, default='triangular', choices=['gaussian', 'normal', 'triangular'], # 'normal' is alias for 'gaussian'
                        help="Noise distribution type to use ('normal' is treated as 'gaussian')")
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run on GPU. Otherwise, run on CPU")
    parser.add_argument('--gpu_rvi', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run RVI on GPU. Otherwise, run on CPU")
    parser.add_argument('--batch_size', type=_positive_int, default=10,
                        help="For computing the transition probability intervals, the number of states to process in a vectorized fashion (Warning: increasing this too much drastically increases memory usage for JIT compilation by JAX!)")
    parser.add_argument('--frs_batch_size', type=_positive_int, default=100,
                        help="Number of state regions to process per batch when computing forward reachable sets. Larger values reduce Python-JAX round trips but increase peak memory usage.")
    parser.add_argument('--num_samples', type=_positive_int, default=5000,
                        help="Number of samples per (cell, action) pair for data-driven forward reachable set computation.")
    parser.add_argument('--frs_num_samples', type=_positive_int, default=100,
                        help="Number of samples per state-action for the forward reachable set.")
    parser.add_argument('--alpha', type=_probability, default=0.00,
                        help="Total tail fraction in [0, 1], split equally between both tails in each "
                             "dimension. Uses the [100*alpha/2, 100*(1-alpha/2)] percentiles of sampled "
                             "next states before Lipschitz inflation. For example, 0.1 selects the "
                             "5th and 95th percentiles and 0 uses the full sampled range.")
    parser.add_argument('--pac_beta', type=_probability, default=0.001,
                        help="Confidence level for Clopper-Pearson PAC bounds on transition "
                             "probability intervals. Smaller values give wider, more conservative intervals.")
    parser.add_argument('--coarse_factor', type=_positive_int, default=3,
                        help="Factor for blocks in 2I-MDP.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level')
    parser.add_argument('--output_root', type=str, default='output',
                        help='Base directory where per-run output folders are created')
    parser.add_argument('--load_checkpoint', type=str, default=None, metavar='PATH',
                        help='Path to a checkpoint.pkl file saved by a previous run. '
                             'When set, the IMDP abstraction generation is skipped and '
                             'model, partition, and IMDP are loaded from the checkpoint.')
    parser.add_argument('--save_checkpoint', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, save checkpoints during execution")

    # Plotting options
    parser.add_argument('--plot_grid', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot unit grids in figures")
    parser.add_argument('--plot_title', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot titles in figures")
    parser.add_argument('--plot_ticks', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, plot ticks in figures")

    # Parse arguments
    args = parser.parse_args(argv)

    # Dtype used for all float storage and computation
    args.floatprecision = np.float64

    # Canonicalize alias.
    if args.noise_distr == 'normal':
        args.noise_distr = 'gaussian'

    return args
