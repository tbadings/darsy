"""
Generate the main results table of the paper (tables/table_main.tex) from the experiment outputs.

Usage (from anywhere):  python experiments/make_results_table.py [--results_dir DIR] [--output FILE]

By default, the runs are read from output_HSCC/ and the table is written to tables/table_main.tex, both
relative to the repository root (the tables/ folder is created if needed). Each run folder (<results_dir>/MultiRun_*/<timestamp>_<abstraction>_<model>/) contains a
summary.csv and a run log, from which the table columns are taken as follows:
    |S|        num_states in summary.csv (including the absorbing state)
    |A|        "(Number of actions: ...)" in the run log
    #Trans.    num_intervals in summary.csv (number of probability intervals in the abstraction)
    Abs. [s]   time_abstraction in summary.csv, with two decimals
    Syn. [s]   time_synthesis in summary.csv, with two decimals
    lambda     lower_bound in summary.csv, rounded down so that the reported value is still a lower bound
Cells without a matching run are filled with '--'.
"""

import argparse
import csv
import re
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Table rows, in order: (model name in summary.csv, displayed name, specification)
BENCHMARKS = (
    ('Temperature1D_reach', 'Temp1D', 'reach-avoid'),
    ('Temperature1D_safety', 'Temp1D', 'safety'),
    ('Temperature2D', 'Temp2D', 'safety'),
    ('Temperature3D', 'Temp3D', 'safety'),
    ('Temperature4D', 'Temp4D', 'safety'),
    ('DoubleIntegrator', 'DoubleInt', 'reach-avoid'),
    ('MountainCar', 'MountainCar', 'reach-avoid'),
    ('Pendulum', 'Pendulum', 'reach-avoid'),
    ('Dubins2D_scLTL', 'Dubins2D', 'scLTL'),
    ('Dubins3D', 'Dubins3D', 'reach-avoid'),
    ('Dubins3D_scLTL', 'Dubins3D', 'scLTL'),
)
METHODS = ('IMDP', '2IMDP')
NUM_SAMPLES = (1000, 10000, 100000)  # one column group per number of samples

MISSING = '--'

HEADER = r'''\begin{tabular}{llrr rrrr rrrr rrrr}
\toprule
\multirow{2}{*}{\textbf{Benchmark}} & \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{$|S|$} & \multirow{2}{*}{$|A|$} & \multicolumn{4}{c}{1k samples} & \multicolumn{4}{c}{10k samples} & \multicolumn{4}{c}{100k samples} \\
\cmidrule(lr){5-8} \cmidrule(lr){9-12} \cmidrule(lr){13-16}
& & & & \#Trans. & Abs.\ [s] & Syn.\ [s] & $\lambda$ & \#Trans. & Abs.\ [s] & Syn.\ [s] & $\lambda$ & \#Trans. & Abs.\ [s] & Syn.\ [s] & $\lambda$ \\
\midrule
'''
FOOTER = '\\bottomrule\n\\end{tabular}%\n'

NUM_ACTIONS_PATTERN = re.compile(r'\(Number of actions: (\d+)\)')


def read_num_actions(run_dir):
    '''
    Read the number of actions from the log of a run.

    :param run_dir: Folder of the run.
    :return: Number of actions, or None if no log reports it.
    '''

    for log_path in sorted(run_dir.glob('run_*.log')):
        match = NUM_ACTIONS_PATTERN.search(log_path.read_text(encoding='utf-8'))
        if match:
            return int(match.group(1))
    return None


def load_runs(results_dir):
    '''
    Load the summaries of all runs in the results folder.

    :param results_dir: Folder with one MultiRun_* subfolder per benchmark.
    :return: Dict from (model, abstraction, num_samples) to the run summary (dict from metric to value).
    '''

    runs = {}
    for summary_path in sorted(results_dir.glob('*/*/summary.csv')):
        with summary_path.open(newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # header row: metric,value
            summary = dict(reader)
        summary['num_actions'] = read_num_actions(summary_path.parent)
        summary['run_dir'] = summary_path.parent

        key = (summary['model'], summary['abstraction'], int(summary['num_samples']))
        if key in runs:
            raise ValueError(f'Multiple runs for {key}:\n  {runs[key]["run_dir"]}\n  {summary_path.parent}')
        runs[key] = summary
    return runs


def is_missing(value):
    return value is None or value == ''


def fmt_int(value):
    '''Integer with thin spaces as thousands separators, e.g. $250\\,501$.'''
    return MISSING if is_missing(value) else f'${int(value):,}$'.replace(',', r'\,')


def fmt_sci(value):
    '''Integer in scientific notation with three significant digits, e.g. $4.36\\cdot 10^{8}$.'''
    if is_missing(value):
        return MISSING
    mantissa, exponent = f'{int(value):.2e}'.split('e')
    return f'${mantissa}\\cdot 10^{{{int(exponent)}}}$'


def fmt_time(value):
    '''Runtime in seconds with two decimals, e.g. $0.26$, $18.73$, $5547.59$.'''
    if is_missing(value):
        return MISSING
    return f'${float(value):.2f}$'


def fmt_lower_bound(value):
    '''Lower bound rounded down to three decimals, so that it remains a valid lower bound.'''
    if is_missing(value):
        return MISSING
    return f'${Decimal(value).quantize(Decimal("0.001"), rounding=ROUND_FLOOR)}$'


def shared_value(runs, metric, description):
    '''
    Value of a metric that must be the same for all given runs (e.g., the number of states).

    :param runs: Run summaries (None for missing runs).
    :param metric: Key of the metric in the run summaries.
    :param description: Description of the runs, used in the error message.
    :return: The shared value, or None if no run reports it.
    '''

    values = {run[metric] for run in runs if run is not None and not is_missing(run[metric])}
    if len(values) > 1:
        raise ValueError(f'Inconsistent {metric} for {description}: {sorted(values)}')
    return values.pop() if values else None


def make_row(runs, model, method):
    '''
    Table row of one method on one benchmark.

    :param runs: All run summaries, as returned by load_runs.
    :param model: Model name in summary.csv.
    :param method: Abstraction (IMDP or 2IMDP).
    :return: The LaTeX row, without the benchmark column.
    '''

    method_runs = [runs.get((model, method, n)) for n in NUM_SAMPLES]
    description = f'{model} ({method})'
    cells = [
        method,
        fmt_int(shared_value(method_runs, 'num_states', description)),
        fmt_int(shared_value(method_runs, 'num_actions', description)),
    ]
    for n, run in zip(NUM_SAMPLES, method_runs):
        if run is None:
            print(f'Warning: no run found for {description} with {n} samples')
            cells += [MISSING] * 4
            continue
        cells += [
            fmt_sci(run['num_intervals']),
            fmt_time(run['time_abstraction']),
            fmt_time(run['time_synthesis']),
            fmt_lower_bound(run['lower_bound']),
        ]
    return ' & ' + ' & '.join(cells) + ' \\\\\n'


def make_table(runs):
    '''
    Generate the full LaTeX table.

    :param runs: All run summaries, as returned by load_runs.
    :return: The LaTeX code of the table.
    '''

    blocks = []
    for model, name, specification in BENCHMARKS:
        block = (f'\\multirow{{{len(METHODS)}}}{{*}}{{\\begin{{tabular}}{{@{{}}l@{{}}}}\\textbf{{{name}}}'
                 f'\\\\{specification}\\end{{tabular}}}}\n')
        block += ''.join(make_row(runs, model, method) for method in METHODS)
        blocks.append(block)
    return HEADER + '\\midrule\n'.join(blocks) + FOOTER


def main():
    parser = argparse.ArgumentParser(description='Generate the main results table of the paper.')
    parser.add_argument('--results_dir', type=Path, default=REPO_ROOT / 'output_HSCC',
                        help='Folder with one MultiRun_* subfolder per benchmark (default: output_HSCC)')
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'tables' / 'table_main.tex',
                        help='LaTeX file to write the table to (default: tables/table_main.tex)')
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise FileNotFoundError(f'No summary.csv files found in {args.results_dir}/*/*/')

    unused = sorted({model for model, _, _ in runs} - {model for model, *_ in BENCHMARKS})
    if unused:
        print(f'Note: runs of {", ".join(unused)} are not part of the table')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_table(runs), encoding='utf-8')
    print(f'Table written to {args.output}')


if __name__ == '__main__':
    main()
