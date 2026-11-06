"""
Generate the main results table of the paper (tables/table_main.tex) from the experiment outputs, plus an
extended version (table_main_extended.tex) that also reports the empirical satisfaction probability.

Usage (from the repository root):  python experiments/make_results_table.py <folder>
e.g.:                               python experiments/make_results_table.py output_HSCC

The runs are read from <folder> and the tables are written to <folder>/table_main.tex and
<folder>/table_main_extended.tex. Each run folder
(<folder>/MultiRun_*/<timestamp>_<abstraction>_<model>/) contains a summary.csv and a run log, from which
the table columns are taken as follows:
    |S|        num_states in summary.csv (including the absorbing state)
    |A|        num_actions in summary.csv (for older runs: "(Number of actions: ...)" in the run log)
    #Trans.    num_intervals in summary.csv (number of probability intervals in the abstraction)
    Abs. [s]   time_abstraction in summary.csv, with two decimals
    Syn. [s]   time_synthesis in summary.csv, with two decimals
    lambda     lower_bound in summary.csv, rounded down so that the reported value is still a lower bound
    Emp.       empirical_satprob in summary.csv (fraction of Monte Carlo simulations that satisfy the
               specification), with three decimals; only in the extended table
Cells without a matching run are filled with '--'.
"""

import argparse
import csv
import re
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

# Table rows, in order: (model name in summary.csv, displayed name, specification)
BENCHMARKS = (
    ('Temperature1D_reach', 'Temp1D', 'reach-avoid'),
    ('Temperature1D_safety', 'Temp1D', 'safety'),
    ('Temperature2D', 'Temp2D', 'safety'),
    ('Temperature3D', 'Temp3D', 'safety'),
    ('Temperature4D', 'Temp4D', 'safety'),
    ('DoubleIntegrator', 'DoubleInt', 'reach-avoid'),
    ('DoubleIntegrator_scLTL', 'DoubleInt', 'scLTL'),
    ('MountainCar', 'MountainCar', 'reach-avoid'),
    ('Pendulum', 'Pendulum-T', 'reach-avoid'),
    ('PendulumHard', 'Pendulum-G', 'reach-avoid'),
    ('Dubins2D_scLTL', 'Dubins2D', 'scLTL'),
    ('Dubins3D', 'Dubins3D', 'reach-avoid'),
    ('Dubins3D_scLTL', 'Dubins3D', 'scLTL'),
)
METHODS = ('IMDP', '2IMDP')
NUM_SAMPLES = (1000, 10000, 100000)  # one column group per number of samples

MISSING = '--'

# Columns per sample-size group; the extended table adds the empirical satisfaction probability after lambda
GROUP_COLUMNS = (r'\#Trans.', r'Abs.\ [s]', r'Syn.\ [s]', r'$\lambda$')
EXTENDED_GROUP_COLUMNS = GROUP_COLUMNS + ('Emp.',)


def make_header(group_columns):
    r'''
    Table header, with one column group per number of samples.

    :param group_columns: Column titles within each group.
    :return: The LaTeX code from \begin{tabular} up to and including the \midrule below the column titles.
    '''

    width = len(group_columns)
    spec = 'llrr' + ''.join(' | ' + 'r' * width for _ in NUM_SAMPLES)
    groups = ' & '.join(f'\\multicolumn{{{width}}}{{c}}{{{fmt_int(n)[1:-1]} samples}}' for n in NUM_SAMPLES)
    rules = ' '.join(f'\\cmidrule(lr){{{5 + i * width}-{4 + (i + 1) * width}}}' for i in range(len(NUM_SAMPLES)))
    titles = ' & '.join(group_columns * len(NUM_SAMPLES))
    # The empty 4th cell of the top row is a \multicolumn so that, like the column groups, the top row has no
    # vertical line
    return (f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n'
            f'& & & \\multicolumn{{1}}{{r}}{{}} & {groups} \\\\\n'
            f'{rules}\n'
            f'\\textbf{{Benchmark}} & \\textbf{{Method}} & $|S|$ & $|A|$ & {titles} \\\\\n'
            '\\midrule\n')


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
        # Runs from before num_actions was added to summary.csv only report it in the log
        num_actions = summary.get('num_actions')
        summary['num_actions'] = (read_num_actions(summary_path.parent) if is_missing(num_actions)
                                  else int(num_actions))
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


def fmt_probability(value):
    '''Probability with three decimals, e.g. $0.994$.'''
    if is_missing(value):
        return MISSING
    return f'${float(value):.3f}$'


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


def make_row(runs, model, method, extended=False):
    '''
    Table row of one method on one benchmark.

    :param runs: All run summaries, as returned by load_runs.
    :param model: Model name in summary.csv.
    :param method: Abstraction (IMDP or 2IMDP).
    :param extended: If True, add the empirical satisfaction probability after the lower bound.
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
            if not extended:  # warn only once, for the regular table
                print(f'Warning: no run found for {description} with {n} samples')
            cells += [MISSING] * (len(EXTENDED_GROUP_COLUMNS) if extended else len(GROUP_COLUMNS))
            continue
        cells += [
            fmt_sci(run['num_intervals']),
            fmt_time(run['time_abstraction']),
            fmt_time(run['time_synthesis']),
            fmt_lower_bound(run['lower_bound']),
        ]
        if extended:
            cells.append(fmt_probability(run.get('empirical_satprob')))
    return ' & ' + ' & '.join(cells) + ' \\\\\n'


def make_table(runs, extended=False):
    '''
    Generate the full LaTeX table.

    :param runs: All run summaries, as returned by load_runs.
    :param extended: If True, add the empirical satisfaction probability after each lower bound.
    :return: The LaTeX code of the table.
    '''

    blocks = []
    for model, name, specification in BENCHMARKS:
        block = (f'\\multirow{{{len(METHODS)}}}{{*}}{{\\begin{{tabular}}{{@{{}}l@{{}}}}\\textbf{{{name}}}'
                 f'\\\\\\textit{{({specification})}}\\end{{tabular}}}}\n')
        block += ''.join(make_row(runs, model, method, extended) for method in METHODS)
        blocks.append(block)
    header = make_header(EXTENDED_GROUP_COLUMNS if extended else GROUP_COLUMNS)
    return header + '\\midrule\n'.join(blocks) + FOOTER


def main():
    parser = argparse.ArgumentParser(description='Generate the main results table of the paper.')
    parser.add_argument('folder', type=Path,
                        help='Folder with one MultiRun_* subfolder per benchmark (e.g., output_HSCC); '
                             'the tables are written to <folder>/table_main.tex and '
                             '<folder>/table_main_extended.tex')
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise FileNotFoundError(f'Folder {args.folder} does not exist')
    runs = load_runs(args.folder)
    if not runs:
        raise FileNotFoundError(f'No summary.csv files found in {args.folder}/*/*/')

    unused = sorted({model for model, _, _ in runs} - {model for model, *_ in BENCHMARKS})
    if unused:
        print(f'Note: runs of {", ".join(unused)} are not part of the table')

    for filename, extended in (('table_main.tex', False), ('table_main_extended.tex', True)):
        output = args.folder / filename
        output.write_text(make_table(runs, extended), encoding='utf-8')
        print(f'Table written to {output}')


if __name__ == '__main__':
    main()
