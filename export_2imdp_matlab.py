"""Export a 2I-MDP abstraction to MATLAB robust VI input structures."""
from pathlib import Path
import numpy as np
from scipy.io import savemat


def save_2imdp_matlab(abstraction, filename, max_iter=10000, tol_VI=1e-6):
    """Save MATLAB structs user_param, initial_condition, solution, solver_opts.

    Raises if a fine successor belongs to a coarse parent missing from Qp,
    because the supplied MATLAB implementation assumes that parent is in Qp.
    """
    n = len(abstraction.states)
    sink = int(abstraction.absorbing_state)
    if sink != n:
        raise ValueError('Expected absorbing state ID to equal number of fine states.')
    na = int(max((len(abstraction.A_id[s]) for s in range(n)), default=0))
    goal = np.asarray(abstraction.goal_regions, dtype=bool)
    critical = np.asarray(abstraction.critical_regions, dtype=bool)
    if goal.shape != (n,) or critical.shape != (n,):
        raise ValueError('Goal/critical masks must have one entry per fine state.')
    enabled = [s for s in range(n) if not (goal[s] or critical[s])]
    for s in enabled:
        if len(abstraction.A_id[s]) != na:
            raise ValueError('MATLAB solver assumes same number of actions for every nonterminal state.')

    parents = np.asarray(abstraction.coarse_partition.parents if hasattr(abstraction, 'coarse_partition') else abstraction.parents, dtype=int)
    raw_children = abstraction.coarse_partition.children if hasattr(abstraction, 'coarse_partition') else abstraction.children
    children = [np.asarray(c, dtype=int) for c in raw_children]
    if parents.shape != (n,):
        raise ValueError('parents must have one coarse ID per fine state.')
    if not np.array_equal(np.sort(np.concatenate(children)), np.arange(n)):
        raise ValueError('children must partition all fine states exactly once.')
    sp = np.empty((n + 1, na), dtype=object)
    qp = np.empty((n + 1, na), dtype=object)
    pl = np.empty((n + 1, na), dtype=object)
    pu = np.empty((n + 1, na), dtype=object)
    pcl = np.empty((n + 1, na), dtype=object)
    pcu = np.empty((n + 1, na), dtype=object)
    for array in (sp, qp, pl, pu, pcl, pcu):
        for idx in np.ndindex(array.shape):
            array[idx] = np.empty((0, 1), dtype=np.float64)

    for s in enabled:
        for local_a in range(na):
            ids = np.asarray(abstraction.S_id[s][local_a], dtype=int)
            probs = np.asarray(abstraction.P_full[s][local_a], dtype=float)
            if probs.shape != (len(ids), 2):
                raise ValueError(f'Fine interval shape mismatch at {(s, local_a)}')
            valid = (ids >= 0) & (ids < n) & (probs[:, 1] > 0)
            ids, probs = ids[valid], probs[valid]
            sink_prob = np.asarray(abstraction.P_absorbing[s][local_a], dtype=float).reshape(1, 2)
            ids = np.append(ids, sink)
            probs = np.vstack((probs, sink_prob))
            qids = np.asarray(abstraction.Q_id[s][local_a], dtype=int)
            qprobs = np.asarray(abstraction.P_coarse[s][local_a], dtype=float)
            if qprobs.shape != (len(qids), 2):
                raise ValueError(f'Coarse interval shape mismatch at {(s, local_a)}')
            qvalid = (qids >= 0) & (qids < len(children)) & (qprobs[:, 1] > 0)
            qids, qprobs = qids[qvalid], qprobs[qvalid]
            missing = set(parents[ids[:-1]].tolist()) - set(qids.tolist())
            if missing:
                raise ValueError(
                    f'At state {s}, action {local_a}, coarse parents {sorted(missing)} '
                    'are absent from Qp. The supplied MATLAB solver uses find(Qp==parent) '
                    'without checking for empty results. Modify that MATLAB code to skip '
                    'missing coarse constraints before exporting.'
                )
            sp[s, local_a] = (ids + 1).astype(np.int32).reshape(-1, 1)
            qp[s, local_a] = (qids + 1).astype(np.int32).reshape(-1, 1)
            pl[s, local_a] = probs[:, 0].reshape(-1, 1)
            pu[s, local_a] = probs[:, 1].reshape(-1, 1)
            pcl[s, local_a] = qprobs[:, 0].reshape(-1, 1)
            pcu[s, local_a] = qprobs[:, 1].reshape(-1, 1)

    matlab_parents = np.empty((n + 1, 1), dtype=object)
    for s in range(n):
        matlab_parents[s, 0] = np.array([[parents[s] + 1]], dtype=np.int32)
    matlab_parents[n, 0] = np.empty((0, 0))  # sink has no coarse parent
    matlab_children = np.empty((len(children), 1), dtype=object)
    for q, kids in enumerate(children):
        matlab_children[q, 0] = (kids + 1).astype(np.int32).reshape(-1, 1)

    vl = np.zeros((n + 1, 1))
    vl[np.flatnonzero(goal)] = 1
    initial = {'V_lower': vl, 'V_upper': vl.copy(), 'sigma': np.ones((n + 1, 1), dtype=np.int32)}
    user_param = {
        'max_iter': int(max_iter), 'Ns': n, 'Na': na,
        'S_target': (np.flatnonzero(goal) + 1).astype(np.int32).reshape(1, -1),
        'S_not_terminal': (np.asarray(enabled, dtype=np.int32) + 1).reshape(1, -1),
    }
    solution = {
        'Sp': sp, 'Qp': qp, 'P_lower': pl, 'P_upper': pu,
        'P_lower_coarse': pcl, 'P_upper_coarse': pcu,
        'parents': matlab_parents, 'children': matlab_children,
    }
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {'user_param': user_param, 'initial_condition': initial,
                   'solution': solution, 'solver_opts': {'tol_VI': float(tol_VI)}},
            do_compression=True)
    return path
