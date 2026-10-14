"""Small checks of label encoding and product lifting (no Spot executable needed)."""
import importlib.util
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('solver', __file__.replace('test_integration.py', 'robust_vi_2layer_dfa.py'))
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

imdp = SimpleNamespace(
    partition=SimpleNamespace(size=2), absorbing_state=2,
    atomic_propositions=['goal', 'wet'],
    states_of_interest={'r_goal': np.array([False, True]), 'r_wet': np.array([True, False])},
    partial_region_cells={'r_goal': np.zeros(2, bool), 'r_wet': np.zeros(2, bool)},
    region_labels={'r_goal': {'goal'}, 'r_wet': {'wet'}}
)
dfa = SimpleNamespace(
    atomic_propositions=['wet', 'goal'],
    transitions=np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int32),
    accepting_states=np.array([False, True]), initial_state=0
)
labels = solver.encode_imdp_labels(imdp, dfa)
assert labels.tolist() == [1, 2, 0], labels
prep = dict(
    states=np.array([0, 1], dtype=np.int32),
    successors=np.array([[[1, 2]], [[0, 2]]], dtype=np.int32),
    successors_mask=np.ones((2, 1, 2), bool),
    action_mask=np.ones((2, 1), bool),
    prob_lb=np.array([[[0.8, 0.2]], [[0.8, 0.2]]], dtype=np.float32),
    prob_ub=np.array([[[0.8, 0.2]], [[0.8, 0.2]]], dtype=np.float32),
    coarse_constraint_idx=np.full((2, 1, 2), -1, dtype=np.int32),
    coarse_mask=np.zeros((2, 1, 0), bool),
    prob_lb_coarse=np.zeros((2, 1, 0), np.float32),
    prob_ub_coarse=np.zeros((2, 1, 0), np.float32),
    children_indices=np.zeros((2, 1, 0, 0), np.int32),
    children_mask=np.zeros((2, 1, 0, 0), bool),
)
prod = solver.preprocess_2layer_dfa_product(prep, dfa, labels)
assert prod['states'].tolist() == [0, 2]
assert prod['successors'][0, 0].tolist() == [3, 4]
assert np.array_equal(prod['prob_lb'][0], prep['prob_lb'][0])
print('PASS: DFA AP ordering, product indexing, terminal exclusion, probability bounds')