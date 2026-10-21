import numpy as np
from types import SimpleNamespace
from robust_vi_2layer_dfa import robust_scLTL_synthesis_2layer_jax, initial_product_state
# One nonterminal state 0, absorbing goal state 1, deterministic transition 0->1.
dfa = SimpleNamespace(transitions=np.array([[0,1],[1,1]],dtype=np.int32),accepting_states=np.array([False,True]),atomic_propositions=['goal'],initial_state=0)
prep=dict(states=np.array([0],np.int32),action_mask=np.array([[True]]),successors=np.array([[[1]]],np.int32),successors_mask=np.array([[[True]]]),prob_lb=np.array([[[1.]]],np.float32),prob_ub=np.array([[[1.]]],np.float32),coarse_constraint_idx=np.array([[[-1]]],np.int32),coarse_mask=np.array([[[False]]]),prob_lb_coarse=np.array([[[0.]]],np.float32),prob_ub_coarse=np.array([[[0.]]],np.float32),children_indices=np.array([[[[-1]]]],np.int32),children_mask=np.array([[[[False]]]]))
vl,vu,sigma=robust_scLTL_synthesis_2layer_jax(prep,dfa,[0,1],max_iter=5)
print('lower',vl,'upper',vu,'policy',sigma)
assert vl[0,0] > .99 and vu[0,0] > .99
assert initial_product_state(0,dfa,[0,1]) == (0,0)