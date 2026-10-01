
# %%

from matlab_2layer_adapter import load_matlab_2layer_inputs
from robust_vi_2layer import robust_max_reach_probability_2layer_jax
from robust_vi_2layer import save_rvi_results_matlab

user_param, initial_condition, solution, solver_opts = \
    load_matlab_2layer_inputs("my_matlab_input.mat")


P_reach_lower, P_reach_upper, sigma = (
    robust_max_reach_probability_2layer_jax(
        user_param,
        initial_condition,
        solution,
        solver_opts,
    )
)

save_rvi_results_matlab(
    "rvi_results.txt",
    P_reach_lower,
    P_reach_upper,
    sigma,
)

print("Done!")