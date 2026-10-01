"""
Adapter from the MATLAB input format of
`robust_max_reach_probability_2layer` to the Python/JAX 2I-MDP solver.

Expected MATLAB top-level variables
-----------------------------------
user_param
initial_condition
solution
solver_opts

The MATLAB implementation uses 1-based state/action indices and MATLAB
cell arrays. This adapter converts those conventions to 0-based Python
indices and ordinary Python / NumPy containers.

Important MATLAB data-structure assumptions
--------------------------------------------
* solution.parents is a vector cell array. Each cell contains either
  zero elements or exactly one parent state ID.
* solution.children is a vector cell array. Each cell contains a
  variable-length vector of child state IDs (possibly empty).
* Coarse sets are disjoint and a fine state has at most one parent.

The adapter deliberately preserves the variable-length `children` lists.
They are padded later by the 2I-MDP/JAX preprocessing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_int_scalar(x: Any) -> int:
    """Convert a MATLAB scalar-like object to a Python int."""
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected a scalar, got shape {arr.shape}.")
    return int(arr.reshape(-1)[0])


def _as_float_scalar(x: Any) -> float:
    """Convert a MATLAB scalar-like object to a Python float."""
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected a scalar, got shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def _as_int_vector(x: Any) -> np.ndarray:
    """Convert a MATLAB numeric vector to a 1-D int32 array."""
    arr = np.asarray(x)
    if arr.size == 0:
        return np.empty(0, dtype=np.int32)
    return arr.reshape(-1).astype(np.int32)


def _as_float_vector(x: Any) -> np.ndarray:
    """Convert a MATLAB numeric vector to a 1-D float64 array."""
    arr = np.asarray(x)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    return arr.reshape(-1).astype(np.float64)


def _get_field(obj: Any, name: str) -> Any:
    """Read a field from a scipy MATLAB struct or dict-like object."""
    if isinstance(obj, dict):
        if name not in obj:
            raise KeyError(f"MATLAB struct is missing field '{name}'.")
        return obj[name]

    if hasattr(obj, name):
        return getattr(obj, name)

    raise KeyError(f"MATLAB struct is missing field '{name}'.")


# ---------------------------------------------------------------------------
# MATLAB cell-array helpers
# ---------------------------------------------------------------------------


def _cell_shape(cell: Any) -> tuple[int, ...]:
    """Return the shape of a MATLAB-style cell container."""
    if isinstance(cell, (list, tuple)):
        # A nested Python list may represent a 2-D cell array.  For the
        # vector-cell cases used for parents/children, len() is sufficient.
        return (len(cell),)

    arr = np.asarray(cell, dtype=object)
    return arr.shape


def _matlab_2d_cell_get(cell: Any, row: int, col: int) -> Any:
    """Get MATLAB cell(row+1,col+1) without flattening the array."""
    if isinstance(cell, (list, tuple)):
        value = cell[row]
        if isinstance(value, (list, tuple)):
            return value[col]
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            if col == 0:
                return arr.item()
            raise IndexError(
                f"Cannot index column {col} from scalar cell row {row}."
            )
        return arr[col]

    arr = np.asarray(cell, dtype=object)

    if arr.ndim == 0:
        if row == 0 and col == 0:
            return arr.item()
        raise IndexError("Scalar cannot be indexed as a 2-D cell array.")

    if arr.ndim == 1:
        # A 1-D cell container is only valid when the caller asks for its
        # first/only column.
        if col != 0:
            raise IndexError(
                f"1-D cell array has no column {col}."
            )
        return arr[row]

    return arr[row, col]


def _matlab_vector_cell_length(cell: Any) -> int:
    """
    Return the actual number of cells in a MATLAB vector cell array.

    MATLAB vectors may arrive from scipy as shape (1,N), (N,1), or as a
    Python list after `simplify_cells=True`. We use the total number of
    cells, not Ns+1, because the MATLAB parents/children arrays may omit
    a trailing special/absorbing state.
    """
    if isinstance(cell, (list, tuple)):
        return len(cell)

    arr = np.asarray(cell, dtype=object)
    if arr.ndim == 0:
        return 1
    return int(arr.size)


def _matlab_vector_cell_get(cell: Any, index: int) -> Any:
    """
    Return the `index`-th entry of a MATLAB vector cell array.

    For an N-by-1 or 1-by-N MATLAB vector, both layouts correspond to the
    same MATLAB linear cell order. For a genuine 1-D Python list, ordinary
    indexing is used.
    """
    n = _matlab_vector_cell_length(cell)
    if index < 0 or index >= n:
        raise IndexError(
            f"Cell index {index} is out of bounds for a cell array "
            f"with {n} entries."
        )

    if isinstance(cell, (list, tuple)):
        return cell[index]

    arr = np.asarray(cell, dtype=object)

    if arr.ndim == 0:
        return arr.item()

    # MATLAB linear indexing traverses column-major order.
    return arr.reshape(-1, order="F")[index]


def _convert_parent_cell(value: Any, sp: int) -> int | None:
    """
    Convert one parents{sp} cell.

    Each cell must contain either zero elements or exactly one state ID.
    MATLAB state IDs are converted from 1-based to 0-based indexing.
    """
    arr = np.asarray(value)

    if arr.size == 0:
        return None

    if arr.size != 1:
        raise ValueError(
            f"solution.parents[{sp}] contains {arr.size} elements; "
            "expected zero or one element."
        )

    return _as_int_scalar(arr) - 1


def _convert_children_cell(value: Any, qp: int) -> np.ndarray:
    """
    Convert one children{qp} cell to a 1-D zero-based int32 array.

    The number of children is allowed to vary from cell to cell.
    """
    return _as_int_vector(value) - 1


def _matlab_struct_to_simple(obj: Any) -> Any:
    """Recursively turn scipy MATLAB structs into Python containers."""
    if isinstance(obj, dict):
        return {k: _matlab_struct_to_simple(v) for k, v in obj.items()}

    if hasattr(obj, "_fieldnames"):
        return {
            name: _matlab_struct_to_simple(getattr(obj, name))
            for name in obj._fieldnames
        }

    if isinstance(obj, (list, tuple)):
        return type(obj)(_matlab_struct_to_simple(v) for v in obj)

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # Preserve the array shape. This is important for Sp/Qp/P arrays,
        # whose two dimensions are state x action.
        converted = [
            _matlab_struct_to_simple(v)
            for v in obj.reshape(-1, order="F")
        ]
        return np.asarray(converted, dtype=object).reshape(
            obj.shape,
            order="F",
        )

    return obj


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class UserParam:
    max_iter: int
    Ns: int
    Na: int
    S_target: np.ndarray
    S_not_terminal: np.ndarray


@dataclass
class InitialCondition:
    V_lower: np.ndarray
    V_upper: np.ndarray
    sigma: np.ndarray


@dataclass
class SolverOpts:
    tol_VI: float


@dataclass
class Solution2Layer:
    """
    Python representation of the MATLAB `solution` struct.

    State and action IDs are converted from MATLAB 1-based indexing to
    Python 0-based indexing.
    """

    A_id: list[np.ndarray]
    Sp: list[list[np.ndarray]]
    Qp: list[list[np.ndarray]]
    P_lower: list[list[np.ndarray]]
    P_upper: list[list[np.ndarray]]
    P_lower_coarse: list[list[np.ndarray]]
    P_upper_coarse: list[list[np.ndarray]]
    parents: list[int | None]
    children: list[np.ndarray]
    n_coarse: int


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_matlab_inputs(
    user_param_mat: Any,
    initial_condition_mat: Any,
    solution_mat: Any,
    solver_opts_mat: Any,
) -> tuple[UserParam, InitialCondition, Solution2Layer, SolverOpts]:
    """
    Convert the four MATLAB inputs into the Python solver representation.

    The MATLAB state/action IDs are assumed to be 1-based.
    """

    # =====================================================================
    # user_param
    # =====================================================================

    ns = _as_int_scalar(_get_field(user_param_mat, "Ns"))
    na = _as_int_scalar(_get_field(user_param_mat, "Na"))
    max_iter = _as_int_scalar(_get_field(user_param_mat, "max_iter"))

    s_target = _as_int_vector(
        _get_field(user_param_mat, "S_target")
    ) - 1

    s_not_terminal = _as_int_vector(
        _get_field(user_param_mat, "S_not_terminal")
    ) - 1

    user_param = UserParam(
        max_iter=max_iter,
        Ns=ns,
        Na=na,
        S_target=s_target,
        S_not_terminal=s_not_terminal,
    )

    # =====================================================================
    # initial_condition
    # =====================================================================

    v_lower = _as_float_vector(
        _get_field(initial_condition_mat, "V_lower")
    )

    v_upper = _as_float_vector(
        _get_field(initial_condition_mat, "V_upper")
    )

    sigma_raw = _as_int_vector(
        _get_field(initial_condition_mat, "sigma")
    )

    # MATLAB stores sigma only for the Ns ordinary states. The value
    # functions contain Ns+1 entries because the last entry is the
    # additional absorbing state. Convert valid MATLAB action IDs from
    # 1-based to 0-based indexing, then append -1 for the absorbing state.
    sigma_converted = np.where(
        sigma_raw > 0,
        sigma_raw - 1,
        sigma_raw,
    ).astype(np.int32)

    if len(sigma_converted) != ns:
        raise ValueError(
            f"sigma has length {len(sigma_converted)}, expected Ns={ns}."
        )

    sigma = np.full(ns + 1, -1, dtype=np.int32)
    sigma[:ns] = sigma_converted

    initial_condition = InitialCondition(
        V_lower=v_lower,
        V_upper=v_upper,
        sigma=sigma,
    )

    # =====================================================================
    # solver_opts
    # =====================================================================

    tol_vi = _as_float_scalar(
        _get_field(solver_opts_mat, "tol_VI")
    )

    solver_opts = SolverOpts(
        tol_VI=tol_vi,
    )

    # =====================================================================
    # solution
    # =====================================================================

    matlab_Sp = _get_field(solution_mat, "Sp")
    matlab_Qp = _get_field(solution_mat, "Qp")
    matlab_P_lower = _get_field(solution_mat, "P_lower")
    matlab_P_upper = _get_field(solution_mat, "P_upper")
    matlab_P_lower_coarse = _get_field(
        solution_mat,
        "P_lower_coarse",
    )
    matlab_P_upper_coarse = _get_field(
        solution_mat,
        "P_upper_coarse",
    )
    matlab_parents = _get_field(solution_mat, "parents")
    matlab_children = _get_field(solution_mat, "children")

    # -------------------------------------------------------------
    # A_id
    # -------------------------------------------------------------

    # The MATLAB function explicitly loops over a = 1:Na for every
    # nonterminal state. Construct the corresponding Python action IDs.
    A_id: list[np.ndarray] = [
        np.arange(na, dtype=np.int32)
        for _ in range(ns + 1)
    ]

    # -------------------------------------------------------------
    # Per-state/per-action transition data
    # -------------------------------------------------------------

    Sp: list[list[np.ndarray]] = [[] for _ in range(ns + 1)]
    Qp: list[list[np.ndarray]] = [[] for _ in range(ns + 1)]
    P_lower: list[list[np.ndarray]] = [[] for _ in range(ns + 1)]
    P_upper: list[list[np.ndarray]] = [[] for _ in range(ns + 1)]
    P_lower_coarse: list[list[np.ndarray]] = [[] for _ in range(ns + 1)]
    P_upper_coarse: list[list[np.ndarray]] = [[] for _ in range(ns + 1)]

    # MATLAB uses solution.Sp{s,a}, etc. These are 2-D state/action cell
    # arrays. Index the two dimensions explicitly; do NOT flatten them.
    for s in range(ns + 1):
        n_actions = na

        for a in range(n_actions):

            sp = _as_int_vector(
                _matlab_2d_cell_get(matlab_Sp, s, a)
            ) - 1

            qp = _as_int_vector(
                _matlab_2d_cell_get(matlab_Qp, s, a)
            ) - 1

            pl = _as_float_vector(
                _matlab_2d_cell_get(matlab_P_lower, s, a)
            )

            pu = _as_float_vector(
                _matlab_2d_cell_get(matlab_P_upper, s, a)
            )

            plc = _as_float_vector(
                _matlab_2d_cell_get(
                    matlab_P_lower_coarse,
                    s,
                    a,
                )
            )

            puc = _as_float_vector(
                _matlab_2d_cell_get(
                    matlab_P_upper_coarse,
                    s,
                    a,
                )
            )

            Sp[s].append(sp)
            Qp[s].append(qp)
            P_lower[s].append(pl)
            P_upper[s].append(pu)
            P_lower_coarse[s].append(plc)
            P_upper_coarse[s].append(puc)

    # -------------------------------------------------------------
    # parents
    # -------------------------------------------------------------

    # The MATLAB `parents` vector can contain fewer entries than Ns+1
    # (for example, because the absorbing state has no parent entry).
    # We preserve the actual entries and pad missing trailing entries with
    # None. Each real cell is required to contain 0 or 1 elements.
    parents: list[int | None] = [None] * (ns + 1)

    n_parent_entries = _matlab_vector_cell_length(matlab_parents)
    n_parent_to_copy = min(n_parent_entries, ns + 1)

    for sp in range(n_parent_to_copy):
        value = _matlab_vector_cell_get(
            matlab_parents,
            sp,
        )
        parents[sp] = _convert_parent_cell(
            value,
            sp,
        )

    # -------------------------------------------------------------
    # children
    # -------------------------------------------------------------

    # Each children cell corresponds to ONE coarse state. Therefore the
    # number of children entries is determined by the coarse partition,
    # not by Ns (the number of fine states).
    n_child_entries = _matlab_vector_cell_length(matlab_children)
    n_coarse = n_child_entries

    children: list[np.ndarray] = [
        np.empty(0, dtype=np.int32)
        for _ in range(n_coarse)
    ]

    for qp in range(n_coarse):
        value = _matlab_vector_cell_get(
            matlab_children,
            qp,
        )
        children[qp] = _convert_children_cell(
            value,
            qp,
        )

    solution = Solution2Layer(
        A_id=A_id,
        Sp=Sp,
        Qp=Qp,
        P_lower=P_lower,
        P_upper=P_upper,
        P_lower_coarse=P_lower_coarse,
        P_upper_coarse=P_upper_coarse,
        parents=parents,
        children=children,
        n_coarse=n_coarse,
    )

    _validate_converted_inputs(
        user_param,
        initial_condition,
        solution,
        solver_opts,
    )

    return (
        user_param,
        initial_condition,
        solution,
        solver_opts,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_converted_inputs(
    user_param: UserParam,
    initial_condition: InitialCondition,
    solution: Solution2Layer,
    solver_opts: SolverOpts,
) -> None:
    """Check structural assumptions needed by the Python solver."""

    expected_value_length = user_param.Ns + 1

    if len(initial_condition.V_lower) != expected_value_length:
        raise ValueError(
            "V_lower has length "
            f"{len(initial_condition.V_lower)}, expected "
            f"{expected_value_length}."
        )

    if len(initial_condition.V_upper) != expected_value_length:
        raise ValueError(
            "V_upper has length "
            f"{len(initial_condition.V_upper)}, expected "
            f"{expected_value_length}."
        )

    if len(initial_condition.sigma) != expected_value_length:
        raise ValueError(
            "sigma has length "
            f"{len(initial_condition.sigma)}, expected "
            f"{expected_value_length}."
        )

    # -------------------------------------------------------------
    # Transition-array consistency
    # -------------------------------------------------------------

    for s in range(user_param.Ns + 1):

        for a in range(user_param.Na):

            sp = solution.Sp[s][a]
            qp = solution.Qp[s][a]

            if np.any(sp < 0) or np.any(
                sp >= expected_value_length
            ):
                raise ValueError(
                    f"Invalid successor state ID at state {s}, "
                    f"action {a}: {sp}."
                )

            if np.any(qp < 0) or np.any(
                qp >= expected_value_length
            ):
                raise ValueError(
                    f"Invalid coarse state ID at state {s}, "
                    f"action {a}: {qp}."
                )

            if len(solution.P_lower[s][a]) != len(sp):
                raise ValueError(
                    f"P_lower length does not match Sp at state {s}, "
                    f"action {a}."
                )

            if len(solution.P_upper[s][a]) != len(sp):
                raise ValueError(
                    f"P_upper length does not match Sp at state {s}, "
                    f"action {a}."
                )

            if len(solution.Qp[s][a]) != len(
                solution.P_lower_coarse[s][a]
            ):
                raise ValueError(
                    f"P_lower_coarse length does not match Qp at "
                    f"state {s}, action {a}."
                )

            if len(solution.Qp[s][a]) != len(
                solution.P_upper_coarse[s][a]
            ):
                raise ValueError(
                    f"P_upper_coarse length does not match Qp at "
                    f"state {s}, action {a}."
                )

    # -------------------------------------------------------------
    # Parent consistency
    # -------------------------------------------------------------

    for sp, qp in enumerate(solution.parents):

        if qp is None:
            continue

        if qp < 0 or qp >= expected_value_length:
            raise ValueError(
                f"Invalid parent ID for state {sp}: {qp}."
            )

    # -------------------------------------------------------------
    # Children consistency
    # -------------------------------------------------------------

    for qp, child_array in enumerate(solution.children):

        if np.any(child_array < 0) or np.any(
            child_array >= expected_value_length
        ):
            raise ValueError(
                f"Invalid child state ID for parent {qp}: "
                f"{child_array}."
            )

    # -------------------------------------------------------------
    # Optional structural check: a state has at most one parent.
    # This is guaranteed by the representation, but we can also verify
    # parent/children consistency for states that are present in both.
    # -------------------------------------------------------------

    for qp, child_array in enumerate(solution.children):
        for sp in child_array:
            sp = int(sp)
            parent = solution.parents[sp]
            if parent is not None and parent != qp:
                raise ValueError(
                    f"State {sp} appears as a child of coarse state {qp}, "
                    f"but parents[{sp}] = {parent}."
                )

    if solver_opts.tol_VI <= 0:
        raise ValueError("tol_VI must be positive.")


# ---------------------------------------------------------------------------
# .mat-file loader
# ---------------------------------------------------------------------------


def load_matlab_2layer_inputs(
    mat_path: str | Path,
) -> tuple[UserParam, InitialCondition, Solution2Layer, SolverOpts]:
    """
    Load the MATLAB inputs from a .mat file and convert them.

    The .mat file must contain:

        user_param
        initial_condition
        solution
        solver_opts

    `simplify_cells=True` is used so MATLAB structs/cells become regular
    Python containers wherever possible.
    """

    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    raw = loadmat(
        mat_path,
        simplify_cells=True,
    )

    required = (
        "user_param",
        "initial_condition",
        "solution",
        "solver_opts",
    )

    missing = [
        name
        for name in required
        if name not in raw
    ]

    if missing:
        raise KeyError(
            "The .mat file is missing required variables: "
            + ", ".join(missing)
        )

    return convert_matlab_inputs(
        user_param_mat=_matlab_struct_to_simple(
            raw["user_param"]
        ),
        initial_condition_mat=_matlab_struct_to_simple(
            raw["initial_condition"]
        ),
        solution_mat=_matlab_struct_to_simple(
            raw["solution"]
        ),
        solver_opts_mat=_matlab_struct_to_simple(
            raw["solver_opts"]
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert MATLAB 2I-MDP solver input data to the "
            "Python representation."
        )
    )

    parser.add_argument(
        "mat_file",
        type=Path,
        help=(
            "MAT-file containing user_param, initial_condition, "
            "solution, solver_opts"
        ),
    )

    args = parser.parse_args()

    user_param, initial_condition, solution, solver_opts = (
        load_matlab_2layer_inputs(args.mat_file)
    )

    print("Successfully loaded MATLAB 2I-MDP inputs.")
    print(f"Ns = {user_param.Ns}")
    print(f"Na = {user_param.Na}")
    print(f"V_lower length = {len(initial_condition.V_lower)}")
    print(f"V_upper length = {len(initial_condition.V_upper)}")
    print(
        "Number of nonterminal states = "
        f"{len(user_param.S_not_terminal)}"
    )
    print(
        "Number of parent cells loaded = "
        f"{sum(parent is not None for parent in solution.parents)}"
    )
    print(
        "Number of children/coarse cells = "
        f"{solution.n_coarse}"
    )
    print(
        "Number of non-empty children cells = "
        f"{sum(len(c) > 0 for c in solution.children)}"
    )
