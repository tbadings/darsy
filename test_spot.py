import subprocess
import warnings
from pathlib import Path

# Spot executables
spot_dir = Path("/home/ibon/miniconda3/envs/spot-env/bin")

spot_executable = spot_dir / "ltl2tgba"
ltlfilt_executable = spot_dir / "ltlfilt"

# LTL specification
# formula = "(!b) U (a)"
formula = "F(a) & G(!b)"
#formula = "(((!unsafe) & (!wet | ((!goal) U carpet)))) U (goal & (!unsafe))"

# Check whether the specification is co-safe
check_result = subprocess.run(
    [
        str(ltlfilt_executable),
        "-f", formula,
        "--guarantee"
    ],
    capture_output=True,
    text=True,
    check=False
)

is_scltl = bool(check_result.stdout.strip())

if not is_scltl:
    warnings.warn(
        "The LTL specification is not co-safe (scLTL). "
        "Its satisfaction cannot generally be reduced to "
        "a reachability problem on the product MDP.",
        UserWarning
    )

# Generate the deterministic automaton
result = subprocess.run(
    [
        str(spot_executable),
        "-f", formula,
        "--deterministic",
        "--complete",
        "--unambiguous",
        "--generic"
    ],
    capture_output=True,
    text=True,
    check=True
)

print(result.stdout)