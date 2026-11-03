#!/usr/bin/env bash
# Run one benchmark for every combination of seed, number of samples per state-action pair,
# and abstraction. Extra args are passed to every run.
#
# Usage (from anywhere):  bash experiments/run_benchmark.sh <Model> --seeds "0 1 2" [extra args...]
# Set PYTHON to use another interpreter, e.g. PYTHON=.venv/bin/python

NUM_SAMPLES="1000 10000 100000"
ABSTRACTIONS="IMDP 2IMDP"
PYTHON="${PYTHON:-python}"

if [ "$2" != "--seeds" ] || [ -z "$3" ]; then
    echo "Usage: $0 <Model> --seeds \"0 1 2\" [extra args...]" >&2
    exit 2
fi
MODEL="$1"
SEEDS="$3"
shift 3

# Run from the repository root, so the entrypoints and output/ resolve correctly
cd "$(dirname "$0")/.."

# All runs of this sweep go into one folder, named after the start time and the model
OUT="output/MultiRun_$(date +%Y-%m-%d_%H-%M-%S)_${MODEL}"

failed=""
for abstraction in $ABSTRACTIONS; do
    for n in $NUM_SAMPLES; do
        for seed in $SEEDS; do
            name="${abstraction}_N${n}_seed${seed}"
            echo "+++++ ${MODEL}: ${name} +++++"
            "$PYTHON" "Run${abstraction}.py" --model "$MODEL" --seed "$seed" --num_samples "$n" \
                --output_root "$OUT" "$@" || failed="${failed} ${name}"
        done
    done
done

if [ -n "$failed" ]; then
    echo "${MODEL}: failed runs:${failed}"
    exit 1
fi
echo "${MODEL}: all runs finished."
