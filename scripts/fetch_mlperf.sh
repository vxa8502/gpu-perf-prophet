#!/usr/bin/env bash
# Sparse-clone MLPerf Inference results repos for GPU Perf Prophet.
#
# Fetches only the files the parser needs from each round:
#   closed/*/systems/*.json              (GPU hardware descriptions)
#   closed/*/results/**/mlperf_log_summary.txt  (performance logs)
#
# Repos are placed under data/raw/mlperf/<version>/.
# Full clones of these repos can exceed several GB; sparse checkout
# keeps each round under ~50 MB.
#
# Usage:
#   bash scripts/fetch_mlperf.sh              # all four rounds
#   bash scripts/fetch_mlperf.sh v6.0         # single round
#   ROUNDS="v5.1 v6.0" bash scripts/fetch_mlperf.sh

set -euo pipefail

DEST="data/raw/mlperf"
ORG="mlcommons"
ROUNDS="${ROUNDS:-v4.1 v5.0 v5.1 v6.0}"

# If a round argument is passed on the command line, use that instead.
if [[ $# -ge 1 ]]; then
    ROUNDS="$*"
fi

mkdir -p "$DEST"

for ROUND in $ROUNDS; do
    # Validate before using $ROUND in path or URL construction.
    # Accepts only the form vN.N (e.g. v6.0) to prevent path traversal.
    if ! [[ "$ROUND" =~ ^v[0-9]+\.[0-9]+$ ]]; then
        echo "Error: invalid round tag '${ROUND}' — expected vN.N (e.g. v6.0)" >&2
        exit 1
    fi

    REPO="${ORG}/inference_results_${ROUND}"
    TARGET="${DEST}/${ROUND}"

    if [[ -d "$TARGET/.git" ]]; then
        echo "→ ${ROUND}: already cloned at ${TARGET}, skipping."
        continue
    fi

    echo "→ Cloning ${REPO} (sparse) into ${TARGET} …"

    git clone \
        --filter=blob:none \
        --sparse \
        --depth=1 \
        --no-checkout \
        "https://github.com/${REPO}.git" \
        "$TARGET"

    pushd "$TARGET" > /dev/null

    # Fetch only the paths the parser needs.
    # --no-cone: use gitignore-style patterns so * wildcards are accepted.
    # (default cone mode rejects wildcards with "specify directories rather
    # than patterns")
    git sparse-checkout set --no-cone \
        "closed/*/systems" \
        "closed/*/results/*/llama2-70b/*/performance" \
        "closed/*/results/*/llama2-70b-99/*/performance" \
        "closed/*/results/*/llama2-70b-99.9/*/performance" \
        "closed/*/results/*/mixtral-8x7b/*/performance" \
        "closed/*/results/*/mixtral-8x7b-99/*/performance" \
        "closed/*/results/*/mixtral-8x7b-99.9/*/performance" \
        "closed/*/results/*/llama3.1-405b/*/performance" \
        "closed/*/results/*/llama3.1-405b-99/*/performance" \
        "closed/*/results/*/llama3.1-405b-99.9/*/performance" \
        "closed/*/results/*/gptj/*/performance" \
        "closed/*/results/*/gptj-99/*/performance" \
        "closed/*/results/*/gptj-99.9/*/performance"

    git checkout

    popd > /dev/null

    echo "   Done: ${TARGET}"
done

echo ""
echo "All requested rounds fetched. Run the parser with:"
echo "  python -m src.data.mlperf_parser --repos-dir ${DEST} --parquet"
