#!/usr/bin/env bash
# Regenerate the CUTI retinal age gaps and the full dataset from scratch.
# Usage: bash run_all.sh   (set PYTHON to choose an interpreter, GPUS for run-cv)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0 1 2}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m retinal_age_factors download
"${PYTHON}" -m retinal_age_factors make-splits
# shellcheck disable=SC2086
"${PYTHON}" -m retinal_age_factors run-cv --gpus ${GPUS}
"${PYTHON}" -m retinal_age_factors build-dataset

# AI-READI external validation (requires the AI-READI dataset under data/aireadi):
# "${PYTHON}" -m retinal_age_factors prepare-aireadi
# "${PYTHON}" -m retinal_age_factors run-automorph --devices cuda:0 cuda:1 cuda:2
# "${PYTHON}" -m retinal_age_factors train-final
# "${PYTHON}" -m retinal_age_factors external
