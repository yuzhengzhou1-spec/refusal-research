#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ENV="${ROOT}/ops/server.local.env"
if [[ ! -f "${LOCAL_ENV}" ]]; then
  echo "缺少 ${LOCAL_ENV}；请从server.local.env.example复制并填写。" >&2
  exit 1
fi
set -a
source "${LOCAL_ENV}"
set +a

BASE="${ROOT}/evidence_posttrain_baselines"
CONFIG="${EXPERIMENT_CONFIG:-${BASE}/configs/experiment.server.yaml}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${ROOT}/.venv-train/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-${ROOT}/.venv-vllm/bin/python}"
STAGE="${1:-}"

case "${STAGE}" in
  prepare)
    "${TRAIN_PYTHON}" "${BASE}/prepare_dataset.py" --config "${CONFIG}"
    ;;
  preflight)
    "${TRAIN_PYTHON}" "${BASE}/preflight.py" --config "${CONFIG}" --tokenizer-check
    ;;
  serve)
    PATH="$(dirname "${VLLM_PYTHON}"):${PATH}" "${VLLM_PYTHON}" "${BASE}/serve_vllm.py" --config "${CONFIG}" --run
    ;;
  eval)
    "${VLLM_PYTHON}" "${ROOT}/ops/run_with_manifest.py" \
      --name "${RUN_NAME:-baseline_eval}" --output-root "${EXPERIMENT_ROOT}/manifests" \
      --config "${CONFIG}" --config "${BASE}/configs/task_prompt.yaml" --cwd "${BASE}" -- \
      "${VLLM_PYTHON}" "${BASE}/run_vllm_eval.py" --config "${CONFIG}" --split "${EVAL_SPLIT:-test}" --run-name "${RUN_NAME:-baseline_eval}"
    ;;
  sft)
    "${TRAIN_PYTHON}" "${ROOT}/ops/run_with_manifest.py" \
      --name "${RUN_NAME:-sft_lora}" --output-root "${EXPERIMENT_ROOT}/manifests" \
      --config "${CONFIG}" --config "${BASE}/configs/training/sft_lora.yaml" --cwd "${BASE}" -- \
      "$(dirname "${TRAIN_PYTHON}")/accelerate" launch "${BASE}/train_sft_lora.py" --config "${CONFIG}"
    ;;
  dpo-build)
    "${TRAIN_PYTHON}" "${BASE}/build_dpo_dataset.py" --config "${CONFIG}"
    ;;
  dpo)
    "${TRAIN_PYTHON}" "${ROOT}/ops/run_with_manifest.py" \
      --name "${RUN_NAME:-dpo_lora}" --output-root "${EXPERIMENT_ROOT}/manifests" \
      --config "${CONFIG}" --config "${BASE}/configs/training/dpo_lora.yaml" --cwd "${BASE}" -- \
      "$(dirname "${TRAIN_PYTHON}")/accelerate" launch "${BASE}/train_dpo_lora.py" --config "${CONFIG}"
    ;;
  *)
    echo "用法：$0 {prepare|preflight|serve|eval|sft|dpo-build|dpo}" >&2
    exit 2
    ;;
esac
