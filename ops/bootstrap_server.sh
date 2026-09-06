#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ENV="${ROOT}/ops/server.local.env"
if [[ -f "${LOCAL_ENV}" ]]; then
  # 本文件只应包含可信的本地路径和环境变量。
  set -a
  source "${LOCAL_ENV}"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TRAIN_ENV="${ROOT}/.venv-train"
VLLM_ENV="${ROOT}/.venv-vllm"

command -v git >/dev/null
command -v nvidia-smi >/dev/null
command -v "${PYTHON_BIN}" >/dev/null
nvidia-smi

if [[ "${1:-check}" == "check" ]]; then
  echo "基础检查通过。运行 '$0 install' 创建训练和vLLM两个隔离环境。"
  exit 0
fi
if [[ "${1}" != "install" ]]; then
  echo "用法：$0 [check|install]" >&2
  exit 2
fi

"${PYTHON_BIN}" -m venv "${TRAIN_ENV}"
"${TRAIN_ENV}/bin/python" -m pip install --upgrade pip
"${TRAIN_ENV}/bin/python" -m pip install torch --index-url "${TORCH_INDEX_URL}"
"${TRAIN_ENV}/bin/python" -m pip install -r "${ROOT}/evidence_posttrain_baselines/requirements/train.txt"

"${PYTHON_BIN}" -m venv "${VLLM_ENV}"
"${VLLM_ENV}/bin/python" -m pip install --upgrade pip
"${VLLM_ENV}/bin/python" -m pip install -r "${ROOT}/evidence_posttrain_baselines/requirements/inference.txt"

if [[ ! -f "${ROOT}/ops/server.local.env" ]]; then
  cp "${ROOT}/ops/server.local.env.example" "${ROOT}/ops/server.local.env"
fi
if [[ ! -f "${ROOT}/evidence_posttrain_baselines/configs/experiment.server.yaml" ]]; then
  cp "${ROOT}/evidence_posttrain_baselines/configs/experiment.server.example.yaml" \
    "${ROOT}/evidence_posttrain_baselines/configs/experiment.server.yaml"
fi
echo "环境已创建。下一步修改两个local/server配置文件中的真实路径。"

