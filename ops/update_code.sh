#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${1:-${GIT_BRANCH:-main}}"
cd "${ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "服务器工作区存在未提交改动；为避免覆盖，停止更新。" >&2
  git status --short
  exit 1
fi

git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
python ops/check_repository.py
echo "服务器代码已更新到 $(git rev-parse HEAD)"

