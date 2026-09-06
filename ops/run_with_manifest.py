from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str], cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"
    except OSError:
        return "UNAVAILABLE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行实验并记录代码、配置、硬件和环境快照")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--cwd", default=str(ROOT))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("缺少要执行的命令；请放在 -- 之后")

    dirty = command_output(["git", "status", "--porcelain"])
    if dirty and not args.allow_dirty:
        raise SystemExit("工作区存在未提交改动；请提交后再跑，或显式使用 --allow-dirty")
    commit = command_output(["git", "rev-parse", "HEAD"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser().resolve() / f"{timestamp}_{args.name}_{commit[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_dir = run_dir / "configs"
    config_dir.mkdir()

    configs = []
    for item in args.config:
        source = Path(item).expanduser().resolve()
        destination = config_dir / source.name
        shutil.copy2(source, destination)
        configs.append({"source": str(source), "snapshot": str(destination), "sha256": sha256(source)})

    manifest: dict[str, Any] = {
        "name": args.name,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": commit, "branch": command_output(["git", "branch", "--show-current"]), "dirty": bool(dirty)},
        "command": command,
        "working_directory": str(Path(args.cwd).resolve()),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "gpu": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
        "packages": command_output([sys.executable, "-m", "pip", "freeze"]),
        "safe_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "NCCL_DEBUG")},
        "configs": configs,
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    exit_code = 1
    try:
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=args.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            exit_code = process.wait()
    finally:
        manifest["status"] = "SUCCESS" if exit_code == 0 else "FAILED"
        manifest["exit_code"] = exit_code
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
