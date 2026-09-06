from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
BLOCKED_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".pem", ".key"}
BLOCKED_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", "credentials.json"}
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
]


def git_files(staged: bool) -> list[Path]:
    if staged:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    else:
        command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="提交前检查大文件、模型权重和常见密钥")
    parser.add_argument("--staged", action="store_true", help="只检查已暂存文件")
    args = parser.parse_args()
    errors = []
    files = git_files(args.staged)
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"大于10MB：{relative}")
        if path.suffix.lower() in BLOCKED_SUFFIXES or path.name.lower() in BLOCKED_NAMES:
            errors.append(f"疑似权重或密钥文件：{relative}")
        if path.stat().st_size <= 2 * 1024 * 1024:
            content = path.read_bytes()
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                errors.append(f"内容疑似包含密钥：{relative}")
    if errors:
        print("仓库安全检查失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"仓库安全检查通过：{len(files)} 个候选文件")


if __name__ == "__main__":
    main()

