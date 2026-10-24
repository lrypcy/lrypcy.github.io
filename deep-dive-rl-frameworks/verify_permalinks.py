#!/usr/bin/env python3
"""抽验文章里的 GitHub permalink 行号是否指向本地仓库里真实存在且非空的代码行。

用法: python3 verify_permalinks.py _posts/xxx.md [--max N]

只做本地 git 校验（不访问网络）：
  git -C <repo> show "<commit>:<path>"  取出文件
  校验行号范围在文件行数内、且摘录内容非空白
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/congyuan/Desktop/Projects/github")

REPOS = {
    "verl-project/verl": "verl",
    "volcengine/verl": "verl",
    "THUDM/slime": "slime",
    "sgl-project/sglang": "sglang",
    "vllm-project/vllm": "vllm",
    "NVIDIA/Megatron-LM": "Megatron-LM",
    "NVIDIA-NeMo/Megatron-Bridge": "Megatron-Bridge",
    "areal-project/AReaL": "AReaL",
    "RLinf/RLinf": "RLinf",
    "huggingface/trl": "trl",
}

PAT = re.compile(
    r"https://github\.com/(?P<org>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"/blob/(?P<commit>[0-9a-f]{7,40})/"
    r"(?P<path>[^#)\s]+)"
    r"(?:#L(?P<start>\d+)(?:-L(?P<end>\d+))?)?"
)


def show(repo_dir: str, commit: str, path: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT / repo_dir), "show", f"{commit}:{path}"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.split("\n")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--max"):
            limit = int(a.split("=")[1]) if "=" in a else None
    if not args:
        print("usage: verify_permalinks.py FILE [--max=N]")
        return 2

    cache: dict[tuple, list[str] | None] = {}
    bad, ok, skipped, no_line = [], 0, 0, 0

    for f in args:
        text = Path(f).read_text(encoding="utf-8")
        seen = set()
        for m in PAT.finditer(text):
            org, commit, path = m.group("org"), m.group("commit"), m.group("path")
            start, end = m.group("start"), m.group("end")
            key = (org, commit, path)
            if key not in seen:
                seen.add(key)
            if not start:
                no_line += 1
                continue
            s = int(start)
            e = int(end) if end else s
            repo = REPOS.get(org)
            if repo is None:
                skipped += 1
                continue
            if key not in cache:
                cache[key] = show(repo, commit, path)
            lines = cache[key]
            if lines is None:
                bad.append((f, org, commit[:9], path, f"L{s}-L{e}", "文件/commit 不存在"))
                continue
            # 去掉末尾 split 产生的空元素
            n = len(lines)
            while n and lines[n - 1] == "":
                n -= 1
            if e > n:
                bad.append((f, org, commit[:9], path, f"L{s}-L{e}", f"超出文件行数 {n}"))
                continue
            excerpt = "\n".join(lines[s - 1: e]).strip()
            if not excerpt:
                bad.append((f, org, commit[:9], path, f"L{s}-L{e}", "摘录全为空白"))
                continue
            ok += 1

    print(f"校验: {ok} 通 / {len(bad)} 坏 / {skipped} 跳过(未知仓库) / {no_line} 无行号")
    for b in bad[:60]:
        print("  坏链:", b[0].split("/")[-1], b[1], b[2], b[3], b[4], "->", b[5])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
