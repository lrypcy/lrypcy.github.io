#!/usr/bin/env python3
"""Audit source permalinks in a post against the pinned Megatron-LM commit.

Verifies, for every `github.com/NVIDIA/Megatron-LM/blob/<commit>/<path>#L<a>-L<b>`
link in the post:
  1. the file exists at that commit
  2. the cited line range lies inside the file's real line count

Also reports links that carry no `#L` anchor (file-level only), which the
project red line (AGENTS.md §8 "行号给 permalink") does not accept as a
substitute for an inline, line-anchored permalink.

Usage:
    ~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <post.md> [...]
"""
import argparse
import collections
import re
import subprocess
import sys

DEFAULT_REPO = "/Users/congyuan/Desktop/Projects/github/Megatron-LM"
DEFAULT_COMMIT = "60e039626"

# Matches .../blob/<COMMIT>/<path>.py[#L<a>[-L<b>]]
LINK_RE = r"https://github\.com/NVIDIA/Megatron-LM/blob/{commit}/([A-Za-z0-9_./-]+\.py)(?:#L(\d+)(?:-L?(\d+))?)?"
# Bare line refs written as （L612） with no filename — ambiguous when a post
# cites many files. Reported so the author can attach a filename or permalink.
BARE_LINE_RE = re.compile(r"（L(\d+)")


def audit(post, repo, commit):
    text = open(post, encoding="utf-8").read()
    pattern = re.compile(LINK_RE.format(commit=commit))
    matches = list(pattern.finditer(text))

    seen = collections.OrderedDict()
    for m in matches:
        key = (m.group(1), m.group(2), m.group(3))
        seen[key] = seen.get(key, 0) + 1

    cache = {}

    def lines_of(path):
        if path not in cache:
            r = subprocess.run(
                ["git", "-C", repo, "show", f"{commit}:{path}"],
                capture_output=True, text=True,
            )
            cache[path] = r.stdout.splitlines() if r.returncode == 0 else None
        return cache[path]

    bad, unanchored, ok = [], [], 0
    for (path, s, e), _count in seen.items():
        lines = lines_of(path)
        if lines is None:
            bad.append((path, "file does not exist at commit"))
            continue
        n = len(lines)
        if not s:
            unanchored.append(path)
            ok += 1
            continue
        a = int(s)
        b = int(e) if e else a
        if a < 1 or b > n or a > b:
            anchor = f"{path}#L{s}-L{e}" if e else f"{path}#L{s}"
            bad.append((anchor, f"line range outside file (file has {n} lines)"))
        else:
            ok += 1

    bare = BARE_LINE_RE.findall(text)

    print(f"== {post}")
    print(f"   permalink occurrences : {len(matches)}")
    print(f"   distinct (path,range) : {len(seen)}")
    print(f"   resolvable            : {ok}")
    print(f"   broken                : {len(bad)}")
    for p, why in bad:
        print(f"      !! {p}: {why}")
    if unanchored:
        print(f"   no #L anchor          : {len(unanchored)}")
        for p in sorted(set(unanchored)):
            print(f"      ?  {p}  ({len(lines_of(p) or [])} lines)")
    if bare:
        print(f"   bare （L<n> refs      : {len(bare)}  -> attach a filename or permalink")
        for n_ in sorted(set(bare), key=int):
            print(f"      ?  （L{n_}")
    if not bad and not unanchored and not bare:
        print("   [ok] all permalinks anchored and in bounds; no bare line refs")
    print()
    return 1 if (bad or unanchored or bare) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("posts", nargs="+")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    args = ap.parse_args()

    rc = 0
    for post in args.posts:
        rc |= audit(post, args.repo, args.commit)
    sys.exit(rc)


if __name__ == "__main__":
    main()
