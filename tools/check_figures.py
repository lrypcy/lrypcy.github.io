#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体检正文配图的图注：来源超链接、许可、改动声明。

为什么需要这个工具
------------------
`.agents/WRITING.md` §1.0 规定「每张图下面必须有一行斜体图注」，§3 规定「配图也是引用」。
但六步校验里没有任何一步看图：漏写图注、光写「图源：DiT 论文」不给链接、
或漏掉 CC BY 4.0 的许可 / 改动声明，都能一路静默提交上线。
2026-11-09 dit-01 首次外链论文原图时补了这套要求，同步落一个机械判定。

只做能机械判定的部分，判断不了的不猜：
  - 图片行后面 6 行内有没有以 `*图源：` / `*图：自绘` / `*图：` 开头的斜体图注
  - 外链图（http(s)://）图注里有没有可点击来源链接、许可字样、改动声明
  - 站内自绘图（/assets/img/...）图注缺链接只报 WARN（实验 permalink 形式不固定）
  - 内联 `<svg>` 不在本工具范围（跨几十行，判据不稳），仍按 §1.0 人工遵守

用法
----
    python3 tools/check_figures.py _posts/xxx.md          # 单篇全量，只报告
    python3 tools/check_figures.py _posts/*.md --summary  # 批量，只打有问题的篇
    python3 tools/check_figures.py --staged               # 只查本次暂存新增的图片行
    python3 tools/check_figures.py --staged --strict      # 有 ERROR 则退出码 1（挂钩子）

退出码：默认恒为 0（只报告，和 check_links.py 一致）。--strict 下有 ERROR 返回 1。
`--staged` 读的是暂存区 blob（`git show :<path>`），不受工作区未暂存改动影响。
存量 100 多张老图不批量迁移（与 TL;DR 同口径），所以默认模式只报告、--staged 只看新增行。

规范条文见 `.agents/WRITING.md` §1.0「图注硬性要求」与 §3。
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys

# Markdown 图片：![alt](src)。只认单独成行的，行内插图不算。
IMG_PAT = re.compile(r"^!\[[^\]]*\]\(([^)\s]+)\)\s*$")
# 图注开头的三种合法写法
CAP_START = ("*图源：", "*图：自绘", "*图：")
# 改动声明（CC BY 4.0 三要素之一）
CHANGE_WORDS = ("未改动", "未搬用原图", "有裁剪", "已改动", "有改动")
# 图注里出现这些说明已经到了下一块内容，别再往后找
STOP_PREFIX = ("#", "![", "```", "---", "|", "<svg", "$$")

LINK_PAT = re.compile(r"\]\((https?://[^)\s]+)\)")


def caption_for(lines: list[str], idx: int) -> tuple[str | None, int | None]:
    """返回图片行 idx 之后的图注文本与其行号（0-based）；找不到返回 (None, None)。"""
    for j in range(idx + 1, min(idx + 7, len(lines))):
        t = lines[j].strip()
        if not t:
            continue
        if t.startswith(CAP_START):
            return t, j + 1
        if t.startswith(STOP_PREFIX):
            return None, None
        # 允许图注前有一两句「左图 / 右图分别说明了什么」，但只放行普通正文行
    return None, None


def check_one(path: str, lines: list[str], targets: set[int] | None = None) -> list[tuple[str, int, str]]:
    """返回 [(级别, 行号, 说明)]；targets 非空时只检查这些 1-based 行号。"""
    issues: list[tuple[str, int, str]] = []
    for i, line in enumerate(lines):
        lineno = i + 1
        if targets is not None and lineno not in targets:
            continue
        m = IMG_PAT.match(line.rstrip("\n"))
        if not m:
            continue
        src = m.group(1)
        cap, cap_at = caption_for(lines, i)
        if cap is None:
            issues.append(("ERROR", lineno, f"缺图注：{src[:70]}"))
            continue
        external = src.startswith(("http://", "https://"))
        if external:
            if "图源：" not in cap:
                issues.append(("ERROR", cap_at or lineno, "外链图图注必须以「图源：」开头"))
            if not LINK_PAT.search(cap):
                issues.append(("ERROR", cap_at or lineno, "图注缺可点击的来源超链接"))
            if "许可" not in cap:
                issues.append(("ERROR", cap_at or lineno, "图注缺许可（如 CC BY 4.0）"))
            if not any(w in cap for w in CHANGE_WORDS):
                issues.append(("ERROR", cap_at or lineno, "图注缺改动声明（未改动 / 有裁剪）"))
            if "Figure" not in cap and "Fig." not in cap:
                issues.append(("WARN", cap_at or lineno, "图注没写 Figure 编号，建议补上"))
        else:
            if not LINK_PAT.search(cap):
                issues.append(("WARN", cap_at or lineno, "自绘图图注没给来源超链接（实验 permalink 或结构对照）"))
    return issues


def staged_targets() -> dict[str, tuple[list[str], set[int]]]:
    """从 git diff --cached -U0 里取出新增的 Markdown 图片行，读暂存区 blob。"""
    out = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", "_posts/*.md"],
        capture_output=True, text=True, check=True,
    ).stdout
    result: dict[str, tuple[list[str], set[int]]] = {}
    path: str | None = None
    new_ln = 0
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            blob = subprocess.run(
                ["git", "show", f":{path}"], capture_output=True, text=True, check=True
            ).stdout
            result[path] = (blob.splitlines(), set())
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,\d+)?", line)
            new_ln = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            if path and IMG_PAT.match(line[1:].rstrip("\n")):
                result[path][1].add(new_ln)
            new_ln += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass
        else:
            new_ln += 1
    return {p: v for p, v in result.items() if v[1]}


def main() -> int:
    ap = argparse.ArgumentParser(description="检查正文配图的图注与来源引用")
    ap.add_argument("files", nargs="*", help="要检查的 .md 文件（默认全站 _posts/*.md）")
    ap.add_argument("--staged", action="store_true", help="只检查暂存区里新增的图片行")
    ap.add_argument("--strict", action="store_true", help="有 ERROR 时退出码 1")
    ap.add_argument("--summary", action="store_true", help="只打印有问题的篇目")
    args = ap.parse_args()

    if args.staged:
        jobs = [(p, lines, tg) for p, (lines, tg) in staged_targets().items()]
        if not jobs:
            print("check_figures: 暂存区没有新增图片行，跳过")
            return 0
    else:
        paths = args.files or sorted(glob.glob("_posts/*.md"))
        jobs = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                jobs.append((p, f.read().splitlines(), None))

    n_err = n_warn = n_img = 0
    for path, lines, targets in jobs:
        issues = check_one(path, lines, targets)
        imgs = sum(1 for i, l in enumerate(lines, 1)
                   if (targets is None or i in targets) and IMG_PAT.match(l.rstrip("\n")))
        n_img += imgs
        errs = [x for x in issues if x[0] == "ERROR"]
        warns = [x for x in issues if x[0] == "WARN"]
        n_err += len(errs)
        n_warn += len(warns)
        if not issues:
            if not args.summary:
                print(f"[OK] {path}  {imgs} 张图，图注齐全")
            continue
        print(f"[{'FAIL' if errs else 'WARN'}] {path}  {imgs} 张图，{len(errs)} ERROR / {len(warns)} WARN")
        for level, ln, msg in issues:
            print(f"    {level:<5} L{ln}: {msg}")

    scope = "暂存新增图片" if args.staged else "全量"
    print(f"\ncheck_figures: {scope} 共 {n_img} 张图，{n_err} ERROR / {n_warn} WARN"
          f"（默认只报告，--strict 才阻断）")
    return 1 if (args.strict and n_err) else 0


if __name__ == "__main__":
    sys.exit(main())
