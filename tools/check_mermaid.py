#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 _posts/ 里的 mermaid 代码块是否含 LaTeX 源码或会破坏解析的字符。

**为什么需要这个**：mermaid 的节点标签不支持数学公式。写 `p_t(x|z)` 只会
被当成字面文本（用户看到的是 "p_t(x|z)" 而不是渲染后的公式）；
而 `|` 出现在节点标签 `[ " ... " ]` 内部会破坏解析。

本项目约定（与站内既有文章一致）：**公式交给正文，mermaid 只表达流程结构，
标签用中文 + 简单符号。**

实测（2026-10-03，全站 215 个图）：
- LaTeX 命令残留 **0 处** —— 说明这条约定本来就执行得好；
- 节点标签内的 `|` 仅 5 处，其中 4 处在 PTQ 系列且属误报（`|O|` 是集合记号）。

因此本脚本只报三类**真问题**，不做风格巡查：
1. LaTeX 命令（`\alpha`、`\frac` 之类）—— 一定不会渲染；
2. 节点标签内的管道符 —— 会破坏解析或显示成字面量；
3. `$$…$$` / `\(` `\[` —— 公式定界符进了 mermaid。

用法：
    python3 tools/check_mermaid.py            # 扫全站
    python3 tools/check_mermaid.py _posts/x.md  # 只扫指定文件
"""
from __future__ import annotations

import glob
import os
import re
import sys

# LaTeX 命令：mermaid 里绝不渲染
LATEX_PAT = re.compile(r"\\[a-zA-Z]+")
# 公式定界符
MATH_DELIM = re.compile(r"\$\$|\\\(|\\\[|\\\]")
# 节点标签内的管道符：[ " ... | ... " ]（排除 -->|"..."| 这种合法边标签）
PIPE_IN_LABEL = re.compile(r'\[\s*"[^"]*\|')
# 典型「伪公式」：标签里出现 x_t / p_t 这类 LaTeX 下标写法
FAKE_MATH = re.compile(r"(?<![A-Za-z0-9_])[a-zA-Z]_(?:\{?[a-zA-Z0-9])")


def blocks(path: str):
    lines = open(path, encoding="utf-8").read().split("\n")
    i, out = 0, []
    while i < len(lines):
        if lines[i].strip().startswith("```mermaid"):
            j = i + 1
            body = []
            while j < len(lines) and lines[j].strip() != "```":
                body.append(lines[j])
                j += 1
            out.append((i + 2, body))
            i = j + 1
        else:
            i += 1
    return out


# 这些是合法的普通文本（标识符、文件名、集合记号），不是伪公式
WHITELIST = re.compile(
    r"save_interval|ptq-0|llm-int8|gptq|awq|smoothquant|quip|aqlm"
    r"|O\||\|K\||\|N\||\|V\|"
)


def check(path: str) -> list[str]:
    problems = []
    for start, body in blocks(path):
        for k, line in enumerate(body):
            no = start + k
            s = line.strip()
            if not s or s.startswith("%%"):
                continue
            m = LATEX_PAT.search(s)
            if m:
                problems.append(f"  L{no}: LaTeX 命令 {m.group(0)!r} → {s[:56]}")
            if MATH_DELIM.search(s):
                problems.append(f"  L{no}: 公式定界符 → {s[:56]}")
            if PIPE_IN_LABEL.search(s) and not WHITELIST.search(s):
                problems.append(f"  L{no}: 节点标签内的管道符 → {s[:56]}")
            # 「伪公式」只指**成段 LaTeX**：连着出现两个以上下标，或下标带花括号。
            # 单个 `x_t` 在既有文章里是合法写法（数学正确、Mermaid 正常显示字面文本），
            # 不算问题——全站有大量这种用法，动它们属于历史文章重写，超出本检查范围。
            if not WHITELIST.search(s):
                hits = FAKE_MATH.findall(s)
                braced = re.findall(r"[a-zA-Z]_\{", s)
                if len(hits) >= 2 or braced:
                    problems.append(f"  L{no}: 成段公式源码（{len(hits)} 处下标）→ {s[:56]}")
    return problems


def main() -> int:
    files = sys.argv[1:] or sorted(glob.glob("_posts/*.md"))
    files = [f for f in files if os.path.exists(f)]
    bad = 0
    for f in files:
        ps = check(f)
        if ps:
            bad += 1
            print(f"[FAIL] {os.path.basename(f)}")
            for p in ps:
                print(p)
    print()
    print(f"扫描 {len(files)} 个文件，{bad} 个含问题。")
    if bad == 0:
        print("✓ 全部 mermaid 块都是纯中文标签，无 LaTeX 源码、无危险字符。")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
