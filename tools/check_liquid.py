#!/usr/bin/env python3
"""体检：抓出会让 GitHub Pages 构建直接挂掉的内容。

为什么需要它
------------
Jekyll 在 Markdown 转 HTML **之前**先跑 Liquid。正文里任何一处 `{{` 或 `{%`
都会被当成 Liquid 标记的开头；Liquid 找不到配对的 `}}` / `%}` 就抛
`Liquid syntax error`，整个站点构建失败 —— GitHub Actions 里那一步只跑 2 秒就红。

最容易踩的是 LaTeX 的双花括号，例如

    $$\\sum_j H(p_{{\\rm data},j})$$        # `{{` → 构建挂
    $$\\sum_j H(p_{\\mathrm{data},j})$$     # 改成 \\mathrm{} 就好了

这类写法本地看不出来（浏览器里公式照常渲染），只有构建会红，而且报错被 GitHub
检查注解截断到 4096 字符，很难一眼定位到出错的那一篇。

本工具做的事
------------
扫描所有会被 Jekyll 渲染的文件（`_posts/`、根目录 *.md/*.html、`wiki/`、
`Clippings/`、`deep-dive-*/`），逐篇：

  1. 剥掉 YAML front matter；
  2. 剥掉 `{% raw %}...{% endraw %}` 区块（那里面的花括号是有意为之）；
  3. 剩下的 `{{` / `{%` 一律报出来 —— 它们都会让构建失败；
  4. 顺带检查 `{% raw %}` 与 `{% endraw %}` 是否配对。

模板类文件（`index.html` / `about.md` / `archives.md` / `categories.md` /
`tags.md` / `search-index.json` / `_includes/` / `_layouts/`）本来就在写 Liquid，
直接跳过。

用法
----
    python3 tools/check_liquid.py              # 扫全部内容文件
    python3 tools/check_liquid.py _posts/x.md  # 只扫指定文件
    python3 tools/check_liquid.py --staged     # 只扫暂存区里待提交的文件

退出码：0 = 干净；1 = 有问题，必须处理；2 = 用错了（路径不存在等）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 这些文件/目录本来就是在写 Liquid，跳过
TEMPLATE_EXACT = {
    "index.html",
    "about.md",
    "archives.md",
    "categories.md",
    "tags.md",
    "search-index.json",
    "quantization-roadmap.md",
}
TEMPLATE_DIRS = ("_includes", "_layouts", "tools", "_site", ".git", ".workbuddy", "assets")

RAW_RE = re.compile(r"\{%-?\s*raw\s*-?%\}.*?\{%-?\s*endraw\s*-?%\}", re.S)
OPEN_RE = re.compile(r"\{\{|\{%-?")
RAW_OPEN_RE = re.compile(r"\{%-?\s*raw\s*-?%\}")
RAW_CLOSE_RE = re.compile(r"\{%-?\s*endraw\s*-?%\}")


def strip_front_matter(text: str):
    """去掉开头的 YAML front matter（若存在）。

    返回 (正文, 被剥掉的行数)，行数用来把报错还原到文件真实行号。
    """
    if not text.startswith("---"):
        return text, 0
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "".join(lines[i + 1:]), i + 1
    return text, 0


def iter_candidates():
    """列出所有需要体检的内容文件。"""
    for name in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, name)
        if os.path.isfile(path) and name.endswith((".md", ".html")):
            if name in TEMPLATE_EXACT:
                continue
            yield path
    for d in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, d)
        if not os.path.isdir(sub) or d in TEMPLATE_DIRS:
            continue
        for dirpath, dirnames, filenames in os.walk(sub):
            dirnames[:] = [x for x in dirnames if x not in TEMPLATE_DIRS]
            for fn in filenames:
                if fn.endswith((".md", ".html")):
                    yield os.path.join(dirpath, fn)


def check_file(path: str) -> list[str]:
    """返回一个文件里的问题列表（每条一行，含行号）。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    problems: list[str] = []
    rel = os.path.relpath(path, ROOT)

    # raw 区块是否配对
    if len(RAW_OPEN_RE.findall(text)) != len(RAW_CLOSE_RE.findall(text)):
        problems.append(
            f"{rel}: raw/endraw 数量不匹配 "
            f"({len(RAW_OPEN_RE.findall(text))} vs {len(RAW_CLOSE_RE.findall(text))})"
        )

    body, fm_lines = strip_front_matter(text)
    # raw 区块整体挖掉，但保留行号：用等量空行占位
    body = RAW_RE.sub(lambda m: "\n" * m.group(0).count("\n"), body)

    for i, line in enumerate(body.splitlines(), 1):
        lineno = i + fm_lines
        for m in OPEN_RE.finditer(line):
            col = m.start() + 1
            snippet = line.strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            problems.append(
                f"{rel}:{lineno}:{col}  裸 {m.group(0)!r}  会被 Liquid 当标记解析\n    {snippet}"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="检查 Liquid 冲突符号")
    ap.add_argument("paths", nargs="*", help="指定要检查的文件（默认扫全部内容文件）")
    ap.add_argument("--staged", action="store_true", help="只检查暂存区待提交的文件")
    args = ap.parse_args()

    if args.staged:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        paths = [os.path.join(ROOT, p) for p in out
                 if p.endswith((".md", ".html")) and os.path.basename(p) not in TEMPLATE_EXACT]
    elif args.paths:
        paths = [os.path.abspath(p) for p in args.paths]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print("找不到文件：", ", ".join(missing), file=sys.stderr)
            return 2
    else:
        paths = list(iter_candidates())

    all_problems: list[str] = []
    for p in paths:
        if not p.endswith((".md", ".html")):
            continue
        if os.path.basename(p) in TEMPLATE_EXACT:
            continue
        rel = os.path.relpath(p, ROOT)
        if any(rel.startswith(d + os.sep) for d in TEMPLATE_DIRS):
            continue
        all_problems.extend(check_file(p))

    print(f"check_liquid: 检查了 {len(paths)} 个文件")
    if not all_problems:
        print("check_liquid: 干净，没有会被 Liquid 误解析的裸标记 ✅")
        return 0

    print(f"check_liquid: 发现 {len(all_problems)} 处问题 ❌\n")
    for line in all_problems:
        print(line)
    print("\n修法：数学里的双花括号改成单花括号写法（如 {\\rm data} → \\mathrm{data}）；"
          "\n     代码块里的 {{ }} 请包一层 {% raw %}...{% endraw %}。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
