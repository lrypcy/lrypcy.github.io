#!/usr/bin/env python3
"""本地离线校验一篇 Jekyll 博文的 kramdown 渲染风险。

为什么需要这个工具
------------------
本地装的 kramdown 2.5.2 **没有 GFM parser**（`_config.yml` 用的是 `input: GFM`），
所以本地只能用 `input: 'kramdown'` 渲染，而该 parser 对 ``` 围栏的处理与 GFM 不同：
它把围栏降级成 `<p><code> ... </code></p>`，**多行块会散成多个段落**，
于是代码块里的 `*` `_` `|` 会被 Markdown 二次解析，伪造出 `<em>` / `<table>`。
（4 空格缩进块才会变成 `<pre>`，这是本地唯一可信的代码块形态。）

因此本工具先**剥离围栏代码块**，只对正文渲染，再把结论限定在两件确实可验的事上：

  1. 公式：源码里的 `$$...$$` 是否被转成 `\\(...\\)`（行内）/ `\\[...\\]`（块级），
     有没有被 `<em>` 撕开。
  2. 表格：渲染出的 `<table>` 数量是否等于源码里真实写的表格数——
     若多出来，说明某段正文里的裸 `|` 把整段判成了表格（TL;DR 被整块吞掉就是这个症状）。

用法
----
    python3 tools/check_post.py _posts/xxx.md
    python3 tools/check_post.py _posts/xxx.md --keep-html /tmp/out.html

需要 ruby + kramdown；找不到时脚本会明确报错并提示安装命令，不会静默跳过。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

RENDER_RB = r"""
require 'kramdown'
src = File.read(ARGV[0], encoding: 'UTF-8')
doc = Kramdown::Document.new(src, input: 'kramdown', math_engine: 'mathjax')
File.write(ARGV[1], doc.to_html, encoding: 'UTF-8')
"""

FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)(?P<info>.*)$")


def strip_fenced_blocks(lines):
    """把 ``` / ~~~ 围栏块整体替换成空行，返回 (新行列表, 被剥离的行号集合)。

    只剥离围栏块；4 空格缩进块保留（本地渲染器能正确识别）。
    """
    out, stripped, inside, fence = [], set(), False, None
    for i, line in enumerate(lines, start=1):
        m = FENCE_RE.match(line)
        if inside:
            stripped.add(i)
            out.append("")
            if m and line.strip().startswith(fence):
                inside, fence = False, None
            continue
        if m:
            inside, fence = True, m.group("fence")
            stripped.add(i)
            out.append("")
            continue
        out.append(line)
    return out, stripped


def count_source_math(lines):
    """统计源码里的行内/块级公式数。块级 = 整行只有一个 $$...$$。

    注意：先剥掉引用块前缀（`> `、`>   `）。TL;DR 与提示块里独占一行的
    `$$...$$` 在 kramdown 里同样渲染成块级 `\\[...\\]`；不剥前缀会把它
    误算成行内，从而报出假的“行内公式没被转换”。
    """
    inline = block = 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("|"):
            continue
        stripped_line = re.sub(r"^(?:>\s*)+", "", s).strip()
        # 整行 $$ ... $$ → 块级
        m = re.fullmatch(r"\$\$(.+?)\$\$", stripped_line)
        if m:
            block += 1
            continue
        # 其余按行内计
        inline += len(re.findall(r"\$\$.+?\$\$", stripped_line))
    return inline, block


def count_source_tables(lines):
    """统计源码里的表格块数：连续的以 | 开头且含至少一个 | 的行算一张表。"""
    tables, in_table = 0, False
    for line in lines:
        s = line.strip()
        is_row = s.startswith("|") and s.count("|") >= 2
        if is_row and not in_table:
            tables += 1
            in_table = True
        elif not is_row:
            in_table = False
    return tables


def find_renderer():
    """返回可用的 ruby 命令前缀，找不到就返回 None。"""
    if not shutil.which("ruby"):
        return None
    gems = os.path.expanduser("~/.gem/ruby")
    candidates = []
    for root, dirs, _ in os.walk(gems):
        for d in dirs:
            if d.startswith("kramdown-"):
                candidates.append(os.path.join(root, d, "lib"))
            if d.startswith("rexml-"):
                candidates.append(os.path.join(root, d, "lib"))
    if not candidates:
        return None
    inc = []
    for p in candidates:
        inc += ["-I", p]
    return ["ruby"] + inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--keep-html", metavar="OUT")
    args = ap.parse_args()

    src = open(args.path, encoding="utf-8").read()
    lines = src.split("\n")
    # 去掉 front matter，避免 YAML 里的内容干扰
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                lines = [""] * (i + 1) + lines[i + 1:]
                break

    clean_lines, stripped = strip_fenced_blocks(lines)
    clean_src = "\n".join(clean_lines)

    print("=" * 66)
    print(f"文件: {args.path}  ({len(lines)} 行)")
    print(f"剥离围栏代码块: {len(stripped)} 行（本地渲染器不认围栏，必须剥离）")
    print("=" * 66)

    renderer = find_renderer()
    if renderer is None:
        print("\n[!] 找不到 ruby + kramdown，跳过渲染校验。")
        print("    安装: gem install --user-install kramdown")
        print("    仅打印源码侧统计：")
        inl, blk = count_source_math(lines)
        print(f"      行内公式 {inl}  块级公式 {blk}  表格 {count_source_tables(lines)}")
        return 2

    with tempfile.TemporaryDirectory() as td:
        md = os.path.join(td, "in.md")
        rb = os.path.join(td, "render.rb")
        html = os.path.join(td, "out.html")
        open(md, "w", encoding="utf-8").write(clean_src)
        open(rb, "w", encoding="utf-8").write(RENDER_RB)
        env = dict(os.environ)
        env["GEM_HOME"] = os.path.expanduser("~/.gem/ruby/2.6.0")
        r = subprocess.run(renderer + [rb, md, html], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("[!] 渲染失败:\n" + r.stderr)
            return 2
        rendered = open(html, encoding="utf-8").read()
        if args.keep_html:
            open(args.keep_html, "w", encoding="utf-8").write(rendered)
            print(f"渲染结果已写出: {args.keep_html}")

    n_inline_out = len(re.findall(r"\\\(", rendered))
    n_block_out = len(re.findall(r"\\\[", rendered))
    n_inline_src, n_block_src = count_source_math(clean_lines)
    n_table_out = len(re.findall(r"<table", rendered))
    n_table_src = count_source_tables(clean_lines)

    ok = True

    print("\n[1] 公式")
    print(f"    源码 行内 {n_inline_src:>4} / 块级 {n_block_src:>4}")
    print(f"    渲染 \\(...\\) {n_inline_out:>4} / \\[...\\] {n_block_out:>4}")
    if n_inline_out < n_inline_src:
        print("    [!] 有行内公式没被转换 —— 很可能被 <em> 撕开，检查下面的 [3]")
        ok = False
    if n_block_out < n_block_src:
        print("    [!] 有块级公式没被转换")
        ok = False
    if n_inline_out >= n_inline_src and n_block_out >= n_block_src:
        print("    [ok] 全部转成 \\(...\\) / \\[...\\]")

    print("\n[2] 表格")
    print(f"    源码表格 {n_table_src}  渲染 <table> {n_table_out}")
    if n_table_out > n_table_src:
        print(f"    [!] 多出 {n_table_out - n_table_src} 张表：正文里有裸 | 把整段判成了表格")
        for m in re.finditer(r"<table[\s\S]{0,200}", rendered):
            print("        " + m.group(0)[:160].replace("\n", " "))
        ok = False
    else:
        print("    [ok] 表格数一致，无裸 | 误判")

    print("\n[3] <em> 出现处（应全部来自有意写的 *斜体*，如论文标题）")
    bad = 0
    for m in re.finditer(r".{0,50}<em>.{0,50}", rendered):
        ctx = m.group(0).replace("\n", " ")
        bad += 1
        print("    " + ctx)
    if bad == 0:
        print("    [ok] 无")
    else:
        print("    提示：论文标题/术语斜体属正常；若片段里出现公式的 _ ^ \\ 则是被撕开的公式")

    print("\n" + "=" * 66)
    print("[ok] 通过" if ok else "[!] 有问题，见上面标记")
    print("注意：本地只验公式与表格两项。代码块高亮、Liquid 模板、")
    print("      Mermaid 渲染都必须以线上 GitHub Pages 构建为准。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
