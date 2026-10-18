#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 _posts/ 里的站内链接能否解析到真实文件。

为什么需要这个工具
------------------
四步校验（check_post / scan_garbled / fix_quotes / 回填比对）管的是
**渲染与内容一致性**，管不了链接目标是否存在。2026-10-18 的 serving /
specdec 全量铺开就踩过：specdec-05/06/07/08 的站内承接链接写成
`/2026/10/18-specdec-…`（日期与 slug 之间少一层斜杠），serving-00 的
megatron 链接日期整个写错——两类错四步校验全部静默通过，只有线上点开
才是 404。本工具把这类错误拦在提交前。

判定规则
--------
1. `/YYYY/MM/DD/<slug>/` 形式 → 必须能在 `_posts/` 找到同名文件
   （Jekyll 的默认 permalink 规则，与站内既有链接一致）；
2. 顶层页面（`/about/` `/archives/` `/categories/` `/tags/` `/stats/` 等）
   → 白名单；
3. `/assets/…` `/css/…` → 白名单前缀（静态资源目录）；
4. `http(s)://` 外链 → 不检查（沙箱出不去，线上也可能变）；
5. 其余 `/…` 一律报错（含查询串的先剥掉 `?…`，含锚点的再剥掉 `#…` 再判；
   `/tags/?tag=量化`、`/categories/?cat=xxx` 这类「带参数只为过滤」的链接按 `/tags/`
   判定，别再误报成坏链）。

用法
----
    python3 tools/check_links.py                  # 扫全站 _posts/
    python3 tools/check_links.py _posts/x.md ...   # 只扫指定文件
退出码：0 = 无坏链；1 = 有坏链（可挂 pre-commit / CI）。
"""

from __future__ import annotations

import glob
import os
import re
import sys

POSTS_DIR = "_posts"

# 静态资源前缀白名单
ASSET_PREFIXES = ("/assets/", "/css/", "/images/", "/js/")

# 根目录里被 _config.yml exclude、不生成页面的文件
EXCLUDED_ROOT_FILES = {"README.md", "TODO.md", "AGENTS.md"}

# markdown 内联链接：](/path) 与 ](/path#anchor)；不匹配图片 ![..](/..)
LINK_PAT = re.compile(r"(?<!\!)\]\((/[^)\s]+)\)")

# HTML 属性里的站内链接（include / layout 不扫，只扫 _posts 正文）
HREF_PAT = re.compile(r'href="(/[^"\s]+)"')


def root_page_permalinks() -> set[str]:
    """根目录 .md/.html 生成的页面 permalink（/name/，index → /）。

    _config.yml 的 permalink 是 /:year/:month/:day/:title/（只对 _posts 生效），
    根目录页面走默认规则：about.md → /about/，index.html → /。
    exclude 掉的文件（README/TODO/AGENTS）不发布，不算。
    """
    out: set[str] = {"/"}
    for name in os.listdir("."):
        if name in EXCLUDED_ROOT_FILES:
            continue
        if name.endswith(".md"):
            out.add(f"/{name[:-3]}/")
        elif name.endswith(".html"):
            out.add("/" if name == "index.html" else f"/{name[:-5]}/")
    return out


def post_permalinks() -> set[str]:
    """_posts/YYYY-MM-DD-slug.md → /YYYY/MM/DD/slug/。"""
    out: set[str] = set()
    for path in glob.glob(os.path.join(POSTS_DIR, "*.md")):
        m = re.match(
            r"(\d{4})-(\d{2})-(\d{2})-(.+)\.md$", os.path.basename(path)
        )
        if m:
            out.add(f"/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}/")
    return out


def resolve(target: str, permalinks: set[str], pages: set[str]) -> bool:
    if target.startswith(("http://", "https://", "mailto:", "//")):
        return True
    if "?" in target:
        target = target.split("?", 1)[0]  # 查询参数：/tags/?tag=x（分类/标签按参数过滤）
    if "#" in target:
        target = target.split("#", 1)[0]
        if target == "":
            return True
    if not target.startswith("/"):
        return False  # 相对链接一律不合法（站点根路径发布）
    if target in permalinks or target in pages:
        return True
    if target.startswith(ASSET_PREFIXES):
        return True
    if not target.endswith("/"):
        # 文件型静态引用（如 /favicon.ico、/search-index.json）
        return target.lstrip("/") and os.path.exists(target.lstrip("/"))
    return False


def check_file(path: str, permalinks: set[str], pages: set[str]) -> list[str]:
    problems: list[str] = []
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        for m in LINK_PAT.finditer(line):
            target = m.group(1)
            if not resolve(target, permalinks, pages):
                problems.append(f"{path}:{lineno}: 坏链 {target}")
        for m in HREF_PAT.finditer(line):
            target = m.group(1)
            if not resolve(target, permalinks, pages):
                problems.append(f"{path}:{lineno}: 坏链(href) {target}")
    return problems


def main(argv: list[str]) -> int:
    files = argv[1:] or sorted(glob.glob(os.path.join(POSTS_DIR, "*.md")))
    permalinks = post_permalinks()
    pages = root_page_permalinks()
    all_problems: list[str] = []
    for f in files:
        all_problems.extend(check_file(f, permalinks, pages))
    if all_problems:
        print(f"check_links: {len(all_problems)} 个坏链")
        for p in all_problems:
            print("  " + p)
        return 1
    print(f"check_links: {len(files)} 个文件的站内链接全部可解析 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
