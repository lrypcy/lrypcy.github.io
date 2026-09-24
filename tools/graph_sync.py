#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱体检 / 同步助手
------------------------------------------------------------
回答一个问题：站点新增/修改文章后，_data/knowledge_graph.yml 需不需要手工改？

用法：
    python3 tools/graph_sync.py            # 体检报告
    python3 tools/graph_sync.py --links    # 额外输出「站内互引」建议的依赖关系

输出四件事：
  1. 每个系列当前命中篇数（与首页图谱圆圈里的数字一致）
  2. 未命中任何系列的文章  -> 需要在 yml 里加前缀规则，否则它不会出现在图谱里
  3. site 分类中未在 yml 配置的分类 -> 首页有兜底显示，但建议补元信息（图标/颜色/描述）
  4. 系列之间的站内互引统计 -> 建议新增/加强的依赖边（--links）
"""
import os
import re
import sys
import glob
import yaml
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTS = os.path.join(ROOT, "_posts")
YML = os.path.join(ROOT, "_data", "knowledge_graph.yml")

FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
LINK_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/([a-z0-9][a-z0-9\-]*)/?")


def load_posts():
    posts = []
    for f in sorted(glob.glob(os.path.join(POSTS, "*.md"))):
        base = os.path.basename(f)
        raw = open(f, encoding="utf-8").read()
        m = FM_RE.match(raw)
        fm = m.group(1) if m else ""
        body = raw[m.end():] if m else raw
        title = re.search(r'^title:\s*"?(.*?)"?\s*$', fm, re.M)
        cats = re.findall(r"^\s*-\s*(.+?)\s*$", fm.split("categories:")[-1].split("tags:")[0], re.M) \
            if "categories:" in fm else []
        slug = base[11:-3]
        y, md, d = base[:10].split("-")
        url = "/%s/%s/%s/%s/" % (y, md, d, slug)
        # 正文里的站内链接
        refs = set()
        for mm in LINK_RE.finditer(body):
            refs.add("/%s/%s/%s/%s/" % mm.groups())
        posts.append({
            "file": base, "title": title.group(1) if title else base, "url": url,
            "categories": cats, "refs": refs,
        })
    return posts


def match(post, series):
    """与 index.html 里的 Liquid 逻辑保持一致：prefixes 命中且 exclude 未命中"""
    hit = any(p in post["file"] for p in series.get("prefixes", []))
    if hit and any(e in post["file"] for e in series.get("exclude", [])):
        hit = False
    return hit


def main():
    cfg = yaml.safe_load(open(YML, encoding="utf-8"))
    series = cfg["series"]
    cats_cfg = {c["name"]: c for c in cfg["categories"]}
    links = {(l["from"], l["to"]) for l in cfg.get("links", [])}
    posts = load_posts()

    # 1) 系列篇数 + 归属
    belong = collections.defaultdict(list)
    for p in posts:
        hits = [s["id"] for s in series if match(p, s)]
        for h in hits:
            belong[h].append(p)
    print("=" * 68)
    print("① 系列篇数（首页图谱圆圈里的数字，构建时自动统计）")
    print("=" * 68)
    node_series = [s for s in series if s.get("node") is not False]
    for s in node_series:
        print("   %-12s %3d 篇   %s" % (s["id"], len(belong[s["id"]]), s["category"]))
    for s in series:
        if s.get("node") is False:
            print("   %-12s %3d 篇   %s  (单篇，不进图谱)" % (s["id"], len(belong[s["id"]]), s["category"]))
    print("   图谱节点覆盖 %d / %d 篇" % (sum(len(belong[s["id"]]) for s in node_series), len(posts)))

    # 2) 未命中任何系列
    orphans = [p for p in posts if not any(match(p, s) for s in series)]
    print()
    print("=" * 68)
    print("② 未归入任何系列的文章（%d）" % len(orphans))
    print("=" * 68)
    if orphans:
        for p in orphans:
            print("   ✗ %s  [%s]" % (p["file"], "/".join(p["categories"])))
        print("   -> 需要在 knowledge_graph.yml 的 series[].prefixes 里补前缀规则")
    else:
        print("   ✓ 全部文章都有系列归属")

    # 3) 未配置的分类
    real_cats = collections.Counter()
    for p in posts:
        for c in p["categories"]:
            real_cats[c] += 1
    unknown = [c for c in real_cats if c not in cats_cfg]
    print()
    print("=" * 68)
    print("③ 站点分类 vs yml 配置")
    print("=" * 68)
    for c, n in real_cats.most_common():
        flag = "✓" if c in cats_cfg else "✗ 未配置（首页走兜底样式）"
        print("   %-10s %3d 篇  %s" % (c, n, flag))

    # 4) 站内互引
    if "--links" in sys.argv:
        print()
        print("=" * 68)
        print("④ 系列间站内互引（可作为依赖边的候选）")
        print("=" * 68)
        url2series = {}
        for s in series:
            for p in belong[s["id"]]:
                url2series[p["url"]] = s["id"]
        pair = collections.Counter()
        for p in posts:
            src = url2series.get(p["url"])
            if not src:
                continue
            for r in p["refs"]:
                tgt = url2series.get(r)
                if tgt and tgt != src:
                    pair[(src, tgt)] += 1
        known = {(a, b) for a, b in links}
        for (a, b), n in pair.most_common(30):
            mark = "已有" if (a, b) in known else ("反向已有" if (b, a) in known else "建议新增")
            print("   %-11s -> %-11s  %2d 处引用   %s" % (a, b, n, mark))
        print()
        print("   当前 yml 里 %d 条边；未被任何边连接的系列：%s" % (
            len(links),
            ", ".join(s["id"] for s in node_series
                      if not any(s["id"] in e for e in links)) or "无"))


if __name__ == "__main__":
    main()
