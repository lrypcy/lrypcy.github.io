#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱体检 / 同步助手
------------------------------------------------------------
回答一个问题：站点新增/修改文章后，_data/knowledge_graph.yml 需不需要手工改？

用法：
    python3 tools/graph_sync.py                # 完整体检报告
    python3 tools/graph_sync.py --links        # 额外输出「站内互引」建议的依赖关系
    python3 tools/graph_sync.py --quiet        # 只打印问题与结论（供 git hook 使用）
    python3 tools/graph_sync.py --strict       # 有未处理项时 exit 1
    python3 tools/graph_sync.py --suggest-yml  # 打印可直接粘贴的 yml 片段（新分类 / 建议边）

输出四件事：
  1. 每个系列当前命中篇数（与首页图谱圆圈里的数字一致）
  2. 未命中任何系列的文章  -> 需要在 yml 里加前缀规则，否则它不会出现在图谱里
  3. site 分类中未在 yml 配置的分类 -> 首页有兜底显示，但建议补元信息（图标/颜色/描述）
  4. 系列之间的站内互引统计 -> 建议新增/加强的依赖边（--links）

退出码：
  0  没有待处理项（或只在非 --strict 下提示）
  1  有 --strict 未通过的待处理项
  2  配置文件本身有问题（yml 解析失败 / 依赖边指向不存在的系列）
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
    argv = sys.argv[1:]
    quiet = "--quiet" in argv
    strict = "--strict" in argv or os.environ.get("GRAPH_SYNC_STRICT") == "1"
    want_links = "--links" in argv or quiet or "--suggest-yml" in argv
    suggest = "--suggest-yml" in argv

    try:
        cfg = yaml.safe_load(open(YML, encoding="utf-8"))
    except Exception as e:  # yml 本身坏了，必须停下来
        print("✗ _data/knowledge_graph.yml 解析失败：%s" % e)
        return 2

    series = cfg["series"]
    cats_cfg = {c["name"]: c for c in cfg["categories"]}
    links = [(l["from"], l["to"]) for l in cfg.get("links", [])]
    known = set(links)
    ids = {s["id"] for s in series}
    bad = [(a, b) for a, b in links if a not in ids or b not in ids]
    if bad:
        print("✗ knowledge_graph.yml 里的依赖边指向了不存在的系列：%s" % bad)
        return 2

    posts = load_posts()
    blockers = []    # 需要人处理：strict 模式下会阻断提交
    suggestions = []  # 仅供参考：建议补的依赖边

    # 1) 系列篇数 + 归属
    belong = collections.defaultdict(list)
    for p in posts:
        for h in [s["id"] for s in series if match(p, s)]:
            belong[h].append(p)
    node_series = [s for s in series if s.get("node") is not False]

    if not quiet:
        print("=" * 68)
        print("① 系列篇数（首页图谱圆圈里的数字，构建时自动统计）")
        print("=" * 68)
        for s in node_series:
            print("   %-12s %3d 篇   %s" % (s["id"], len(belong[s["id"]]), s["category"]))
        for s in series:
            if s.get("node") is False:
                print("   %-12s %3d 篇   %s  (单篇，不进图谱)" % (s["id"], len(belong[s["id"]]), s["category"]))
        print("   图谱节点覆盖 %d / %d 篇" % (sum(len(belong[s["id"]]) for s in node_series), len(posts)))

    # 2) 未命中任何系列
    orphans = [p for p in posts if not any(match(p, s) for s in series)]
    if not quiet:
        print()
        print("=" * 68)
        print("② 未归入任何系列的文章（%d）" % len(orphans))
        print("=" * 68)
        for p in orphans:
            print("   ✗ %s  [%s]" % (p["file"], "/".join(p["categories"]) or "无分类"))
        if not orphans:
            print("   ✓ 全部文章都有系列归属")
    for p in orphans:
        blockers.append("文章未归入任何系列：%s（需在 series[].prefixes 补前缀规则）" % p["file"])

    # 3) 未配置的分类
    real_cats = collections.Counter()
    for p in posts:
        for c in p["categories"]:
            real_cats[c] += 1
    unknown = [c for c in real_cats if c not in cats_cfg]
    if not quiet:
        print()
        print("=" * 68)
        print("③ 站点分类 vs yml 配置")
        print("=" * 68)
        for c, n in real_cats.most_common():
            flag = "✓" if c in cats_cfg else "✗ 未配置（首页走兜底样式）"
            print("   %-10s %3d 篇  %s" % (c, n, flag))
    for c in unknown:
        blockers.append("分类未在 yml 配置：%s（%d 篇，首页兜底显示但缺图标/颜色/描述）" % (c, real_cats[c]))

    # 4) 站内互引 -> 建议边 + 孤立节点
    new_edges = []
    if want_links:
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
        isolated = [s["id"] for s in node_series if not any(s["id"] in e for e in known)]
        if not quiet:
            print()
            print("=" * 68)
            print("④ 系列间站内互引（可作为依赖边的候选）")
            print("=" * 68)
            for (a, b), n in pair.most_common(30):
                mark = "已有" if (a, b) in known else ("反向已有" if (b, a) in known else "建议新增")
                print("   %-11s -> %-11s  %2d 处引用   %s" % (a, b, n, mark))
            print()
            print("   当前 yml 里 %d 条边；未被任何边连接的系列：%s"
                  % (len(links), ", ".join(isolated) or "无"))
        for (a, b), n in pair.most_common(30):
            if (a, b) in known or (b, a) in known:
                continue
            # 方向统一为「层号小的 -> 层号大的」，同层则按引用方向
            la = next((s.get("layer", 9) for s in series if s["id"] == a), 9)
            lb = next((s.get("layer", 9) for s in series if s["id"] == b), 9)
            src, tgt = (a, b) if la <= lb else (b, a)
            new_edges.append((src, tgt, n))
        for sid in isolated:
            blockers.append("系列孤立在图谱里没有任何连线：%s" % sid)
        for src, tgt, n in new_edges:
            if n >= 2:
                suggestions.append("建议补边：%s -> %s（文章互引 %d 处）" % (src, tgt, n))

    # 结论
    print()
    if quiet:
        print("── 知识图谱体检 ──")
    else:
        print("=" * 68)
        print("体检结论")
        print("=" * 68)
    if not blockers and not suggestions:
        print("✓ 图谱与文章完全同步，无需改动 _data/knowledge_graph.yml")
    for b in blockers:
        print("  ✗ " + b)
    for s in suggestions:
        print("  · " + s)
    total = len(blockers) + len(suggestions)
    print("  共 %d 项待处理（%d 项需处理 / %d 项建议）" % (total, len(blockers), len(suggestions)))

    if suggest and (unknown or new_edges):
        print()
        print("=" * 68)
        print("可粘贴到 _data/knowledge_graph.yml 的片段")
        print("=" * 68)
        if unknown:
            print("# 放在 categories: 下")
            for c in unknown:
                print('  - name: %s\n    icon: "📌"\n    color: "#64748b"\n    desc: TODO：%d 篇，补一句定位' % (c, real_cats[c]))
        if new_edges:
            print("# 放在 links: 下（label 请改成有信息量的依赖关系说明）")
            for src, tgt, n in new_edges:
                print("  - { from: %s, to: %s, label: %d 处互引，待补说明 }" % (src, tgt, n))

    if blockers and strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
