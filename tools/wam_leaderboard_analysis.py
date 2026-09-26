#!/usr/bin/env python3
"""从 Awesome-WAM 抓取论文库 + leaderboard，输出 WAM 论文的分榜成绩与排名。

用法:
    python3 tools/wam_leaderboard_analysis.py [--dump out.json]

数据源（页面是 JS 渲染的，HTML 里没有论文库）:
    https://raw.githubusercontent.com/OpenMOSS/Awesome-WAM/main/README.md
    https://openmoss.ai/Awesome-WAM/leaderboard/data/{leaderboard,benchmarks,coverage,citations}.json
"""
import argparse
import collections
import json
import re
import statistics
import urllib.request

README_URL = "https://raw.githubusercontent.com/OpenMOSS/Awesome-WAM/main/README.md"
LB_URL = "https://openmoss.ai/Awesome-WAM/leaderboard/data/leaderboard.json"
# README 中 "## World Action Model" 到 "## World Model for VLA" 之间为 WAM 论文库
SEC_START = "## World Action Model"
SEC_END = "## World Model for VLA"
TIMEOUT = 60


def fetch(url):
    return urllib.request.urlopen(url, timeout=TIMEOUT).read().decode("utf-8", "ignore")


def parse_papers(md):
    """解析 README 的 WAM 段落，返回 [{name,title,arxiv,venue,section}]"""
    body = md.split(SEC_START, 1)[1].split(SEC_END, 1)[0]
    lines = body.split("\n")
    items, section = [], None
    for i, line in enumerate(lines):
        m = re.match(r"^#{3,4}\s+(.*)", line)
        if m:
            section = m.group(1).strip("# ").strip()
            continue
        m = re.match(r'^- \*\*(.+?)\*\*:\s*"(.*?)"\s*,\s*(.*)$', line)
        if not m:
            continue
        name, title, rest = m.groups()
        # 链接在下一行
        nxt = ""
        for j in range(i + 1, min(i + 3, len(lines))):
            if lines[j].strip().startswith("[["):
                nxt = lines[j]
                break
        ax = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", nxt) or re.search(
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", line
        )
        vm = re.search(r'"\s*,\s*([A-Za-z][A-Za-z0-9\-\s\(\)\.&/]{2,40}?)\s*[.\s]*(!\[|\[\[)', line)
        items.append(
            dict(
                name=name,
                title=title,
                arxiv=ax.group(1) if ax else None,
                venue=vm.group(1).strip() if vm else "",
                section=section,
            )
        )
    return items


def arxiv_id(url):
    m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", url or "")
    return m.group(1) if m else None


def score_of(r):
    """overall_score 缺失时用 suite/task 均值兜底"""
    if r.get("overall_score") is not None:
        return r["overall_score"]
    ss = r.get("suite_scores") or {}
    if ss:
        return statistics.mean(ss.values())
    ts = r.get("task_scores") or {}
    return statistics.mean(ts.values()) if ts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="把完整结果写成 JSON")
    args = ap.parse_args()

    papers = parse_papers(fetch(README_URL))
    lb = json.loads(fetch(LB_URL))
    print(f"论文库 {len(papers)} 篇 | 榜单 {len(lb['results'])} 条 | last_updated {lb['last_updated']}")

    pid = {p["arxiv"]: p for p in papers}
    names = {p["name"].lower(): p["arxiv"] for p in papers}

    def is_wam(r):
        a = arxiv_id(r.get("model_paper"))
        if a in pid:
            return pid[a]["name"]
        dn = (r.get("name_in_paper") or "").lower()
        for nm, aid in names.items():
            if dn == nm or dn.startswith(nm + " ") or dn.startswith(nm + "("):
                return pid[aid]["name"]
        return None

    by_bm = collections.defaultdict(list)
    for r in lb["results"]:
        s = score_of(r)
        if s is not None:
            by_bm[r["benchmark"]].append((s, r))

    wam_rows = collections.defaultdict(list)
    for bm, rows in by_bm.items():
        rows.sort(key=lambda x: -x[0])
        n = len(rows)
        for i, (s, r) in enumerate(rows, 1):
            w = is_wam(r)
            if w:
                wam_rows[w].append(dict(benchmark=bm, score=s, rank=i, total=n,
                                        display=r["display_name"], notes=r.get("notes")))

    for w in sorted(wam_rows, key=lambda k: -len(wam_rows[k])):
        print(f"\n### {w}")
        for row in sorted(wam_rows[w], key=lambda x: x["benchmark"]):
            pct = 100 * row["rank"] / row["total"]
            print(f"  {row['benchmark']:12s} {row['score']:7.2f}  "
                  f"rank {row['rank']:4d}/{row['total']:4d} (top {pct:4.1f}%)  {row['display']}")

    missing = [p["name"] for p in papers if p["name"] not in wam_rows]
    print(f"\n榜内无数据的 WAM 论文 ({len(missing)}): {', '.join(missing)}")

    if args.dump:
        json.dump({"papers": papers, "wam_rows": wam_rows},
                  open(args.dump, "w"), ensure_ascii=False, indent=1)
        print(f"\n已写出 {args.dump}")


if __name__ == "__main__":
    main()
