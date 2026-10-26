#!/usr/bin/env python3
"""分类归组的日报骨架生成器。

作用：把「写资讯条目」这步半自动化。从 rated-*.json（优先）或 brief-*.json
读候选，按**分类词表顺序**生成 markdown 骨架，人工再精读润色。
这样「同一分类写在一起」不靠人记 —— 骨架已经分好节了。

分节用 raw HTML：`<h2 id="cat-<key>"><label></h2>`。
**不要**用 kramdown 的 `## 标题 {: #id}`：本机 kramdown 2.5.2 会把 `{: #id}`
当字面文字留在标题里，GitHub Pages 远程构建同样不可靠；raw HTML 的 h2 100% 稳定，
CSS 照样命中。

骨架里的星级与 rating 都是**脚本建议值**，写完必须人工复核。

用法：
  python3 tools/daily/scaffold.py                       # 出 out/draft-<今天>.md
  python3 tools/daily/scaffold.py --top 12
  python3 tools/daily/scaffold.py --out out/draft.md
  python3 tools/daily/scaffold.py --publish --force     # 写进 _daily/（默认不写）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import categories as C  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
REPO_ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = REPO_ROOT / "_daily"

STAR = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}


def _md(s: str) -> str:
    """标题里可能有方括号 / 反引号，转义掉避免 markdown 断链。"""
    return (s or "").replace("[", "(").replace("]", ")").replace("`", "'").strip()


def _link(label: str, url: str) -> str:
    return f"[{label}]({url})" if url else ""


def _stars(n) -> str:
    try:
        return STAR[max(1, min(5, int(round(float(n)))))]
    except Exception:  # noqa: BLE001
        return STAR[3]


def build_item(idx: int, x: dict) -> list[str]:
    """一条候选 → 骨架行列表。"""
    lines: list[str] = []
    title = _md(x.get("title") or "(无标题)")
    lines.append(f"### {idx}. {title}")
    lines.append("")
    w = x.get("worth")
    im = x.get("impact")
    mark = ""
    if w and im:
        mark = f"值得读 {_stars(w)} ｜ 影响力 {_stars(im)}"
    else:
        mark = "值得读 ★★★☆☆ ｜ 影响力 ★★★☆☆"
    lines.append(f"`{mark}`")
    lines.append("")

    # 来源 / 论文 / 代码 链接
    bits = []
    if x.get("url"):
        src_label = x.get("source") or "来源"
        heat = x.get("upvotes")
        bits.append(_link(f"{src_label}" + (f"（⭐{heat}）" if heat else ""), x["url"]))
    if x.get("arxiv"):
        pid = str(x["arxiv"]).rstrip("/").split("/")[-1]
        bits.append("论文：" + _link(f"arXiv:{pid}", x["arxiv"]))
    if x.get("github"):
        repo = str(x["github"]).replace("https://github.com/", "").strip("/")
        bits.append("代码：" + _link(repo, x["github"]))
    if bits:
        lines.append("**" + " ｜ ".join(bits) + "**")
        lines.append("")

    # 摘要占位（不编内容，只给位置）
    lines.append("<!-- TODO: 一句话说清它解决了什么问题 -->")
    excerpt = (x.get("excerpt") or x.get("body") or "").strip().replace("\n", " ")
    if excerpt:
        lines.append("")
        lines.append(f"> 采集摘要（写作时替换）：{excerpt[:220]}")

    # 候选配图：取前 1–2 张，没有就不编
    imgs = [i for i in (x.get("images") or []) if i.get("url")][:2]
    if imgs:
        lines.append("")
        for img in imgs:
            alt = _md(img.get("alt") or (x.get("title") or "配图")[:60])
            lines.append(f"![{alt}]({img['url']})")

    lines.append("")
    return lines


def build(days_items: list[dict], day: str, cats: list[dict]) -> tuple[str, dict]:
    """返回 (markdown, {key: 条数})。"""
    order = [c["key"] for c in cats]
    label = {c["key"]: c["label"] for c in cats}
    rank = {k: i for i, k in enumerate(order)}

    buckets: dict[str, list[dict]] = {}
    for x in days_items:
        k = x.get("category") or "misc"
        if k not in rank:
            k = "misc"
        buckets.setdefault(k, []).append(x)

    used_keys = [k for k in order if buckets.get(k)]

    # front matter
    # 占位符**不能带 HTML 注释也不能带冒号**：`title: xxx <!-- TODO: yyy -->`
    # 会让 YAML 报 "mapping values are not allowed here"，GitHub Pages 直接构建失败。
    d = datetime.strptime(day, "%Y-%m-%d")
    title = f"AI 日报 · {d.month}/{d.day}：（待填主线，一句话概括今天）"
    cats_fm = ", ".join(used_keys)
    rating = 3
    vals = []
    for x in days_items:
        for f in ("worth", "impact"):
            v = x.get(f)
            if isinstance(v, (int, float)):
                vals.append(float(v))
    if vals:
        rating = max(1, min(5, int(round(sum(vals) / len(vals)))))

    out: list[str] = [
        "---",
        f"title: {title}",
        f"date: {day}",
        "summary: （待填，一到两句，挑本期最硬的两个数字）",
        f"rating: {rating}",
        f"cats: [{cats_fm}]",
        "---",
        "",
    ]

    n = 0
    for k in used_keys:
        out.append(f'<h2 id="cat-{k}">{label.get(k, k)}</h2>')
        out.append("")
        for x in buckets[k]:
            n += 1
            out += build_item(n, x)
        out.append("")

    counts = {k: len(buckets[k]) for k in used_keys}
    return "\n".join(out).rstrip() + "\n", counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=None, help="默认 out/draft-<date>.md")
    ap.add_argument("--publish", action="store_true",
                    help="显式才写进 _daily/（目标已存在时需 --force）")
    ap.add_argument("--force", action="store_true", help="--publish 时覆盖已存在的文件")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    rated = OUT_DIR / f"rated-{day}.json"
    brief = OUT_DIR / f"brief-{day}.json"
    src_path = rated if rated.exists() else brief
    if not src_path.exists():
        print(f"[error] 没有 {rated} 也没有 {brief}，先跑 pick_topics.py / rate.py",
              file=sys.stderr)
        return 1

    data = json.loads(src_path.read_text(encoding="utf-8"))
    items = data.get("rated") or data.get("picked") or []
    if not items:
        print(f"[error] {src_path} 里没有条目", file=sys.stderr)
        return 1

    # rated 已按 (worth+impact, heat) 排好；brief 按 score 排好。直接取前 N。
    picked = items[: args.top]

    cats = C.load_categories()
    md, counts = build(picked, day, cats)

    if args.publish:
        target = DAILY_DIR / f"{day}-ai-daily.md"
        if target.exists() and not args.force:
            print(f"[error] {target} 已存在，加 --force 覆盖（或换 --out）", file=sys.stderr)
            return 1
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(md, encoding="utf-8")
        out_path = target
    else:
        out_path = Path(args.out) if args.out else OUT_DIR / f"draft-{day}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")

    label = {c["key"]: c["label"] for c in cats}
    print(f"[ok] 骨架 → {out_path}")
    print(f"     来源：{src_path.name}（{src_path.name.startswith('rated') and 'rated' or 'brief'}）")
    print(f"     合计 {len(picked)} 条 ｜ 分类分布：")
    for k, v in counts.items():
        print(f"       {label.get(k,k)}（{k}）：{v} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
