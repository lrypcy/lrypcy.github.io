#!/usr/bin/env python3
"""给候选条目打 1–5 星建议分（两个维度），供写作时参考。

**输出的是建议分，不是终判。** 脚本只能看证据信号（有没有硬数字、有没有论文、
有没有代码、社区热度），看不懂「这个结果重不重要」。最终星级由写作时人工定。

两个维度（用户 2026-10-26 明确要求）：
  - `worth`  值不值得读：读者要不要花时间看这条
  - `impact` 领域影响力：这条对领域冲击多大

评分依据（可核验的信号，不是感觉）：

| 信号 | 说明 |
| --- | --- |
| 有量化数字 | 正文中出现 百分比 / pp / × / fps / ms 等，且不是作者自吹 |
| 有 arXiv 论文 | 有 `arxiv` 字段或正文含 arxiv.org 链接 |
| 有开源代码 | 有 `github` 字段或正文含 github.com 链接 |
| 社区热度 | HF upvotes / GitHub stars |
| 有配图 | 有 `images`（写作时能上图，读者更容易判断） |
| 有对比基线 | 正文出现「相比 / vs / baseline / 超过」——说明做了对照 |
| 立场论文 / 综述 | 无实验数字，天然低 worth（除非是重量级路线判断） |

用法：
  python3 tools/daily/rate.py                    # 给 brief-*.json 的候选打分
  python3 tools/daily/rate.py --date 2026-10-26
  python3 tools/daily/rate.py --all              # 给 raw 全部条目打分（不只看候选）
  python3 tools/daily/rate.py --top 20 --by-category  # 按分类聚合打印

分类（`category` / `category_label`）由 pick_topics.py 打好，这里只**透传**到
rated-*.json，供 scaffold.py 按分类分节；本脚本不重新打标。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import categories as C  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

STAR = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}

# 硬数字的证据：百分比、百分点、倍数、速度、时延、参数量
# Infra 类条目常报吞吐 / 时延 / 成本，所以 tokens/s、GB/s、TFLOPS、MFU、加速比
# 都要认 —— 以前只认 % 和 ms，vLLM / TensorRT 这类会全被判成「没有数字」。
NUM_PAT = re.compile(
    r"(\d+(?:\.\d+)?\s*%)"                    # 45.2%
    r"|(\d+(?:\.\d+)?\s*(?:pp|个百分点))"        # 7.9 pp
    r"|(\d+(?:\.\d+)?\s*(?:×|x)\s)"            # 3.5×
    r"|(\d+(?:\.\d+)?\s*(?:fps|FPS))"          # 20 fps
    r"|(\d+(?:\.\d+)?\s*ms\b)"                 # 107.4 ms
    r"|(\d+(?:\.\d+)?\s*(?:s|sec|seconds)\b)"   # 12.4 s
    r"|(\d+(?:\.\d+)?\s*B\s+(?:参数|param))"    # 1.7B 参数
    r"|(\d+(?:\.\d+)?\s*dB)"                   # 15.25 dB
    # --- Infra 形态 ---
    r"|(\d+(?:\.\d+)?\s*(?:tokens?|tok)/s\b)"   # 2400 tokens/s
    r"|(\d+(?:\.\d+)?\s*(?:tok|tokens?)\s*/?\s*(?:per\s*)?sec)"  # 2400 tokens/sec
    r"|(\d+(?:\.\d+)?\s*(?:GB|MB|TB)/s\b)"      # 3.2 GB/s
    r"|(\d+(?:\.\d+)?\s*(?:GB|MB|TB|GiB|GiB)\b)"  # 80GB（显存 / 带宽）
    r"|(\d+(?:\.\d+)?\s*(?:TFLOPS|TFLOP/s|FLOPS|PFLOPS)\b)"  # 1.4 TFLOPS
    r"|(\d+(?:\.\d+)?\s*MFU\b)"                # 42% MFU / 0.42 MFU
    r"|(\d+(?:\.\d+)?\s*(?:×|x)\s*(?:加速|speedup))"  # 3.2× 加速
    r"|(speedup\s*(?:of\s*)?\d+(?:\.\d+)?)"     # speedup of 3.2
    r"|(\d+(?:\.\d+)?\s*(?:req|requests)/s\b)"  # 120 req/s
    r"|(\d+(?:\.\d+)?\s*(?:QPS|qps)\b)"        # 800 QPS
    r"|(\$\s*\d+(?:\.\d+)?)"                   # 成本 $0.42
    r"|(\d+(?:\.\d+)?\s*(?:USD|美元|万元|亿元))",   # 成本
    re.I,
)
COMPARE_PAT = re.compile(
    r"(相比|较之|vs\.?|versus|baseline|超过|优于| outperform| outperform| higher than"
    r"| state[- ]of[- ]the[- ]art| SOTA| 最高| 最好| 首次)",
    re.I,
)
POSITION_PAT = re.compile(
    r"(position paper|立场论文|survey|综述|roadmap|路线图|展望|call for|perspective)",
    re.I,
)
# 直播预告：只看标题。正文里青稞每篇都挂「欢迎投稿…」模板，
# 那是站点 footer 噪声不是内容，不能拿来降权。
PROMO_PAT = re.compile(r"(直播预告|live stream|直播时间|webinar)", re.I)


def _evidence(x: dict) -> dict:
    body = (x.get("body") or "") + " " + (x.get("excerpt") or "")
    title = x.get("title") or ""
    blob = title + " " + body
    nums = NUM_PAT.findall(body)
    return {
        "has_number": bool(nums),
        "n_numbers": len(nums),
        "has_arxiv": bool(x.get("arxiv")) or "arxiv.org" in blob.lower(),
        "has_code": bool(x.get("github")) or "github.com" in blob.lower(),
        "has_image": bool(x.get("images")),
        "has_compare": bool(COMPARE_PAT.search(blob)),
        "is_position": bool(POSITION_PAT.search(title))
        or bool(POSITION_PAT.search(body[:800])),
        "is_promo": bool(PROMO_PAT.search(title)),
        "heat": x.get("upvotes") or 0,
    }


def rate_worth(e: dict) -> int:
    """值不值得读：读者花 3 分钟看这条能拿回什么。"""
    s = 1.5  # 基线：能进候选说明至少相关
    if e["n_numbers"] >= 5:
        s += 1.0
    elif e["n_numbers"] >= 2:
        s += 0.7
    elif e["has_number"]:
        s += 0.4
    if e["has_compare"]:
        s += 0.6
    if e["has_arxiv"]:
        s += 0.5
    if e["has_code"]:
        s += 0.4
    if e["heat"] >= 50:
        s += 0.8
    elif e["heat"] >= 20:
        s += 0.4
    if e["is_position"]:
        s -= 0.6
    if e["is_promo"]:
        s -= 1.5  # 直播预告 / 投稿引导，几乎没有阅读价值
    if not e["has_number"] and not e["has_arxiv"]:
        s -= 0.5  # 既没数字也没论文，多半是观点文
    return max(1, min(5, round(s)))


def rate_impact(e: dict) -> int:
    """领域影响力：这条对领域的冲击。"""
    s = 1.5
    if e["has_arxiv"]:
        s += 0.6
    if e["has_code"]:
        s += 0.7  # 开源的东西才可能被别人接着做
    if e["has_number"] and e["has_compare"]:
        s += 0.8  # 有对照的量化提升
    elif e["has_number"]:
        s += 0.3
    if e["heat"] >= 50:
        s += 0.8
    elif e["heat"] >= 20:
        s += 0.4
    if e["is_position"]:
        s += 0  # 立场论文可能很有影响力，但脚本判断不了，留给人工
    if e["is_promo"]:
        s -= 0.5
    return max(1, min(5, round(s)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--all", action="store_true", help="给 raw 全部条目打分")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--by-category", action="store_true", help="按分类分组打印（便于挑条目）")
    ap.add_argument("--categories", default=None, help="逗号分隔，只看这些分类（key）")
    args = ap.parse_args()

    CATS = C.load_categories()
    CAT_KEYS = [c["key"] for c in CATS]
    CAT_LABEL = {c["key"]: c["label"] for c in CATS}
    want_cats = None
    if args.categories:
        want_cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        bad = [c for c in want_cats if c not in CAT_KEYS]
        if bad:
            print(f"[error] 未知分类 {bad}；可用：{', '.join(CAT_KEYS)}", file=sys.stderr)
            return 1

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    src_path = OUT_DIR / (f"raw-{day}.json" if args.all else f"brief-{day}.json")
    if not src_path.exists():
        print(f"[error] 没有 {src_path}", file=sys.stderr)
        return 1
    data = json.loads(src_path.read_text(encoding="utf-8"))
    items = data["items"] if args.all else data.get("picked", [])

    rows = []
    for x in items:
        # 分类由 pick_topics.py 打好，这里透传；缺了就归 misc（--all 跑 raw 时会发生）
        cat = x.get("category") or "misc"
        if cat not in CAT_LABEL:
            cat = "misc"
        if want_cats and cat not in want_cats:
            continue
        e = _evidence(x)
        label = CAT_LABEL.get(cat, cat)
        rows.append((rate_worth(e), rate_impact(e), e, x, cat, label))
    rows.sort(key=lambda r: (-(r[0] + r[1]), -(r[2]["heat"] or 0)))

    def _sig(e: dict) -> str:
        sig = [
            ("数字" if e["has_number"] else None),
            (f"数字×{e['n_numbers']}" if e["n_numbers"] >= 3 else None),
            ("论文" if e["has_arxiv"] else None),
            ("代码" if e["has_code"] else None),
            ("有图" if e["has_image"] else None),
            ("对照" if e["has_compare"] else None),
            ("立场/综述" if e["is_position"] else None),
            ("预告/推广" if e["is_promo"] else None),
            (f"热度{e['heat']}" if e["heat"] else None),
        ]
        return ' '.join('·' + s for s in sig if s)

    print(f"日期 {day} ｜ {len(rows)} 条 ｜ worth=值不值得读，impact=领域影响力\n")

    # 每类占比：全局 top N 会被最大的一类（通常是 agents/github）占满，
    # 分类视图的意义正是「每类都能看到」，所以每类给一个上限。
    per_cat = max(1, args.top // max(1, len(CAT_KEYS)))
    if args.by_category:
        grouped: dict[str, list] = {}
        for r in rows:  # rows 已按 (worth+impact, heat) 排好
            g = grouped.setdefault(r[4], [])
            if len(g) < per_cat:
                g.append(r)
        for key in CAT_KEYS:
            if key not in grouped:
                continue
            print(f"### {CAT_LABEL.get(key, key)}（{key}）— {len(grouped[key])} 条"
                  f"（每类上限 {per_cat}）\n")
            for w, im, e, x, cat, label in grouped[key]:
                tag = x.get("source", "?")
                print(f"  worth {STAR[w]}  impact {STAR[im]}  [{tag}] {x['title'][:44]}")
                print(f"       {x['url'][:100]}")
                print(f"       证据：{_sig(e)}")
                print()
    else:
        for w, im, e, x, cat, label in rows[: args.top]:
            tag = x.get("source", "?")
            print(f"worth {STAR[w]}  impact {STAR[im]}  [{tag}] ({label}) {x['title'][:42]}")
            print(f"     证据：{_sig(e)}")
            print()

    out = OUT_DIR / f"rated-{day}.json"
    out.write_text(
        json.dumps(
            {
                "date": day,
                "categories": CAT_KEYS,
                "rated": [
                    {"worth": w, "impact": im, "evidence": e,
                     "category": cat, "category_label": label,
                     **{k: v for k, v in x.items() if k != "body"}}
                    for w, im, e, x, cat, label in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
