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
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"

STAR = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}

# 硬数字的证据：百分比、百分点、倍数、速度、时延、参数量
NUM_PAT = re.compile(
    r"(\d+(?:\.\d+)?\s*%)"                    # 45.2%
    r"|(\d+(?:\.\d+)?\s*(?:pp|个百分点))"        # 7.9 pp
    r"|(\d+(?:\.\d+)?\s*(?:×|x)\s)"            # 3.5×
    r"|(\d+(?:\.\d+)?\s*(?:fps|FPS))"          # 20 fps
    r"|(\d+(?:\.\d+)?\s*ms\b)"                 # 107.4 ms
    r"|(\d+(?:\.\d+)?\s*B\s+(?:参数|param))"    # 1.7B 参数
    r"|(\d+(?:\.\d+)?\s*dB)",                  # 15.25 dB
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
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    src_path = OUT_DIR / (f"raw-{day}.json" if args.all else f"brief-{day}.json")
    if not src_path.exists():
        print(f"[error] 没有 {src_path}", file=sys.stderr)
        return 1
    data = json.loads(src_path.read_text(encoding="utf-8"))
    items = data["items"] if args.all else data.get("picked", [])

    rows = []
    for x in items:
        e = _evidence(x)
        rows.append((rate_worth(e), rate_impact(e), e, x))
    rows.sort(key=lambda r: (-(r[0] + r[1]), -(r[2]["heat"] or 0)))

    print(f"日期 {day} ｜ {len(rows)} 条 ｜ worth=值不值得读，impact=领域影响力\n")
    for w, im, e, x in rows[: args.top]:
        tag = x.get("source", "?")
        print(f"worth {STAR[w]}  impact {STAR[im]}  [{tag}] {x['title'][:46]}")
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
        print(f"     证据：{' '.join('·' + s for s in sig if s)}")
        print()

    out = OUT_DIR / f"rated-{day}.json"
    out.write_text(
        json.dumps(
            {
                "date": day,
                "rated": [
                    {"worth": w, "impact": im, "evidence": e,
                     **{k: v for k, v in x.items() if k != "body"}}
                    for w, im, e, x in rows
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
