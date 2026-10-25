#!/usr/bin/env python3
"""从 raw-*.json 里筛出具身智能 / 世界动作模型 / 强化学习相关的条目。

判定：标题 + 正文前 3000 字做关键词命中，命中带权重（**标题命中权重翻倍**）。
只做「找出相关条目」，不做「值不值得写」的判断 —— 那是写作阶段的事。

用法：
  python3 tools/daily/pick_topics.py
  python3 tools/daily/pick_topics.py --date 2026-10-26
  python3 tools/daily/pick_topics.py --top 10 --by-source
  python3 tools/daily/pick_topics.py --dump-body       # 打印正文摘要供阅读
  python3 tools/daily/pick_topics.py --source hf,arxiv # 只看某些信源
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"

# (关键词, 权重)：标题命中按 2 倍计
TOPICS: list[tuple[str, int]] = [
    # 具身智能 / 机器人
    ("具身", 5), ("embodied", 5), ("机器人", 4), ("robot", 3), ("机械臂", 4),
    ("manipulation", 4), ("locomotion", 4), ("人形", 4), ("humanoid", 4),
    ("VLA", 4), ("vision-language-action", 5), ("端到端自驾", 3), ("自动驾驶", 2),
    ("dexterous", 3), ("灵巧手", 4), ("grasp", 3), ("抓取", 3), ("sim-to-real", 4),
    ("仿真到真实", 4), ("模仿学习", 3), ("imitation learning", 3),
    # 世界模型 / 世界动作模型
    ("世界模型", 6), ("world model", 6), ("世界动作模型", 8),
    ("world action model", 8), ("WAM", 5), ("神经模拟器", 4),
    ("world simulator", 4), ("dynamics model", 3), ("scene flow", 4),
    # 强化学习
    ("强化学习", 5), ("reinforcement learning", 5), ("RLVR", 5),
    ("GRPO", 4), ("PPO", 3), ("policy gradient", 4), ("RLHF", 3),
    ("reward", 3), ("奖励", 3), ("credit assignment", 4), ("信用分配", 4),
    ("post-training", 3), ("后训练", 3), ("在线强化学习", 5),
    ("flow policy", 5), ("flow matching", 3),
    # 训练范式
    ("trajectory", 2), ("轨迹", 2), ("test-time", 2), ("推理时扩展", 2),
]

# 明显不相关的降权
NEGATIVE: list[tuple[str, int]] = [
    ("直播预告", -3), ("招聘", -6), ("秋招", -6), ("实习", -4),
    ("知识星球", -6), ("扫码", -5), ("广告", -6), ("课程", -3),
    ("量化", -2), ("quantization", -2), ("编译", -2), ("编译器", -3),
    ("芯片", -3), ("GPU", -2), ("数据库", -5), ("前端", -4),
]

# 各信源基础分：青稞是中文解读（信息密度高），HF/arXiv 是一手论文
SOURCE_BONUS = {"qingke": 0, "hf": 4, "arxiv": 4, "github": 2, "nvidia": 2}

SOURCE_LABEL = {
    "qingke": "青稞", "hf": "HF Papers", "arxiv": "arXiv",
    "github": "GitHub", "nvidia": "NVIDIA",
}


def score(title: str, body: str, source: str = "") -> tuple[int, list[str]]:
    head = ((title + " ") * 2 + " " + (body or "")[:3000]).lower()
    total = SOURCE_BONUS.get(source, 0)
    hits: list[tuple[str, int]] = []
    for kw, w in TOPICS:
        n = head.count(kw.lower())
        if n:
            v = w * min(n, 3)
            total += v
            hits.append((kw, v))
    for kw, w in NEGATIVE:
        if kw.lower() in head:
            total += w
            hits.append((kw, w))
    hits.sort(key=lambda x: -x[1])
    return total, [f"{k}({v:+d})" for k, v in hits[:7]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-score", type=int, default=8)
    ap.add_argument("--source", default=None, help="逗号分隔，只看这些信源")
    ap.add_argument("--by-source", action="store_true", help="按信源分组输出")
    ap.add_argument("--dump-body", action="store_true")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    path = OUT_DIR / f"raw-{day}.json"
    if not path.exists():
        print(f"[error] 没有 {path}，先跑 fetch_all.py", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    only = set(args.source.split(",")) if args.source else None

    rows = []
    for x in data["items"]:
        if only and x.get("source") not in only:
            continue
        s, hits = score(x["title"], x.get("body", ""), x.get("source", ""))
        rows.append((s, hits, x))
    rows.sort(key=lambda r: -r[0])
    kept = [r for r in rows if r[0] >= args.min_score][: args.top]

    print(f"日期 {day} ｜ 总条目 {len(rows)} ｜ 入选 {len(kept)}（阈值 {args.min_score}）\n")
    for i, (s, hits, x) in enumerate(kept, 1):
        tag = SOURCE_LABEL.get(x.get("source"), x.get("source", "?"))
        extra = f" ⭐{x['upvotes']}" if x.get("upvotes") else ""
        print(f"[{i}] {s:>3}分 [{tag}]{extra} {x.get('published','')[:10]}  {x['title'][:52]}")
        print(f"     {x['url']}")
        if x.get("arxiv"):
            print(f"     arXiv: {x['arxiv']}")
        print(f"     {', '.join(hits)}")
        if args.dump_body:
            print("     摘要：" + (x.get("body") or "")[:700])
        print()

    if args.by_source:
        print("── 按信源统计（全部条目中的相关数）──")
        agg: dict[str, int] = {}
        for s, _, x in rows:
            if s >= args.min_score:
                agg[x.get("source", "?")] = agg.get(x.get("source", "?"), 0) + 1
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {SOURCE_LABEL.get(k,k)}: {v}")

    brief = OUT_DIR / f"brief-{day}.json"
    brief.write_text(
        json.dumps(
            {
                "date": day,
                "picked": [
                    {"score": s, "hits": h, **{k: v for k, v in x.items() if k != "body"}}
                    for s, h, x in kept
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] 候选清单 → {brief}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
