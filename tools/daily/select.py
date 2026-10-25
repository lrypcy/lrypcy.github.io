#!/usr/bin/env python3
"""从 raw-*.json 里筛出具身智能 / 世界动作模型 / 强化学习相关的条目。

判定方式：标题 + 正文前 3000 字做关键词命中，命中带权重，输出候选排序。
只做「找出相关条目」，不做「值不值得写」的判断 —— 那是写作阶段的事。

用法：
  python3 tools/daily/select.py                       # 默认处理今天的 raw
  python3 tools/daily/select.py --date 2026-10-26
  python3 tools/daily/select.py --top 6 --dump-body   # 输出正文摘要供阅读
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"

# (关键词, 权重)：标题命中权重翻倍
TOPICS: list[tuple[str, int]] = [
    # 具身智能 / 机器人
    ("具身", 5), ("embodied", 5), ("机器人", 4), ("robot", 3), ("机械臂", 4),
    ("manipulation", 4), ("locomotion", 4), ("人形", 4), ("humanoid", 4),
    ("VLA", 4), ("vision-language-action", 5), ("自动驾驶", 2), ("端到端自驾", 3),
    # 世界模型 / 世界动作模型
    ("世界模型", 6), ("world model", 6), ("世界动作模型", 8),
    ("world action model", 8), ("WAM", 5), ("神经模拟器", 4),
    ("world simulator", 4), ("dynamics model", 3), ("策略评估", 2),
    # 强化学习
    ("强化学习", 5), ("reinforcement learning", 5), ("RLVR", 5),
    ("GRPO", 4), ("PPO", 3), ("policy gradient", 4), ("RLHF", 3),
    ("reward", 3), ("奖励", 3), ("credit assignment", 4), ("信用分配", 4),
    ("RL ", 2), ("post-training", 3), ("后训练", 3), ("在线强化学习", 5),
    ("test-time", 2), ("推理时扩展", 2),
    # 与 RL 强相关的训练范式
    ("蒸馏", 2), ("trajectory", 2), ("轨迹", 2), ("SFT", 1),
    ("sim-to-real", 4), ("仿真到真实", 4), ("模仿学习", 3), ("imitation", 2),
]

# 明显不相关的降权（正文里顺带提了一句的不算）
NEGATIVE: list[tuple[str, int]] = [
    ("直播预告", -3), ("招聘", -6), ("秋招", -6), ("实习", -4),
    ("知识星球", -6), ("扫码", -5), ("广告", -6), ("课程", -3),
]


def score(title: str, body: str) -> tuple[int, list[str]]:
    head = (title + " " + title + " " + body[:3000]).lower()
    total = 0
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
    return total, [f"{k}({v:+d})" for k, v in hits[:8]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--top", type=int, default=8, help="输出前 N 条")
    ap.add_argument("--min-score", type=int, default=6, help="入选最低分")
    ap.add_argument("--dump-body", action="store_true", help="打印正文前 800 字")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    path = OUT_DIR / f"raw-{day}.json"
    if not path.exists():
        print(f"[error] 没有 {path}，先跑 fetch_qingke.py", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for x in data["items"]:
        s, hits = score(x["title"], x["body"])
        rows.append((s, hits, x))
    rows.sort(key=lambda r: -r[0])

    kept = [r for r in rows if r[0] >= args.min_score][: args.top]
    print(f"日期 {day} ｜ 总条目 {len(rows)} ｜ 入选 {len(kept)}（阈值 {args.min_score}）\n")
    for i, (s, hits, x) in enumerate(kept, 1):
        print(f"[{i}] {s:>3} 分  {x['published'][:10]}  {x['title']}")
        print(f"     {x['url']}")
        print(f"     命中：{', '.join(hits)}")
        if args.dump_body:
            print("     正文摘要：" + x["body"][:800].replace("\n", " "))
        print()

    brief = OUT_DIR / f"brief-{day}.json"
    brief.write_text(
        json.dumps(
            {
                "date": day,
                "picked": [
                    {"score": s, "hits": hits, **{k: v for k, v in x.items() if k != "body"}}
                    for s, hits, x in kept
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
