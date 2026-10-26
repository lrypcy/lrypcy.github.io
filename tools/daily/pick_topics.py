#!/usr/bin/env python3
"""从 raw-*.json 里筛出 AI 日报相关条目，并给每条打**主分类**。

**这个脚本不能改名成 `select.py`** —— 会遮蔽标准库 `select`，
本目录所有网络脚本（含 fetch_all.py）都会导入失败。

判定：标题 + 正文前 3000 字做关键词命中，命中带权重（**标题命中权重翻倍**）。
只做「找出相关条目 + 归到哪个分类」，不做「值不值得写」的判断 —— 那是写作阶段的事。

分类词表在 `_data/daily_categories.yml`（与前端共享，顺序即展示顺序；
缺 yaml 模块或文件缺失时回退 `categories.py` 里的内置表）。
每条输出：
  - `category`        主分类 key，如 "embodied"
  - `category_label`  中文标签，如 "具身智能"
  - `category_scores` 九类各自的得分，便于人工复核打标是否离谱

用法：
  python3 tools/daily/pick_topics.py
  python3 tools/daily/pick_topics.py --date 2026-10-26
  python3 tools/daily/pick_topics.py --top 10 --by-source
  python3 tools/daily/pick_topics.py --top 20 --by-category      # 按分类分组打印
  python3 tools/daily/pick_topics.py --categories embodied,infra # 只看指定分类
  python3 tools/daily/pick_topics.py --source hf,arxiv           # 只看某些信源
  python3 tools/daily/pick_topics.py --dump-body                 # 打印正文摘要供阅读
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

# 每类一套关键词：(关键词, 权重)。标题命中按 2 倍计。
# 词表 key 必须与 _data/daily_categories.yml 对齐；不认识的 key 会被忽略。
TOPICS: dict[str, list[tuple[str, int]]] = {
    "embodied": [
        ("具身", 5), ("embodied", 5), ("机器人", 4), ("robot", 3), ("机械臂", 4),
        ("manipulation", 4), ("locomotion", 4), ("人形", 4), ("humanoid", 4),
        ("VLA", 4), ("vision-language-action", 5), ("dexterous", 3), ("灵巧手", 4),
        ("grasp", 3), ("抓取", 3), ("sim-to-real", 4), ("仿真到真实", 4),
        ("模仿学习", 3), ("imitation learning", 3), ("teleoperation", 3),
        ("遥操作", 3), ("动作专家", 4), ("action expert", 4), ("flow policy", 4),
        ("四足", 3), ("quadruped", 3), ("navigation", 2), ("导航", 2),
    ],
    "world-model": [
        ("世界模型", 6), ("world model", 6), ("世界动作模型", 8),
        ("world action model", 8), ("WAM", 5), ("神经模拟器", 4),
        ("world simulator", 4), ("dynamics model", 3), ("scene flow", 4),
        ("视频生成", 3), ("video generation", 3), ("JEPA", 4),
        ("next-frame", 3), ("未来帧", 3), ("3D 场景", 2), ("occupancy", 2),
    ],
    "rl": [
        ("强化学习", 5), ("reinforcement learning", 5), ("RLVR", 5),
        ("GRPO", 4), ("PPO", 3), ("policy gradient", 4), ("RLHF", 3),
        ("reward", 3), ("奖励", 3), ("credit assignment", 4), ("信用分配", 4),
        ("post-training", 3), ("后训练", 3), ("在线强化学习", 5),
        ("reward model", 4), ("奖励模型", 4), ("verifier", 2), ("验证器", 2),
        ("flow matching", 3), ("trajectory", 2), ("轨迹", 2),
        ("test-time", 2), ("推理时扩展", 2), ("self-play", 3), ("自我博弈", 3),
    ],
    "llm": [
        ("大模型", 4), ("LLM", 3), ("large language model", 4), ("语言模型", 3),
        ("GPT", 3), ("Claude", 3), ("Gemini", 3), ("Llama", 3), ("Qwen", 3),
        ("DeepSeek", 3), ("预训练", 3), ("pretraining", 3), ("pre-training", 3),
        ("tokenizer", 2), ("词表", 2), ("上下文", 2), ("context window", 3),
        ("长上下文", 3), ("long context", 3), ("MoE", 3), ("混合专家", 3),
        ("attention", 2), ("注意力", 2), ("distillation", 3), ("蒸馏", 3),
        ("multimodal", 2), ("多模态", 2), ("benchmark", 2), ("评测", 2),
        ("frontier model", 4), ("前沿模型", 4), ("model release", 3),
    ],
    "agents": [
        ("智能体", 5), ("agent", 4), ("agentic", 5), ("多智能体", 5),
        ("multi-agent", 5), ("工具调用", 4), ("tool use", 4), ("tool calling", 4),
        ("long-horizon", 4), ("长程", 3), ("planning", 2), ("规划", 2),
        ("self-improving", 5), ("自我改进", 5), ("超级智能", 6),
        ("superintelligence", 6), ("AGI", 5), ("ASI", 5), ("通用人工智能", 5),
        ("computer use", 4), ("memory", 2), ("记忆", 2), ("orchestration", 3),
        ("coding agent", 5), ("编程智能体", 5), ("MCP", 3), ("skills", 2),
    ],
    "infra": [
        ("推理", 3), ("inference", 3), ("serving", 4), ("vLLM", 5), ("SGLang", 5),
        ("TensorRT", 4), ("吞吐", 4), ("throughput", 4), ("时延", 3),
        ("latency", 3), ("KV cache", 5), ("KV 缓存", 5), ("speculative decoding", 6),
        ("投机解码", 6), ("投机采样", 5), ("quantization", 4), ("量化", 3),
        ("编译", 3), ("compiler", 3), ("kernel", 4), ("CUDA", 4), ("Triton", 4),
        ("GPU", 2), ("芯片", 2), ("显存", 4), ("memory", 1), ("并行", 3),
        ("parallelism", 3), ("分布式训练", 4), ("distributed training", 4),
        ("MFU", 5), ("TFLOPS", 4), ("算力", 3), ("集群", 3), ("cluster", 3),
        ("成本", 2), ("cost", 1), ("调度", 2), ("scheduler", 2), ("部署", 3),
        ("Ray", 3), ("Kubernetes", 2), ("加速比", 4), ("speedup", 4),
    ],
    "auto-driving": [
        ("自动驾驶", 5), ("autonomous driving", 5), ("端到端自驾", 6),
        ("end-to-end driving", 6), ("自驾", 4), ("self-driving", 5),
        ("BEV", 4), ("occupancy network", 4), ("占用网络", 4), ("数据闭环", 4),
        ("闭环仿真", 4), ("驾驶", 3), ("driving", 3), ("智驾", 4),
        ("泊车", 3), ("planning", 1), ("轨迹预测", 3), ("Waymo", 3),
        ("Tesla FSD", 4), ("FSD", 3), ("城市 NOA", 4), ("NOA", 3),
    ],
    "safety": [
        ("对齐", 5), ("alignment", 5), ("安全", 3), ("safety", 3),
        ("可解释", 4), ("interpretability", 5), ("mechanistic", 4),
        ("red team", 4), ("红队", 4), ("越狱", 4), ("jailbreak", 4),
        ("幻觉", 3), ("hallucination", 3), ("governance", 3), ("治理", 3),
        ("eval", 1), ("评估", 1), ("refusal", 3), ("拒答", 3),
        ("monitoring", 2), ("监管", 2), ("risk", 2), ("风险", 2),
        ("sycophancy", 4), ("谄媚", 4), ("deception", 4), ("欺骗", 4),
    ],
    "misc": [
        ("综述", 3), ("survey", 3), ("立场", 3), ("position paper", 4),
        ("观点", 2), ("评论", 2), ("路线图", 3), ("roadmap", 3),
        ("开源", 2), ("open source", 2), ("工具", 1), ("数据集", 2),
        ("dataset", 2), ("教程", 2), ("tutorial", 2), ("展望", 2),
    ],
}

# 明显不相关的降权。
# 注意：量化 / 编译 / GPU / 芯片 / 数据库 现在**不再是噪声**——
# AI Infra 正是兴趣方向，压掉它们会漏掉 vLLM / TensorRT / kernel 类进展。
# 这里只留真正的广告 / 招聘 / 直播预告 / 知识星球类。
NEGATIVE: list[tuple[str, int]] = [
    ("直播预告", -3), ("招聘", -6), ("秋招", -6), ("实习", -4), ("内推", -5),
    ("知识星球", -6), ("扫码", -5), ("广告", -6), ("课程", -3), ("报名", -3),
    ("抽奖", -5), ("加微信", -5), ("商务合作", -4), ("带货", -5),
]

# 各信源基础分。
# 青稞是中文深度解读（信息密度最高，但已是二手）；HF/arXiv 是一手论文；
# 前沿实验室官方博客是一手发布；媒体源噪声大，给 0 或负分压一压。
SOURCE_BONUS = {
    "qingke": 0,
    "qbitai": -2,          # 中文 AI 媒体，快讯多、深度少
    "hf": 4,
    "arxiv": 4,
    "github": 2,
    "nvidia": 2,
    "deepmind": 3,
    "bair": 3,
    "hf_blog": 2,
    "openai": 3,
    "google_research": 3,
    "msr": 2,
    "vllm": 4,             # Infra 一手，含金量高
    "together": 2,
    "aws_ml": 0,           # 云厂商博客，教程 / 产品向居多
    "pytorch": 2,
    "ieee_robotics": 1,
    "robohub": 1,
    "marktechpost": 0,
    "techcrunch_ai": -2,   # 融资 / 产品新闻为主
    "arstechnica_ai": -1,
    "interconnects": 1,
    "raschka": 2,
    "lesswrong": 0,        # 观点流，靠关键词命中本身入选，不给基础分
}

SOURCE_LABEL = {
    "qingke": "青稞", "qbitai": "量子位", "hf": "HF Papers", "arxiv": "arXiv",
    "github": "GitHub", "nvidia": "NVIDIA", "deepmind": "DeepMind", "bair": "BAIR",
    "hf_blog": "HF Blog", "openai": "OpenAI", "google_research": "Google Research",
    "msr": "MSR", "vllm": "vLLM", "together": "Together", "aws_ml": "AWS ML",
    "pytorch": "PyTorch", "ieee_robotics": "IEEE Spectrum", "robohub": "Robohub",
    "marktechpost": "MarkTechPost", "techcrunch_ai": "TechCrunch AI",
    "arstechnica_ai": "Ars Technica", "interconnects": "Interconnects",
    "raschka": "Raschka", "lesswrong": "LessWrong",
}

# 英文源的降权词：融资、收购、产品发布一般不是研究进展
NEGATIVE_EN: list[tuple[str, int]] = [
    ("funding", -4), ("raises", -4), ("series a", -5), ("series b", -5),
    ("acquisition", -4), ("acquires", -4), ("ipo", -4), ("valuation", -4),
    ("partnership", -3), ("toy", -6), ("christmas", -6), ("gift", -5),
    ("podcast", -3), ("webinar", -4), ("newsletter", -3),
    ("job", -5), ("hiring", -5), ("conference", -2), ("award", -3),
]

CATS = C.load_categories()
CAT_KEYS = [c["key"] for c in CATS]
CAT_LABEL = {c["key"]: c["label"] for c in CATS}


def _kw_score(text: str, kw: str, w: int) -> int:
    """标题命中权重翻倍：score() 里把标题重复两遍实现。"""
    n = text.count(kw.lower())
    return w * min(n, 3) if n else 0


def classify(title: str, body: str, source: str = "") -> tuple[str, str, dict, int, list[str]]:
    """返回 (category_key, category_label, category_scores, total, hits)。

    category_scores 是九类各自的原始命中分（不含信源基础分、不含负分）。

    total 用**九类命中分之和** + 信源基础分 + 负分词，与改造前的单一 TOPICS 表
    行为一致。这里刻意**不**用「最高分类分」：跨领域条目（具身 + RL、世界模型 +
    Infra）原来靠多类命中累加拿到高分，改成只取最大值会让它们的总分掉下来，
    同一个 --min-score 阈值下会被静默筛掉 —— 分类是打标，不该顺手降低召回。
    """
    head = ((title + " ") * 2 + " " + (body or "")[:3000]).lower()
    scores: dict[str, int] = {}
    hits: list[tuple[str, int]] = []
    for key in CAT_KEYS:
        s = 0
        for kw, w in TOPICS.get(key, []):
            v = _kw_score(head, kw, w)
            if v:
                s += v
                hits.append((f"{kw}@{key}", v))
        scores[key] = s
    best = max(CAT_KEYS, key=lambda k: scores.get(k, 0))
    if scores.get(best, 0) <= 0:
        best = "misc"

    relevance = sum(scores.values())
    total = relevance + SOURCE_BONUS.get(source, 0)
    for kw, w in NEGATIVE + NEGATIVE_EN:
        if kw.lower() in head:
            total += w
            hits.append((kw, w))

    hits.sort(key=lambda x: -x[1])
    return best, CAT_LABEL.get(best, best), scores, total, [f"{k}({v:+d})" for k, v in hits[:7]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-score", type=int, default=8)
    ap.add_argument("--source", default=None, help="逗号分隔，只看这些信源")
    ap.add_argument("--categories", default=None, help="逗号分隔，只看这些分类（key）")
    ap.add_argument("--by-source", action="store_true", help="按信源分组输出")
    ap.add_argument("--by-category", action="store_true", help="按分类分组输出")
    ap.add_argument("--dump-body", action="store_true")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    path = OUT_DIR / f"raw-{day}.json"
    if not path.exists():
        print(f"[error] 没有 {path}，先跑 fetch_all.py", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    only = set(args.source.split(",")) if args.source else None
    want_cats = None
    if args.categories:
        want_cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        bad = [c for c in want_cats if c not in CAT_KEYS]
        if bad:
            print(f"[error] 未知分类 {bad}；可用：{', '.join(CAT_KEYS)}", file=sys.stderr)
            return 1

    rows = []
    for x in data["items"]:
        if only and x.get("source") not in only:
            continue
        cat, label, scores, s, hits = classify(x["title"], x.get("body", ""), x.get("source", ""))
        if want_cats and cat not in want_cats:
            continue
        rows.append((s, hits, x, cat, label, scores))
    rows.sort(key=lambda r: -r[0])
    kept = [r for r in rows if r[0] >= args.min_score][: args.top]

    def _print_item(i, s, hits, x, cat, label):
        tag = SOURCE_LABEL.get(x.get("source"), x.get("source", "?"))
        extra = f" ⭐{x['upvotes']}" if x.get("upvotes") else ""
        print(f"[{i}] {s:>3}分 [{tag}]{extra} {x.get('published','')[:10]}  {x['title'][:52]}")
        print(f"     分类：{label}（{cat}）")
        print(f"     {x['url']}")
        if x.get("arxiv"):
            print(f"     arXiv: {x['arxiv']}")
        print(f"     {', '.join(hits)}")
        if args.dump_body:
            print("     摘要：" + (x.get("body") or "")[:700])
        print()

    if args.by_category:
        print(f"日期 {day} ｜ 总条目 {len(rows)} ｜ 入选 {len(kept)}（阈值 {args.min_score}）\n")
        print("── 按分类分组 ──")
        grouped: dict[str, list] = {}
        for r in kept:
            grouped.setdefault(r[3], []).append(r)
        print()
        for key in CAT_KEYS:
            if key not in grouped:
                continue
            print(f"### {CAT_LABEL.get(key, key)}（{key}）— {len(grouped[key])} 条\n")
            for i, (s, hits, x, cat, label, scores) in enumerate(grouped[key], 1):
                _print_item(i, s, hits, x, cat, label)
        # 分类分布（全部命中阈值的，不只 top N）
        print("── 分类分布（全部达标条目）──")
        dist: dict[str, int] = {}
        for s, _, _, cat, _, _ in rows:
            if s >= args.min_score:
                dist[cat] = dist.get(cat, 0) + 1
        for k in CAT_KEYS:
            if dist.get(k):
                print(f"  {CAT_LABEL.get(k,k)}（{k}）: {dist[k]}")
    else:
        print(f"日期 {day} ｜ 总条目 {len(rows)} ｜ 入选 {len(kept)}（阈值 {args.min_score}）\n")
        for i, (s, hits, x, cat, label, scores) in enumerate(kept, 1):
            _print_item(i, s, hits, x, cat, label)

    if args.by_source:
        print("── 按信源统计（全部条目中的相关数）──")
        agg: dict[str, int] = {}
        for s, _, x, *_ in rows:
            if s >= args.min_score:
                agg[x.get("source", "?")] = agg.get(x.get("source", "?"), 0) + 1
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {SOURCE_LABEL.get(k,k)}: {v}")

    brief = OUT_DIR / f"brief-{day}.json"
    # brief 里若没有正文，回 raw 取 —— 评分要看正文里的 arXiv / github 链接与数字
    body_by_url: dict[str, str] = {}
    raw_path = OUT_DIR / f"raw-{day}.json"
    if raw_path.exists():
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            for it in raw.get("items", []):
                if it.get("url"):
                    body_by_url[it["url"]] = (it.get("body") or "") + " " + (
                        it.get("excerpt") or ""
                    )
        except Exception:  # noqa: BLE001
            pass

    brief.write_text(
        json.dumps(
            {
                "date": day,
                "categories": [c["key"] for c in CATS],
                "picked": [
                    {
                        "score": s,
                        "hits": h,
                        "category": cat,
                        "category_label": label,
                        "category_scores": scores,
                        "body": (x.get("body") or "")[:1200]
                        or body_by_url.get(x.get("url", ""), "")[:1200],
                        **{k: v for k, v in x.items() if k != "body"},
                    }
                    for s, h, x, cat, label, scores in kept
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
