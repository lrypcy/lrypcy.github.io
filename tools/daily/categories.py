#!/usr/bin/env python3
"""分类词表的唯一读取入口（脚本侧）。

词表本体在 `_data/daily_categories.yml`，前端（news.md / _layouts/daily.html）
与本目录的脚本**共用同一份**，顺序即展示顺序。

为什么单独一个模块：pick_topics.py 与 scaffold.py 都要「按词表顺序」输出，
各自读一遍 yml 容易漂移；放这里只解析一次。

为什么 `try: import yaml`：本机 python3 有 PyYAML，但 managed python 不一定有。
缺失时回退到下面的内置表（内容与 yml 一致），脚本照跑，只是不跟随 yml 改动。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATS_PATH = REPO_ROOT / "_data" / "daily_categories.yml"

# 与 _data/daily_categories.yml 保持一致的回退表（改 yml 时同步改这里）
FALLBACK: list[dict] = [
    {"key": "embodied", "label": "具身智能",
     "desc": "VLA、操作与抓取、人形与灵巧手、locomotion、sim-to-real、模仿学习"},
    {"key": "world-model", "label": "世界模型",
     "desc": "world model / WAM、视频生成即模拟、JEPA、神经仿真器、dynamics model"},
    {"key": "rl", "label": "强化学习与后训练",
     "desc": "RL 算法、GRPO / PPO / RLVR / RLHF、奖励建模、信用分配、post-training"},
    {"key": "llm", "label": "大模型与前沿实验室",
     "desc": "模型发布与评测、预训练与推理技术、OpenAI / DeepMind 等一手发布"},
    {"key": "agents", "label": "智能体与超级智能",
     "desc": "agent 与多智能体、工具使用、自我改进、long-horizon、AGI / 超级智能路线"},
    {"key": "infra", "label": "AI Infra",
     "desc": "训练与推理系统、并行与显存、kernel 与量化部署、集群、吞吐与成本"},
    {"key": "auto-driving", "label": "自动驾驶",
     "desc": "端到端、驾驶 VLA、BEV / occupancy、数据闭环与仿真"},
    {"key": "safety", "label": "安全与对齐",
     "desc": "interpretability、alignment、red teaming、评测与治理"},
    {"key": "misc", "label": "交叉与观点",
     "desc": "立场论文、综述、开源工具、其他 AI 进展"},
]


def load_categories() -> list[dict]:
    """返回 [{"key","label","desc"}, ...]，顺序即展示顺序。"""
    try:
        import yaml  # type: ignore
    except Exception:  # noqa: BLE001
        return [dict(c) for c in FALLBACK]
    try:
        data = yaml.safe_load(CATS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [dict(c) for c in FALLBACK]
    if not isinstance(data, list):
        return [dict(c) for c in FALLBACK]
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        k = row.get("key")
        if not k:
            continue
        out.append(
            {"key": str(k), "label": str(row.get("label") or k),
             "desc": str(row.get("desc") or "")}
        )
    return out or [dict(c) for c in FALLBACK]


def cat_keys() -> list[str]:
    return [c["key"] for c in load_categories()]


def cat_label(key: str) -> str:
    for c in load_categories():
        if c["key"] == key:
            return c["label"]
    return key


def sort_by_order(keys: list[str]) -> list[str]:
    """把一批 key 按词表顺序排序并去重（用于 front matter 的 cats:）。"""
    order = cat_keys()
    rank = {k: i for i, k in enumerate(order)}
    uniq = list(dict.fromkeys(keys))
    return sorted(uniq, key=lambda k: rank.get(k, len(rank)))


if __name__ == "__main__":
    for c in load_categories():
        print(f"{c['key']:<14} {c['label']:<12} {c['desc']}")
