#!/usr/bin/env python3
"""AI 日报统一采集器：汇总多个信源到一个 raw json。

信源（2026-10-26 实测，共 24 个；多 feed 的已按名字合并去重）：
   中文 / 论文
     ✅ qingke     青稞社区（Halo API + 正文抓取）—— 中文解读主力
     ✅ hf         HuggingFace Daily Papers（arXiv id / 摘要 / github / upvotes）
     ✅ arxiv      arXiv API（具身 / 世界模型 / RL / AI Infra / 自驾 / agentic）—— 走直连，见 _get
     ✅ github     GitHub 仓库搜索（按 push 时间找活跃项目）
   中文 AI 媒体
     ✅ qbitai     量子位
   前沿实验室 / 厂商官方（一手发布）
     ✅ nvidia     NVIDIA Developer Blog + blogs.nvidia.com（2 feed 合并）
     ✅ deepmind   Google DeepMind Blog
     ✅ bair       BAIR Blog（Berkeley）
     ✅ hf_blog    HuggingFace Blog（与 hf 论文流不同，是官方发布 / 教程）
     ✅ openai     OpenAI News
     ✅ google_research  Google Research Blog
     ✅ msr        Microsoft Research
   英文机器人 / 科技媒体（AI 向）
     ✅ ieee_robotics   IEEE Spectrum Robotics
     ✅ robohub         Robohub
     ✅ marktechpost    MarkTechPost
     ✅ techcrunch_ai   TechCrunch AI
     ✅ arstechnica_ai  Ars Technica AI
   英文 AI Infra
     ✅ vllm       vLLM Blog
     ✅ together   Together AI Blog
     ✅ aws_ml     AWS Machine Learning Blog
     ✅ pytorch    PyTorch Blog
   英文观点 / 深度
     ✅ interconnects  Interconnects（Nathan Lambert）
     ✅ raschka        Sebastian Raschka 的 Magazine
     ✅ lesswrong      LessWrong（AGI / 超级智能路线讨论）

实测不可用，**别加**：Anthropic(404)、ai.meta.com(400)、机器之心 RSS(返回 HTML)、
Synced(停在 2025)、SemiAnalysis(停在 2025)、Anyscale(0 item)、Modal(404)、
新智元(不通)、InfoQ(405)、ImportAI(不通)、知乎(匿名 403)、
therobotreport(403)、venturebeat(429)。

网络：**必须走本机代理 127.0.0.1:7890**（ClashX）。系统代理设置里就有，
直连 HF / NVIDIA / 知乎全部不通。代理挂了本脚本会大量失败 —— 先看 §排查。

输出：out/raw-<YYYY-MM-DD>.json
  {"date", "fetched_at", "sources": {name: {...}}, "items": [...]}
  items 统一字段：{source, title, url, published, excerpt, body}

用法：
  python3 tools/daily/fetch_all.py                    # 全部信源
  python3 tools/daily/fetch_all.py --only hf,arxiv    # 只跑指定信源
  python3 tools/daily/fetch_all.py --days 7
  python3 tools/daily/fetch_all.py --no-body          # 跳过正文抓取（快）
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"

# ---------------------------------------------------------------- 网络

PROXY = os.environ.get("DAILY_PROXY", "http://127.0.0.1:7890")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _opener():
    if PROXY:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
        )
    return urllib.request.build_opener()


OP = _opener()


_OP_DIRECT = None


def _get(url: str, timeout: int = 25, retries: int = 3, headers: dict | None = None,
         proxy: bool = True):
    """proxy=False 走直连。

    export.arxiv.org **必须直连**：本机 ClashX 代理会在它身上反复
    `[Errno 54] Connection reset by peer` / SSL 握手超时，9 条检索式能整批挂掉；
    绕开代理同一个瞬间就拿到 85 条（2026-10-28 实测）。
    手册原来写「arXiv 走代理更稳」是反的，已一并改正。
    """
    global _OP_DIRECT
    if proxy:
        opener = OP
    else:
        if _OP_DIRECT is None:
            _OP_DIRECT = urllib.request.build_opener()
        opener = _OP_DIRECT
    h = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}
    if headers:
        h.update(headers)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=h)
            with opener.open(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                # 指数退避：arXiv 的 reset 常常要等几秒才恢复
                time.sleep(2 ** i * 1.5)
    raise RuntimeError(str(last))


def _text(s: str) -> str:
    s = re.sub(r"<(script|style)\b.*?</\1>", " ", s, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def _b64_bing_url(u: str) -> str:
    """Bing 结果链接是 /ck/a?...&u=<base64>，解出真实地址。"""
    m = re.search(r"[?&]u=([^&]+)", u)
    if not m:
        return u
    s = m.group(1)
    for skip in (2, 0, 1):
        try:
            d = base64.b64decode(s[skip:] + "==").decode("utf-8")
            if d.startswith("http"):
                return d
        except Exception:  # noqa: BLE001
            continue
    return u


# ---------------------------------------------------------------- 信源


def src_qingke(days: int, size: int, no_body: bool) -> list[dict]:
    API = "https://qingkeai.online/apis/api.content.halo.run/v1alpha1/posts"
    BASE = "https://qingkeai.online"
    try:
        data = json.loads(_get(f"{API}?page=1&size={size}"))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 青稞列表失败：{e}", file=sys.stderr)
        return []
    items = []
    for it in data.get("items", []):
        sp = it.get("spec", {})
        if not sp.get("publish", True) or sp.get("deleted"):
            continue
        slug = sp.get("slug")
        if not slug:
            continue
        items.append(
            {
                "source": "qingke",
                "title": (sp.get("title") or "").strip(),
                "url": f"{BASE}/blog/{slug}",
                "published": (sp.get("publishTime") or "")[:19].replace("T", " "),
                "excerpt": _text((sp.get("excerpt") or {}).get("raw", "")),
                "body": "",
            }
        )
    items.sort(key=lambda x: x["published"], reverse=True)
    items = [x for x in items if x["published"] >= _cutoff(days)]
    for x in items:
        x["images"] = []
        if no_body:
            continue
        try:
            page = _get(x["url"])
            m = re.search(r"<article\b.*?</article>", page, flags=re.S | re.I)
            frag = m.group(0) if m else ""
            x["body"] = _text(frag)
            x["images"] = _extract_images(frag, x["url"])
        except Exception:  # noqa: BLE001
            x["body"] = ""
        time.sleep(0.4)
    return items


def src_hf(days: int) -> list[dict]:
    try:
        data = json.loads(_get("https://huggingface.co/api/daily_papers?limit=60", timeout=30))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] HF 失败：{e}", file=sys.stderr)
        return []
    cut = _cutoff(days)
    items = []
    for p in data:
        paper = p.get("paper") or {}
        pid = paper.get("id")
        if not pid:
            continue
        pub = (p.get("publishedAt") or paper.get("publishedAt") or "")[:19]
        pub = pub.replace("T", " ")
        if pub and pub < cut:
            continue
        summ = (p.get("summary") or paper.get("summary") or "").strip()
        ai = (paper.get("ai_summary") or "").strip()
        kw = paper.get("ai_keywords") or []
        gh = paper.get("githubRepo")
        items.append(
            {
                "source": "hf",
                "title": (p.get("title") or paper.get("title") or "").strip(),
                "url": f"https://huggingface.co/papers/{pid}",
                "arxiv": f"https://arxiv.org/abs/{pid}",
                "published": pub,
                "upvotes": paper.get("upvotes", 0),
                "github": gh,
                "excerpt": summ[:400],
                "body": (summ + "\n" + ai)[:4000],
                "keywords": kw,
                # HF 的 thumbnail 是论文首页自动生成的预览图，可直接当配图
                "images": (
                    [{"url": p["thumbnail"], "alt": (paper.get("title") or "")[:120],
                      "w": None, "h": None}]
                    if p.get("thumbnail")
                    else []
                ),
            }
        )
    items.sort(key=lambda x: -(x.get("upvotes") or 0))
    return items


# arXiv 检索式：具身 / 世界模型 / RL / AI Infra / 自动驾驶端到端 / agentic / 后训练
# 每条 max_results=12，检索式之间 sleep 3s（arXiv 官方要求请求间隔）。
ARXIV_QUERIES = [
    # 具身智能
    'abs:"vision-language-action" OR abs:"VLA" OR abs:"embodied"',
    'abs:"sim-to-real" OR abs:"imitation learning" OR abs:"policy gradient"',
    'abs:"reinforcement learning" AND (abs:"robot" OR abs:"manipulation")',
    # 世界模型 / 世界动作模型
    'abs:"world model" OR ti:"world action model" OR abs:"world simulator"',
    # AI Infra：推理 / 投机解码 / KV cache / 服务化
    'abs:"LLM inference" OR abs:"speculative decoding" OR abs:"KV cache"',
    'abs:"LLM serving" OR abs:"inference throughput" OR abs:"distributed inference"',
    # 自动驾驶端到端
    # AND / OR 混写必须显式加括号：arXiv 的解析对优先级并不直观，
    # 不写括号会退化成「OR 全放行」，捞回一堆和驾驶无关的 trajectory planning。
    'abs:"end-to-end autonomous driving" OR (abs:"trajectory planning" AND abs:"driving")',
    # agentic / 多智能体
    'abs:"LLM agent" OR abs:"multi-agent" OR abs:"tool use"',
    # RLHF / 奖励模型 / 后训练
    'abs:"RLHF" OR abs:"reward model" OR abs:"post-training"',
]


def src_arxiv(days: int, per_query: int = 12) -> list[dict]:
    import xml.etree.ElementTree as ET

    cut = _cutoff(days)
    seen: dict[str, dict] = {}
    for q in ARXIV_QUERIES:
        url = (
            "https://export.arxiv.org/api/query?search_query="
            + urllib.parse.quote(q)
            + f"&start=0&max_results={per_query}"
            + "&sortBy=submittedDate&sortOrder=descending"
        )
        # export.arxiv.org 经常间歇性 Connection reset / 502（不是我们的错）。
        # 加重试次数 + 退避，仍失败则整源降级（HF 源已覆盖大部分论文）。
        try:
            raw = _get(url, timeout=35, retries=4, proxy=False)   # arXiv 必须直连，见 _get 注释
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] arXiv 检索失败（{q[:30]}…）：{e}", file=sys.stderr)
            continue
        try:
            root = ET.fromstring(raw)
        except Exception:  # noqa: BLE001
            continue
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in root.findall("a:entry", ns):
            idu = (e.findtext("a:id", "", ns) or "").strip()
            if not idu:
                continue
            pub = (e.findtext("a:published", "", ns) or "")[:19].replace("T", " ")
            if pub and pub < cut:
                continue
            seen[idu] = {
                "source": "arxiv",
                "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(),
                "url": idu,
                "arxiv": idu,
                "published": pub,
                "upvotes": None,
                "github": None,
                "excerpt": re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()[:400],
                "body": re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip(),
                "keywords": [],
                "images": [],  # arXiv API 不给图；需要时取 arxiv.org/abs 页的首图
            }
        time.sleep(3)  # arXiv 官方要求请求间隔
    return list(seen.values())


GITHUB_QUERIES = [
    # 具身 / 世界模型
    "embodied OR robotics OR VLA",
    "world model OR world action model",
    # RL 基础设施
    "reinforcement learning infrastructure",
    # AI Infra：推理 / 训练系统
    "llm inference OR inference engine OR kv cache",
    "distributed training OR gpu kernel OR quantization",
    # 自动驾驶端到端
    "autonomous driving end-to-end OR bev perception",
    # agent
    "llm agent framework OR multi-agent",
]


def src_github(days: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    items = []
    for q in GITHUB_QUERIES:
        url = (
            "https://api.github.com/search/repositories?q="
            + urllib.parse.quote(f"{q} pushed:>{since}")
            + "&sort=stars&order=desc&per_page=8"
        )
        try:
            data = json.loads(
                _get(url, headers={"Accept": "application/vnd.github+json"})
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] GitHub 搜索失败：{e}", file=sys.stderr)
            continue
        for r in data.get("items", []):
            topics = r.get("topics") or []
            items.append(
                {
                    "source": "github",
                    "title": r.get("full_name", ""),
                    "url": r.get("html_url", ""),
                    "published": (r.get("pushed_at") or "")[:19].replace("T", " "),
                    "upvotes": r.get("stargazers_count", 0),
                    "github": r.get("html_url", ""),
                    "excerpt": (r.get("description") or "")[:300],
                    "body": (r.get("description") or "") + " topics: " + ", ".join(topics[:10]),
                    "keywords": topics[:8],
                    # GitHub 的 social preview 图（仓库卡片），可当配图
                    "images": (
                        [{"url": f"https://opengraph.githubassets.com/1/{r.get('full_name','')}",
                          "alt": r.get("full_name", ""), "w": None, "h": None}]
                        if r.get("full_name")
                        else []
                    ),
                }
            )
    items.sort(key=lambda x: -(x.get("upvotes") or 0))
    return items


def _extract_images(fragment: str, base_url: str, limit: int = 4) -> list[dict]:
    """从正文 HTML 片段里挑候选配图。

    只收「看起来是内容图」的：有 alt 说明、或在 figure 里、或尺寸够大。
    排除头像 / 二维码 / 站点 UI / 表情。
    """
    out: list[dict] = []
    seen: set[str] = set()

    # figure 内的 img 优先（通常是论文插图）
    fig_imgs = []
    for fig in re.findall(r"<figure\b.*?</figure>", fragment, flags=re.S | re.I):
        fig_imgs += re.findall(r"<img\b[^>]*>", fig, flags=re.I)
    plain_imgs = [
        i for i in re.findall(r"<img\b[^>]*>", fragment, flags=re.I) if i not in fig_imgs
    ]

    for tag in fig_imgs + plain_imgs:
        src = _img_attr(tag, "src") or _img_attr(tag, "data-src")
        if not src:
            continue
        src = urllib.parse.urljoin(base_url, src)
        if src in seen:
            continue
        alt = _img_attr(tag, "alt") or ""
        w = _to_int(_img_attr(tag, "width"))
        h = _to_int(_img_attr(tag, "height"))
        low = (src + " " + alt).lower()
        # 排除噪声图
        if any(
            k in low
            for k in (
                "avatar", "qrcode", "qr_", "icon", "logo", "emoji", "badge",
                "button", "banner", "placeholder", "loading", "spinner",
                "wx", "wechat", "zhihu", "xiaohongshu", "bilibili", "video-number",
            )
        ):
            continue
        # 太小的多半是装饰
        if w and w < 300:
            continue
        if not alt and not (w and w >= 600):
            continue
        seen.add(src)
        out.append({"url": src, "alt": alt[:200], "w": w, "h": h})
        if len(out) >= limit:
            break
    return out


def _img_attr(tag: str, name: str) -> str:
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', tag, flags=re.I) or re.search(
        rf"{name}\s*=\s*'([^']*)'", tag, flags=re.I
    )
    return html.unescape(m.group(1)).strip() if m else ""


def _to_int(s) -> int | None:
    try:
        return int(str(s).strip().rstrip("px"))
    except Exception:  # noqa: BLE001
        return None


# --- 通用 RSS/Atom 源 -------------------------------------------------
# 每条：(信源名, feed URL)。英文源是补国内看不到的官方一手信息与业界动态的。
RSS_FEEDS: list[tuple[str, str]] = [
    # NVIDIA（两个 feed 有重叠，末尾统一去重）
    ("nvidia", "https://developer.nvidia.com/blog/feed/"),
    ("nvidia", "https://blogs.nvidia.com/feed/"),
    # 前沿实验室 / 厂商官方博客（一手发布）
    ("deepmind", "https://deepmind.google/blog/rss.xml"),
    ("bair", "https://bair.berkeley.edu/blog/feed.xml"),
    ("hf_blog", "https://huggingface.co/blog/feed.xml"),
    ("openai", "https://openai.com/news/rss.xml"),
    ("google_research", "https://research.google/blog/rss/"),
    ("msr", "https://www.microsoft.com/en-us/research/feed/"),
    # 英文：机器人垂直媒体 + 科技媒体 AI 版
    ("ieee_robotics", "https://spectrum.ieee.org/feeds/topic/robotics.rss"),
    ("robohub", "https://robohub.org/feed/"),
    ("marktechpost", "https://www.marktechpost.com/feed/"),
    ("techcrunch_ai", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("arstechnica_ai", "https://arstechnica.com/ai/feed/"),
    # 英文：AI Infra（推理 / 服务 / 训练系统）
    ("vllm", "https://vllm.ai/blog/rss.xml"),
    ("together", "https://www.together.ai/blog/rss.xml"),
    ("aws_ml", "https://aws.amazon.com/blogs/machine-learning/feed/"),
    ("pytorch", "https://pytorch.org/blog/feed.xml"),
    # 中文 AI 媒体
    ("qbitai", "https://www.qbitai.com/feed"),
    # 英文：深度观点 / 超级智能讨论
    ("interconnects", "https://www.interconnects.ai/feed"),
    ("raschka", "https://magazine.sebastianraschka.com/feed"),
    ("lesswrong", "https://www.lesswrong.com/feed.xml"),
]


def _parse_feed(raw: str, source: str, cut: str) -> list[dict]:
    """RSS <item> 与 Atom <entry> 通吃。"""
    entries = re.findall(r"<entry>(.*?)</entry>", raw, flags=re.S)
    if not entries:
        entries = re.findall(r"<item>(.*?)</item>", raw, flags=re.S)
    out = []
    for e in entries:

        def g(tag: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", e, flags=re.S)
            if not m:
                return ""
            v = re.sub(r"^<!\[CDATA\[|\]\]>$", "", m.group(1).strip())
            return _text(v)

        link_m = re.search(r'<link[^>]*href="([^"]+)"', e) or re.search(
            r"<link>(.*?)</link>", e, flags=re.S
        )
        url = link_m.group(1) if link_m else ""
        pub = _norm_date(g("published") or g("pubDate") or g("updated"))
        if pub and pub < cut:
            continue
        title = g("title")
        if not title:
            continue
        raw_body = g("summary") or g("content") or g("description")
        body = raw_body[:3000]
        # feed 里常带 <img>（enclosure / media:content / 正文内嵌）
        imgs = _extract_images(e, url)
        if not imgs:
            enc = re.search(r'<enclosure[^>]*url="([^"]+)"[^>]*type="image', e, flags=re.I)
            if enc:
                imgs = [{"url": enc.group(1), "alt": title, "w": None, "h": None}]
        out.append(
            {
                "source": source,
                "title": title,
                "url": url,
                "published": pub,
                "upvotes": None,
                "github": None,
                "excerpt": body[:300],
                "body": body,
                "keywords": [],
                "images": imgs,
            }
        )
    return out


def src_rss(name: str, urls: list[str]):
    def run(days: int, no_body: bool = False) -> list[dict]:
        cut = _cutoff(days)
        items: list[dict] = []
        for f in urls:
            try:
                raw = _get(f, timeout=30)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {name} feed 失败（{f}）：{e}", file=sys.stderr)
                continue
            got = _parse_feed(raw, name, cut)
            # 抓到 200 但一条都没解析出来，有两种完全不同的原因，得分开报：
            #   - 返回的根本不是 feed（反爬页 / 登录墙 / 网关错误页）→ 源坏了，要修
            #   - 是 feed，只是窗口内没有新条目 → 正常（BAIR 常年如此）
            # 不区分的话，源被反爬挡掉会一直静默返回 0 条，跟「今天没更新」长得一样。
            if not got:
                n_blocks = len(re.findall(r"<item\b|<entry\b", raw, flags=re.I))
                head = raw[:200].lstrip()
                looks_feed = head.startswith("<?xml") or "<rss" in head.lower() or "<feed" in head.lower()
                if n_blocks == 0 or not looks_feed:
                    print(
                        f"  [warn] {name} 返回的不是可解析的 feed"
                        f"（可能反爬 / 登录墙，前 60 字：{head[:60]!r}）",
                        file=sys.stderr,
                    )
                else:
                    print(f"  [info] {name}：feed 正常，{days} 天窗口内无新条目", file=sys.stderr)
            items += got
        seen, out = set(), []
        for x in items:
            k = x["title"][:60].lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return run


# ---------------------------------------------------------------- 工具


def _cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _norm_date(s: str) -> str:
    s = (s or "").strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:25].rstrip("Z"), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            continue
    return s[:19].replace("T", " ")


SOURCES = {
    "qingke": lambda days, nb: src_qingke(days, 40, nb),
    "hf": lambda days, nb: src_hf(days),
    "arxiv": lambda days, nb: src_arxiv(days),
    "github": lambda days, nb: src_github(days),
}
# RSS 源按名字聚合（nvidia 有两个 feed）
for _name in dict.fromkeys(n for n, _ in RSS_FEEDS):
    SOURCES[_name] = src_rss(_name, [u for n, u in RSS_FEEDS if n == _name])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--only", default=None, help="逗号分隔，只跑指定信源")
    ap.add_argument("--no-merge", action="store_true", help="--only 时不保留上次其他源的数据")
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-body", action="store_true", help="跳过青稞正文抓取")
    args = ap.parse_args()

    if PROXY:
        # 先探一下代理是否活着，挂了就早说，别让每个源各自超时
        try:
            _get("https://api.github.com", timeout=10, retries=1)
        except Exception as e:  # noqa: BLE001
            print(
                f"[error] 代理 {PROXY} 不通（{e}）。HF/NVIDIA/arXiv 全部依赖它。\n"
                f"        确认 ClashX 在跑；不带代理运行： DAILY_PROXY= python3 ...",
                file=sys.stderr,
            )
            return 1

    # 源会间歇性 reset（arXiv / IEEE 尤其），重跑单个源时不能把别的源冲掉。
    # 默认按 source 合并：本次抓到的覆盖同名的旧数据，其他源原样保留。
    all_items: list[dict] = []
    stats: dict[str, int] = {}
    names = args.only.split(",") if args.only else list(SOURCES)
    if args.only and not args.no_merge:
        prev = OUT_DIR / f"raw-{args.date or datetime.now().strftime('%Y-%m-%d')}.json"
        if prev.exists():
            try:
                old = json.loads(prev.read_text(encoding="utf-8"))
                all_items += [x for x in old.get("items", []) if x.get("source") not in names]
                if all_items:
                    print(f"（保留上次其他源 {len(all_items)} 条）")
            except Exception:  # noqa: BLE001
                pass

    for n in names:
        if n not in SOURCES:
            print(f"[warn] 未知信源 {n}", file=sys.stderr)
            continue
        print(f"→ {n} ...", flush=True)
        try:
            got = SOURCES[n](args.days, args.no_body)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {n} 失败：{e}", file=sys.stderr)
            got = []
        stats[n] = len(got)
        print(f"  {n}: {len(got)} 条")
        all_items += got

    all_items.sort(key=lambda x: (x.get("published") or ""), reverse=True)
    day = args.date or datetime.now().strftime("%Y-%m-%d")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": day,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": args.days,
        "stats": stats,
        "items": all_items,
    }
    out = OUT_DIR / f"raw-{day}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] {out}  合计 {len(all_items)} 条  {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
