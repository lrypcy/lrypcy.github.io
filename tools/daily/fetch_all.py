#!/usr/bin/env python3
"""每日资讯统一采集器：汇总多个信源到一个 raw json。

信源（2026-10-26 实测）：
  ✅ qingke     青稞社区（Halo API + 正文抓取）—— 中文解读主力
  ✅ hf         HuggingFace Daily Papers（带 arXiv id / 摘要 / github / upvotes）
  ✅ arxiv      arXiv API（按检索式）
  ✅ github     GitHub 仓库搜索（按 push 时间找活跃项目）
  ✅ nvidia     NVIDIA Developer Blog（Atom feed）
  ❌ zhihu      匿名访问 403（需登录态）；青稞正文里常附作者知乎原文链接，间接覆盖

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


def _get(url: str, timeout: int = 25, retries: int = 3, headers: dict | None = None):
    h = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}
    if headers:
        h.update(headers)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=h)
            with OP.open(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
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
        if no_body:
            continue
        try:
            page = _get(x["url"])
            m = re.search(r"<article\b.*?</article>", page, flags=re.S | re.I)
            x["body"] = _text(m.group(0) if m else "")
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
            }
        )
    items.sort(key=lambda x: -(x.get("upvotes") or 0))
    return items


# arXiv 检索式：具身 / 世界模型 / WAM / RL 四大类
ARXIV_QUERIES = [
    'abs:"world action model" OR abs:"world model" OR ti:"world action model"',
    'abs:"vision-language-action" OR abs:"VLA" OR abs:"embodied"',
    'abs:"reinforcement learning" AND (abs:"robot" OR abs:"manipulation")',
    'abs:"sim-to-real" OR abs:"imitation learning" OR abs:"policy gradient"',
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
        try:
            raw = _get(url, timeout=35)
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
            }
        time.sleep(3)  # arXiv 官方要求请求间隔
    return list(seen.values())


GITHUB_QUERIES = [
    "embodied OR robotics OR VLA",
    "world model OR world action model",
    "reinforcement learning infrastructure",
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
                }
            )
    items.sort(key=lambda x: -(x.get("upvotes") or 0))
    return items


NV_FEEDS = [
    "https://developer.nvidia.com/blog/feed/",
    "https://blogs.nvidia.com/feed/",
]


def src_nvidia(days: int) -> list[dict]:
    cut = _cutoff(days)
    items = []
    for f in NV_FEEDS:
        try:
            raw = _get(f, timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] NVIDIA feed 失败（{f}）：{e}", file=sys.stderr)
            continue
        entries = re.findall(r"<entry>(.*?)</entry>", raw, flags=re.S)
        if not entries:
            entries = re.findall(r"<item>(.*?)</item>", raw, flags=re.S)
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
            pub = g("published") or g("pubDate") or g("updated")
            pub = _norm_date(pub)
            if pub and pub < cut:
                continue
            title = g("title")
            if not title:
                continue
            items.append(
                {
                    "source": "nvidia",
                    "title": title,
                    "url": url,
                    "published": pub,
                    "upvotes": None,
                    "github": None,
                    "excerpt": (g("summary") or g("content"))[:300],
                    "body": (g("summary") or g("content"))[:3000],
                    "keywords": [],
                }
            )
    # 去重（两个 feed 有重叠）
    seen = set()
    out = []
    for x in items:
        k = x["title"][:60]
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


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
    "nvidia": lambda days, nb: src_nvidia(days),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--only", default=None, help="逗号分隔，只跑指定信源")
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

    names = args.only.split(",") if args.only else list(SOURCES)
    all_items: list[dict] = []
    stats: dict[str, int] = {}
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
