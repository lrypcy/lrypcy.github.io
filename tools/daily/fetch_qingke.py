#!/usr/bin/env python3
"""青稞社区采集器：只做抓取与结构化，不做任何筛选判定。

数据源（Halo 2.x 站点）：
  - 列表：GET /apis/api.content.halo.run/v1alpha1/posts?page=1&size=N
    （注意：带 sort / fieldSelector 参数会 500，只能取默认序再本地排序）
  - 正文：GET /blog/<slug> 的 <article> 片段
  - 兜底：GET /feed.xml（RSS，20 条）

输出：out/raw-<YYYY-MM-DD>.json
  {"date": ..., "fetched_at": ..., "items": [{title, url, published, excerpt, body_len, body}]}

用法：
  python3 tools/daily/fetch_qingke.py              # 抓最近 40 条
  python3 tools/daily/fetch_qingke.py --size 60    # 抓更多
  python3 tools/daily/fetch_qingke.py --days 3     # 只保留 3 天内发布的
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://qingkeai.online"
API = f"{BASE}/apis/api.content.halo.run/v1alpha1/posts"
FEED = f"{BASE}/feed.xml"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
OUT_DIR = Path(__file__).resolve().parent / "out"

# 正文低于此字符数视为抓取失败（页面是 JS 渲染或文章极短）
MIN_BODY = 200


def _get(url: str, timeout: int = 25, retries: int = 3) -> str:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} 失败：{last}")


def _strip_tags(s: str) -> str:
    s = re.sub(r"<(script|style)\b.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def fetch_list(size: int) -> list[dict]:
    raw = _get(f"{API}?page=1&size={size}")
    data = json.loads(raw)
    items = []
    for it in data.get("items", []):
        spec = it.get("spec", {})
        if not spec.get("publish", True) or spec.get("deleted"):
            continue
        slug = spec.get("slug")
        if not slug:
            continue
        items.append(
            {
                "title": (spec.get("title") or "").strip(),
                "url": f"{BASE}/blog/{slug}",
                "published": (spec.get("publishTime") or "")[:19].replace("T", " "),
                "excerpt": _strip_tags((spec.get("excerpt") or {}).get("raw", "")),
            }
        )
    # 默认序不是严格时间序，本地按发布时间倒排
    items.sort(key=lambda x: x["published"], reverse=True)
    return items


def fetch_body(url: str) -> str:
    try:
        page = _get(url)
    except Exception:  # noqa: BLE001
        return ""
    m = re.search(r"<article\b.*?</article>", page, flags=re.S | re.I)
    body = _strip_tags(m.group(0) if m else "")
    # 去掉正文里塞的知乎/公众号导流尾巴之外的噪声：只截到明显结束标记
    body = re.sub(r"\s*相关阅读\s*$", "", body)
    return body


def fetch_feed() -> list[dict]:
    """RSS 兜底：列表 API 挂掉时至少有 20 条可用。"""
    try:
        xml = _get(FEED, timeout=20)
    except Exception:  # noqa: BLE001
        return []
    items = []
    for block in re.findall(r"<item>(.*?)</item>", xml, flags=re.S):

        def g(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.S)
            if not m:
                return ""
            v = m.group(1)
            v = re.sub(r"^<!\[CDATA\[|\]\]>$", "", v.strip())
            return html.unescape(v).strip()

        pub = g("pubDate")
        try:
            dt = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S")
            pub = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            pass
        items.append(
            {"title": g("title"), "url": g("link"), "published": pub, "excerpt": ""}
        )
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=40, help="列表拉取条数")
    ap.add_argument("--days", type=int, default=5, help="只保留最近 N 天发布的")
    ap.add_argument("--date", default=None, help="输出日期标签，默认今天")
    ap.add_argument("--no-body", action="store_true", help="跳过正文抓取（调试用）")
    args = ap.parse_args()

    try:
        items = fetch_list(args.size)
        source = "api"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 列表 API 失败（{e}），回退 RSS", file=sys.stderr)
        items = fetch_feed()
        source = "rss"

    if not items:
        print("[error] 未取到任何条目", file=sys.stderr)
        return 1

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    items = [x for x in items if x["published"] >= cutoff]

    ok = 0
    for x in items:
        if args.no_body:
            x["body"] = ""
            x["body_len"] = 0
            ok += 1
            continue
        x["body"] = fetch_body(x["url"])
        x["body_len"] = len(x["body"])
        if x["body_len"] >= MIN_BODY:
            ok += 1
        time.sleep(0.4)  # 别把人家站点打疼

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": day,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "window_days": args.days,
        "count": len(items),
        "body_ok": ok,
        "items": items,
    }
    out = OUT_DIR / f"raw-{day}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {out}  条目 {len(items)}（正文达标 {ok}）  来源 {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
