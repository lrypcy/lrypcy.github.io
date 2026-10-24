#!/usr/bin/env python3
"""体检文章 TL;DR 块：格式、字数、断言强度、证据出处。

为什么需要这个工具
------------------
TL;DR 是读者唯一必读的 30 秒。它出问题的方式不是"写得短"，而是三种：

  1. **格式漂移**——同一站里并存 `> **TL;DR**`、`## TL;DR`、`> **TL;DR 三连**`、
     裸 `**TL;DR**` 四种写法。`## TL;DR` 尤其坏：`_layouts/post.html` 的目录脚本
     只抓 `h2`，它会变成正文章节；裸 `**TL;DR**` 紧跟列表时还会和后面段落粘连。
  2. **断言过强**——"全部 / 唯一 / 必然 / 必须 / 完全 / 已经收敛"把某个实验设置下的
     观察升级成领域定律，读者没法判断适用性。
  3. **数字没有出处**——"SNR 3.02 dB" 之类的实测值不写条件（模型 / 位宽 / 粒度 /
     样本量），换台机器、换个校准集就不成立，读者无法复核。

所以本工具只做可机械判定的部分，判断不了的不猜：

  - 格式：是否 `> **TL;DR**` + `> *` 列表
  - 字数：TL;DR 块内汉字总数（默认期望 120–260）
  - 绝对化词：命中且**同句没有条件限定词**才报
  - 数字出处：带量纲的数字所在条目里，必须有链接 / 表号 / 章节号 / 实验目录名
  - 必备槽位：缺"边界/代价"槽位的提示（这是本轮 TL;DR 改造的重点）

用法
----
    python3 tools/check_tldr.py _posts/xxx.md            # 单篇
    python3 tools/check_tldr.py _posts/*.md --summary    # 批量，只打问题篇
    python3 tools/check_tldr.py _posts/*.md --strict     # 有 ERROR 则退出码 1

退出码：默认恒为 0（只报告）。--strict 下有 ERROR 时返回 1。

规范条文见 `.agents/WRITING.md` §9。
"""

import argparse
import glob
import re
import sys

# ---------- 绝对化词 ----------
ABSOLUTE_WORDS = [
    "全部", "唯一", "必然", "必须", "完全", "一定",
    "已经收敛", "毫无疑问", "不可能", "绝对", "彻底",
]

# 同句出现这些词，说明绝对化断言已经带了条件，不报
CONDITION_MARKERS = [
    "在", "当", "若", "仅当", "只", "前提", "条件", "设定", "配置",
    "本文", "本篇", "实验", "论文", "测试", "范围内", "下成立",
]

# ---------- 数字出处 ----------
# 带量纲的数字（dB / % / 倍 / 码字 / 样本数…）才算"实测数字"
NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:dB|%|倍|ms|GB|MB|个码字|条样本|步|token)"
)
# 一个条目里出现这些，视为数字有出处
EVIDENCE_MARKERS = [
    "http://", "https://", "§", "表 ", "Table", "arXiv", "README",
    "实验", "results", "图 ", "附录", "篇", "组", "章节",
]

# 边界 / 代价 槽位的关键词
BOUNDARY_MARKERS = [
    "边界", "代价", "条件", "不成立", "失效", "只适用", "不适用",
    "成本", "限制", "注意", "前提",
]

CJK = re.compile(r"[\u4e00-\u9fff]")
LATEX = re.compile(r"\$\$.*?\$\$", re.S)
URL = re.compile(r"\]\([^)]*\)")
INLINE_CODE = re.compile(r"`[^`]*`")

# TL;DR 起始位置的四种历史写法，按"必须迁移"的优先级排列
MARKER_PATTERNS = [
    ("blockquote", re.compile(r"^> \*\*TL;DR\*\*\s*$", re.M)),
    ("blockquote_variant", re.compile(r"^> \*\*TL;DR[^*]*\*\*", re.M)),
    ("h2", re.compile(r"^##+ +TL;DR.*$", re.M)),
    ("bare_bold", re.compile(r"^\*\*TL;DR\*\*\s*$", re.M)),
]


def strip_noise(text):
    """去掉公式、链接、行内代码，只留下真正被读者读的字。"""
    text = LATEX.sub(" ", text)
    text = URL.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    return text


def find_tldr(text):
    """返回 (style, block_lines, start_line)。找不到返回 (None, [], 0)。"""
    head = text[:12000]
    best = None
    for style, pat in MARKER_PATTERNS:
        m = pat.search(head)
        if m and (best is None or m.start() < best[1].start()):
            best = (style, m)
    if best is None:
        return None, [], 0

    style, m = best
    start_line = text.count("\n", 0, m.start()) + 1
    # 块边界：下一个 `---`（front matter / 分隔线）或 `## ` 一级标题
    rest = text[m.end():]
    end = re.search(r"^(?:---|## )", rest, re.M)
    body = rest[:end.start()] if end else rest[:4000]

    lines = []
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过标记行本身（`**TL;DR**` / `## TL;DR`），它不是条目
        if "TL;DR" in stripped and not stripped.startswith(("* ", "- ", "> *", "> -")):
            continue
        if stripped.startswith(">"):
            lines.append(stripped.lstrip("> ").strip())
        elif style in ("bare_bold", "h2", "blockquote_variant"):
            # 老格式：条目可能是 `* x`，也可能是 `**1. 问题：…**` 这样的加粗段落
            lines.append(stripped.lstrip("*- ").strip())
        else:
            break
    return style, lines, start_line


def split_items(lines):
    """把 TL;DR 块拆成"条目"。

    新格式以 `*` 开头的引用行起条目；老格式（`## TL;DR` + 加粗段落）以 `**数字.` 起条目，
    段落内的换行合并进当前条目。
    """
    items, cur = [], None
    new_item = re.compile(r"^(?:[*-] |\*\*\d+\.)")
    for line in lines:
        if not line.strip():
            continue
        if new_item.match(line):
            if cur is not None:
                items.append(cur)
            cur = line.lstrip("*- ").strip()
        elif cur is None:
            cur = line
        else:
            cur += " " + line
    if cur is not None:
        items.append(cur)
    return items


def check_item_absolute(item, raw):
    """同句里没有条件限定词的绝对化词才算问题。"""
    hits = []
    for word in ABSOLUTE_WORDS:
        for m in re.finditer(word, raw):
            sentence_start = max(
                raw.rfind("。", 0, m.start()), raw.rfind("；", 0, m.start()), -1
            )
            sentence_end = min(
                [i for i in (raw.find("。", m.end()), raw.find("；", m.end())) if i != -1]
                or [len(raw)]
            )
            sentence = raw[sentence_start + 1:sentence_end]
            if not any(marker in sentence for marker in CONDITION_MARKERS):
                hits.append(word)
                break
    return hits


def check_file(path, min_cjk=120, max_cjk=260):
    text = open(path, encoding="utf-8").read()
    style, lines, start_line = find_tldr(text)

    report = {"path": path, "style": style, "start_line": start_line,
              "cjk": 0, "items": 0, "errors": [], "warnings": []}

    if style is None:
        report["errors"].append("没有 TL;DR 块（长文应有一个）")
        return report

    items = split_items(lines)
    report["items"] = len(items)
    plain = strip_noise("\n".join(items))
    report["cjk"] = len(CJK.findall(plain))

    if style == "h2":
        # `## TL;DR` 会被 _layouts/post.html 的目录脚本（只抓 h2）当成正文章节
        report["errors"].append(
            "TL;DR 写成 `## TL;DR` 标题——会被文章目录抓成章节，必须改成 `> **TL;DR**`"
        )
    elif style != "blockquote":
        report["warnings"].append(
            f"标记形式是 `{style}`，规范形式是 `> **TL;DR**`（渲染差异小，可批量迁移）"
        )

    if not items:
        report["errors"].append("TL;DR 里没有条目（应为 `> *` 列表）")
        return report

    if report["cjk"] > max_cjk:
        report["warnings"].append(
            f"汉字 {report['cjk']} 超出上限 {max_cjk}，压缩到 3–5 条"
        )
    elif report["cjk"] < min_cjk:
        report["warnings"].append(
            f"汉字 {report['cjk']} 低于下限 {min_cjk}，可能没交代证据或边界"
        )

    if len(items) > 5:
        report["warnings"].append(f"条目 {len(items)} 条，规范上限 5 条")

    # 绝对化词
    for item in items:
        for word in check_item_absolute(item, item):
            report["errors"].append(f"无条件绝对化断言「{word}」：{item[:60]}…")

    # 数字要有出处
    for item in items:
        for m in NUM_UNIT.finditer(strip_noise(item)):
            window = item
            if not any(marker in window for marker in EVIDENCE_MARKERS):
                report["warnings"].append(
                    f"数字「{m.group(0).strip()}」所在条目没有出处标记"
                    f"（链接 / 表号 / 章节号 / 实验目录名）：{item[:50]}…"
                )
                break

    # 边界槽位
    joined = "\n".join(items)
    if not any(marker in joined for marker in BOUNDARY_MARKERS):
        report["warnings"].append("没有「适用边界 / 代价」类内容——读者无法判断何时别用")

    return report


def print_report(report, verbose=True):
    name = report["path"]
    status = "FAIL" if report["errors"] else ("WARN" if report["warnings"] else "OK")
    print(f"\n[{status}] {name}")
    if report["style"] is None:
        for e in report["errors"]:
            print(f"  ERROR {e}")
        return
    print(f"  格式 {report['style']}（第 {report['start_line']} 行）"
          f" ｜ 条目 {report['items']} ｜ 汉字 {report['cjk']}")
    for e in report["errors"]:
        print(f"  ERROR {e}")
    for w in report["warnings"]:
        print(f"  WARN  {w}")


def main():
    ap = argparse.ArgumentParser(description="体检 TL;DR 块（格式/字数/断言/出处）")
    ap.add_argument("paths", nargs="+", help="_posts/*.md 或单篇路径")
    ap.add_argument("--strict", action="store_true", help="有 ERROR 时退出码 1")
    ap.add_argument("--summary", action="store_true", help="只列有问题的篇目")
    ap.add_argument("--min-cjk", type=int, default=120)
    ap.add_argument("--max-cjk", type=int, default=260)
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        matched = glob.glob(p)
        paths.extend(matched or [p])

    reports = [check_file(p, args.min_cjk, args.max_cjk) for p in paths]

    for rep in reports:
        if args.summary and not (rep["errors"] or rep["warnings"]):
            continue
        print_report(rep)

    n_err = sum(1 for r in reports if r["errors"])
    n_warn = sum(1 for r in reports if r["warnings"] and not r["errors"])
    n_ok = len(reports) - n_err - n_warn
    print(f"\n汇总：{len(reports)} 篇 ｜ OK {n_ok} ｜ WARN {n_warn} ｜ FAIL {n_err}")

    if args.strict and n_err:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
