#!/usr/bin/env python3
"""扫长中文文档里的乱码词 / 伪词 / 伪 Unicode。

用法:
    python3 tools/scan_garbled.py FILE [FILE ...] [options]

检查四项:
1. 可疑 Unicode —— 非 ASCII、非 CJK、不在符号白名单里的字符 (伪 Unicode / 混入其他文字)
2. 已知伪词黑名单命中
3. 低频长英文词 —— 全文只出现 1 次且长度 > --min-len, 需人工核对是否是伪词
4. 疑似单 $ 行内公式 —— kramdown 只保护 $$...$$, 单 $ 会被 Markdown 二次解析

退出码 0 表示第 1/2/4 项干净 (第 3 项永远需要人工过目)。
"""

import argparse
import re
import sys
import unicodedata
from collections import Counter

# 允许出现在中文技术文档里的非 ASCII 符号
ALLOWED_SYMBOLS = set(
    "，。、；：？！（）《》【】—…·“”‘’「」"
    "×÷→←↔⇒≤≥≠≈≡±∑∏√∫∞⁃–—"
    "αβγδεηθλμπρστφχψωΔΣΩ"  # 希腊字母：η 用于重叠效率，χψ 为常见记号
    "∘"  # 环算子：AR = RS ∘ AG 这类复合写法
    "①②③④⑤⑥⑦⑧⑨⑩"
    "─│├└┌┐┘┬┴┼═║╔╗╚╝｜←→"
    "§▼▲"  # § 章节引用号 / ▼▲ ASCII 流程图箭头：站内多篇文章在用
    "／＋−∈ρ₀₁₂₃₄₅₆₇₈₉"  # 全角斜杠是站内「A ／ B」并列惯例；其余为常用数学符号
    "²³"  # 上标二/三：站内多篇文章用 S²、L² 记平方，非乱码
    "∝"  # 正比于：写「单步时间 ∝ 批量」这类定性关系
    "0123456789%°"
)

# 已知会混进中文文档的伪词黑名单 (大小写不敏感命中)
KNOWN_GARBLED = [
    "Sector", "danshai", "Lyndonvious", "iskeymic", "ischemic",
    "tungsten", "mantle", "buildsi", "rookeeper", "handover",
    "param sewer", "Casa de Papel", "those knobs", "FRAME",
    "Pursuit", "opiremental", "radio ", "科技企业识别",
]


def check_unicode(text):
    hits = Counter()
    for ch in text:
        if ord(ch) < 128:
            continue
        if "\u4e00" <= ch <= "\u9fff":  # CJK 统一汉字
            continue
        if ch in ALLOWED_SYMBOLS:
            continue
        hits[ch] += 1
    return hits


def check_lowfreq(text, min_len):
    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{%d,}\b" % (min_len - 1), text)
    cnt = Counter(words)
    return sorted(w for w, c in cnt.items() if c == 1)


def check_single_dollar(lines):
    """找疑似单 $ 行内公式。代码块（``` 围栏）内部跳过——shell 提示符会误报。"""
    hits = []
    in_code = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        stripped = re.sub(r"`[^`]*`", "", line)  # 去掉行内代码
        if re.search(r"(?<!\$)\$(?!\$)", stripped):
            hits.append((i, line.strip()[:80]))
    return hits


def check_known(text):
    """黑名单命中检查。

    纯 ASCII 单词用词边界匹配，避免子串误报——例如黑名单里的 'FRAME'
    会把正常词 'framework' 全部命中。含空格或非 ASCII 的条目按子串匹配。
    """
    low = text.lower()
    hits = []
    for w in KNOWN_GARBLED:
        lw = w.lower()
        if re.fullmatch(r"[a-z0-9]+", lw):
            if re.search(r"\b%s\b" % re.escape(lw), low):
                hits.append(w)
        elif lw in low:
            hits.append(w)
    return hits


def scan(path, min_len, skip_lowfreq):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")

    print("=" * 66)
    print("文件: %s   (%d 行)" % (path, len(lines)))
    print("=" * 66)

    dirty = False

    bad_uni = check_unicode(text)
    if bad_uni:
        dirty = True
        print("\n[1] 可疑 Unicode:")
        for ch, n in sorted(bad_uni.items(), key=lambda x: -x[1]):
            name = unicodedata.name(ch, "<?>")
            print("    U+%04X  %-28s x%d" % (ord(ch), name, n))
    else:
        print("\n[1] 可疑 Unicode: 无")

    known = check_known(text)
    if known:
        dirty = True
        print("\n[2] 已知伪词黑名单命中:")
        for w in known:
            print("    %r" % w)
    else:
        print("\n[2] 已知伪词: 无")

    if not skip_lowfreq:
        sus = check_lowfreq(text, min_len)
        print("\n[3] 单次出现且长度 >%d 的英文词 (%d 个, 需人工核):"
              % (min_len, len(sus)))
        if sus:
            for i in range(0, len(sus), 4):
                print("    " + "  ".join(sus[i:i + 4]))

    dollars = check_single_dollar(lines)
    if dollars:
        dirty = True
        print("\n[4] 疑似单 $ 行内公式:")
        for lineno, content in dollars:
            print("    L%-4d %s" % (lineno, content))
    else:
        print("\n[4] 单 $ 公式: 无")

    print("")
    return dirty


def main():
    ap = argparse.ArgumentParser(description="扫描长中文文档的乱码词 / 伪 Unicode")
    ap.add_argument("files", nargs="+", help="待检查的文件")
    ap.add_argument("--min-len", type=int, default=6, help="低频英文词的长度阈值 (默认 6)")
    ap.add_argument("--skip-lowfreq", action="store_true", help="跳过第 3 项检查")
    args = ap.parse_args()

    any_dirty = False
    for path in args.files:
        any_dirty |= scan(path, args.min_len, args.skip_lowfreq)

    if any_dirty:
        print(">>> 存在需要处理的项")
        sys.exit(1)
    print(">>> 第 1/2/4 项全部干净；第 3 项请人工过目")
    sys.exit(0)


if __name__ == "__main__":
    main()
