#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Megatron 系列验收门禁（契约 SERIES_SPEC.md §5）。

用法：
    python3 tools/megatron_gate.py _posts/2026-10-11-megatron-04-*.md
    python3 tools/megatron_gate.py _posts/*megatron-*.md

口径说明（2026-10-09 监工定稿，**此前口径不一致导致 04 篇被误放行**）：

- §3.2 的「外部引用」**按唯一来源计数，不是按出现次数**。
  出现次数会被「同一个 URL 重复挂 20 次」刷高，而契约 §3.2 明写
  「禁止为凑数堆URL」，所以唯一来源数才是抗堆砌的那一个。
  出现次数与唯一数的比值一并报出（比值过高说明在重复挂同一个来源）。

- 计数时排除：
    - `github.com/lrypcy/...`（配套实验脚本，A 级证据，不算外部引用）
    - `github.com/NVIDIA/Megatron-LM/blob/...`（源码 permalink，属 §3.1 的「源码出处」）
    - `lrypcy.github.io`（站内互链）

退出码 0 = 全部通过；1 = 有门禁未过；2 = 用法错误。
"""
import math
import pathlib
import re
import subprocess
import sys

MD = pathlib.Path(__file__).resolve().parent.parent

# 契约 §3.2：每 300 行至少 8 条外部引用（按唯一来源）
DENSITY_PER = 300
DENSITY_MIN = 8
# 契约 §3.1：正文行号必须给 permalink，密度目标每 150 行至少 1 个
PERMALINK_PER = 150

EXCLUDE = (
    "github.com/lrypcy",
    "github.com/NVIDIA/Megatron-LM",
    "lrypcy.github.io",
)
PERMALINK_RE = re.compile(
    r"github\.com/NVIDIA/Megatron-LM/blob/60e039626/[^\s)>\]]+#L\d+"
)
URL_RE = re.compile(r"https?://[^\s)>\]\"]+")


def external_urls(text):
    return [
        u for u in URL_RE.findall(text)
        if not any(x in u for x in EXCLUDE)
    ]


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except FileNotFoundError:
        return 127, "NOT FOUND"


def check_post(path, lines):
    """§5-1 check_post：公式 / 表格 / <em>"""
    rc, out = run(["python3", str(MD / "tools/check_post.py"), str(path)])
    ok = "[ok]" in out
    return ok, f"{'通过' if ok else '未通过'}（rc={rc}）"


def check_mermaid(path):
    """§5-2 check_mermaid：纯中文标签、无 LaTeX、无危险字符"""
    rc, out = run(["python3", str(MD / "tools/check_mermaid.py"), str(path)])
    ok = "0 个含问题" in out or "全部 mermaid 块" in out
    return ok, "通过" if ok else f"未通过（rc={rc}）"


def scan_garbled(path):
    """§5-3 scan_garbled：第 1/2/4 项必须干净"""
    rc, out = run(["python3", str(MD / "tools/scan_garbled.py"), str(path)])
    ok = "第 1/2/4 项全部干净" in out
    return ok, "第 1/2/4 项干净" if ok else f"未通过（rc={rc}）"


def fix_quotes(path):
    """§5-4 fix_quotes：必须 0 转换。**只跑 --dry-run**——
    2026-10-08 监工漏了这个参数，脚本对全仓库写盘，转了 104 处引号。"""
    rc, out = run(["python3", str(MD / "tools/fix_quotes.py"), "--dry-run", str(path)])
    m = re.search(r"quotes converted:\s*(\d+)", out)
    n = int(m.group(1)) if m else -1
    return n == 0, f"{n} 转换" + ("" if n == 0 else "（应为 0）")


def audit_permalinks(path):
    """§5-5 audit_permalinks：文件在 commit 存在 + 行号在范围内 + 无裸行号"""
    rc, out = run([
        str(pathlib.Path("~/Software/miniconda3/bin/python3").expanduser()),
        str(MD / "tools/audit_permalinks.py"), str(path),
    ])
    broken = re.search(r"broken\s*:\s*(\d+)", out)
    bare = re.search(r"bare （L<n> refs\s*:\s*(\d+)", out)
    nb = int(broken.group(1)) if broken else -1
    nbar = int(bare.group(1)) if bare else 0
    return (nb == 0 and nbar == 0), f"broken={nb} bare={nbar}"


def bare_line_refs(text):
    """§5-6 行号必须带文件名：裸 `（L<数字>` 为零容忍"""
    hits = [i for i, l in enumerate(text.splitlines(), 1) if re.search(r"（L\d+", l)]
    return len(hits) == 0, f"{len(hits)} 处"


def density(text, lines):
    """§5-7 外部引用密度 —— **按唯一来源**，门槛 ceil(行数/300 * 8)。

    先剥代码区：代码块里的 URL 是示例命令的一部分，不是引用。
    """
    body = strip_code(text)
    urls = external_urls(body)
    uniq, occ = len(set(urls)), len(urls)
    need = math.ceil(lines * DENSITY_MIN / DENSITY_PER)
    ok = uniq >= need
    ratio = occ / uniq if uniq else 0
    return ok, f"唯一 {uniq} / 门槛 {need}（出现 {occ}，比值 {ratio:.1f}）"


def permalink_density(text, lines):
    """§3.1 行内 permalink 密度：每 150 行至少 1 个"""
    n = len(PERMALINK_RE.findall(text))
    need = math.ceil(lines / PERMALINK_PER)
    ok = n >= need
    return ok, f"{n} 个 / 需 {need}（1/{lines // max(n, 1)} 行）"


def strip_code(text):
    """剥掉围栏代码块、行内代码与 YAML front matter。

    没有这一步，正则会去扫代码里的内容而误报。监工已在 12 篇上踩到：
      - 行内代码里的美元符号 `` `$86K` `` 被当成「单 $ 公式」
    与 tools/fix_quotes.py 的保护区逻辑保持一致。
    """
    out = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)      # front matter
    out = re.sub(r"^[ \t]*```.*?^[ \t]*```", "", out, flags=re.S | re.M)  # 围栏
    out = re.sub(r"``[^`]*``", "", out)                          # 双反引号
    out = re.sub(r"`[^`\n]*`", "", out)                          # 行内代码
    return out


def redlines(text):
    """§5-9 红线自检（可机器判的部分）—— 先剥代码区再判"""
    body = strip_code(text)
    res = []
    n1 = len(re.findall(r"(我们|笔者|本人)", body))
    res.append(("第一人称叙述", n1 == 0, f"{n1} 处"))
    n2 = len(re.findall(r"(本文接下来|下面我们来|让我们|接下来我们|需要注意的一点是)", body))
    res.append(("元叙述", n2 == 0, f"{n2} 处"))
    n3 = len(re.findall(r"(^|[^$])\$[^$]", body, re.M))
    res.append(("单 $ 公式", n3 == 0, f"{n3} 处"))
    return res


def captions(text):
    """§3.3 图注：紧跟围栏之后、以 `> ` 或 `图 N：` 开头、含出处/来源/URL/行号。
    引用关系网图豁免 per-edge permalink。出处可写在图注的第二行。"""
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if l.strip() == "```mermaid"]
    rel = {i for i, l in enumerate(lines)
           if l.startswith("## ") and "引用关系网" in l}
    ok = 0
    bad = []
    for k, i in enumerate(idx, 1):
        j = i + 1
        while j < len(lines) and lines[j].strip() != "```":
            j += 1
        # 图注可能跨多行，合并接下来 4 行非空内容再判
        cap = " ".join(l.strip() for l in lines[j + 1:j + 5] if l.strip())
        in_rel = any(h < i for h in rel)
        shaped = cap.startswith(">") or re.match(r"^\*?\*?图\s*\d+[：:]", cap)
        sourced = in_rel or any(k2 in cap for k2 in ("出处", "来源")) \
            or "http" in cap or re.search(r"L\d+", cap)
        if shaped and sourced:
            ok += 1
        else:
            bad.append(k)
    return len(bad) == 0, f"{len(idx)} 张 / 合规 {ok}" + (f"，缺 图{bad}" if bad else "")


def gate(path):
    text = path.read_text()
    lines = len(text.splitlines())
    checks = [
        ("§5-1 check_post", *check_post(path, lines)),
        ("§5-2 check_mermaid", *check_mermaid(path)),
        ("§5-3 scan_garbled", *scan_garbled(path)),
        ("§5-4 fix_quotes", *fix_quotes(path)),
        ("§5-5 permalink 审计", *audit_permalinks(path)),
        ("§5-6 裸行号", *bare_line_refs(text)),
        ("§5-7 外部引用密度", *density(text, lines)),
        ("§3.1 permalink 密度", *permalink_density(text, lines)),
        ("§3.3 图注", *captions(text)),
    ]
    for name, ok, _detail in redlines(text):
        checks.append((f"§5-9 {name}", ok, _detail))

    print(f"\n=== {path.name}  {lines} 行 ===")
    failed = []
    for name, ok, detail in checks:
        if not ok:
            failed.append(name)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:24} {detail}")
    if failed:
        print(f"  → 退回：{len(failed)} 项 —— {'、'.join(failed)}")
    else:
        print("  → 可提交")
    return not failed


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    paths = []
    for a in argv:
        p = pathlib.Path(a)
        paths.extend(sorted(p.glob("*.md")) if p.is_dir() else [p])
    if not paths:
        print("没有匹配到文件")
        return 2
    allok = all(gate(p) for p in paths)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))