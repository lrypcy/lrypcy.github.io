#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 markdown 正文里的半角双引号 " 统一成中文弯引号 “ ”，并修正已有的 “” 配对错误
（典型症状：kramdown 把正文里的 " 全部转成右引号 ”，渲染出 ”xxx” 两边都是右引号）。

保护区域（不改动）：
  - YAML front matter
  - 围栏代码块 ``` / ~~~
  - 缩进代码块（空行之后连续 4 空格/制表符缩进的行块）
  - 行内代码 `...` / ``...``
  - 公式 $$...$$  \[...\]
  - HTML 标签及其属性 <tag attr="...">
  - Liquid {% ... %} / {{ ... }}
  - Markdown 链接/图片目标 (url "title")

用法：
  python3 tools/fix_quotes.py --dry-run        # 只报告，不写盘
  python3 tools/fix_quotes.py                  # 实际写入
  python3 tools/fix_quotes.py path/to/a.md ... # 只处理指定文件
"""
import os
import re
import sys

L = '\u201c'   # “
R = '\u201d'   # ”
SKIP_DIRS = {'_site', '.git', 'node_modules', '.workbuddy', 'assets', 'images', '.venv'}

# ---------- 保护区 ----------
PATTERNS = [
    (re.compile(r'^[ \t]*```.*?^[ \t]*```', re.S | re.M), None),   # 围栏代码块
    (re.compile(r'^[ \t]*~~~.*?^[ \t]*~~~', re.S | re.M), None),   # 围栏代码块
    (re.compile(r'\[\$.*?\$\]', re.S), None),                      # \[ ... \]
    (re.compile(r'\$\$.*?\$\$', re.S), None),                      # $$ ... $$
    (re.compile(r'\{%.*?%\}', re.S), None),                        # Liquid tag
    (re.compile(r'\{\{.*?\}\}', re.S), None),                      # Liquid var
    (re.compile(r'``.+?``', re.S), None),                          # 双反引号行内代码
    (re.compile(r'`[^`\n]+`'), None),                              # 行内代码
    (re.compile(r'<[a-zA-Z/!][^<>]*>', re.S), None),               # HTML 标签与属性
    (re.compile(r'\]\([^()]*\)'), None),                           # 链接目标
]
# 缩进代码块：空行之后开始的、连续 4+ 空格或制表符缩进的行块
INDENT_BLOCK = re.compile(r'(?:(?<=\n)\n|\A)(?:(?: {4,}|\t)[^\n]*\n?)+', re.M)


def mask(text):
    """把保护区域替换成占位符，返回 (masked, slots)"""
    slots = []

    def _sub(m):
        slots.append(m.group(0))
        return '\x00%d\x00' % (len(slots) - 1)

    masked = text
    for pat, _ in PATTERNS:
        masked = pat.sub(_sub, masked)
    masked = INDENT_BLOCK.sub(_sub, masked)
    return masked, slots


PH = re.compile(r'\x00(\d+)\x00')


def unmask(masked, slots):
    """循环解开占位符：保护区可能嵌套（HTML 标签里含 Liquid/链接占位符），单趟替换会漏解。"""
    for _ in range(10):
        if not PH.search(masked):
            break
        masked = PH.sub(lambda m: slots[int(m.group(1))], masked)
    return masked


def split_front_matter(text):
    if text.startswith('---'):
        m = re.match(r'^---\n.*?\n---\n', text, re.S)
        if m:
            return m.group(0), text[m.end():]
    return '', text


def fix_prose(prose):
    """在正文片段里把 " 交替替换成 “ ”，每遇到空行分段重置配对。返回 (new, odd_blocks)"""
    # 先把已有的弯引号拉平，统一重新配对（顺带修好 ”xxx” / ”xxx“ 这类错误）
    prose = prose.replace(L, '"').replace(R, '"')

    blocks = re.split(r'(\n[ \t]*\n+)', prose)   # 保留分隔符
    out = []
    odd = []
    for b in blocks:
        if '"' not in b:
            out.append(b)
            continue
        n = b.count('"')
        buf = []
        idx = 0
        for ch in b:
            if ch == '"':
                buf.append(L if idx % 2 == 0 else R)
                idx += 1
            else:
                buf.append(ch)
        if n % 2 == 1:
            odd.append(b.strip()[:100])
        out.append(''.join(buf))
    return ''.join(out), odd


def flatten(s):
    """把所有弯引号拉平成半角，用于校验“只改引号、不动其它内容”"""
    return s.replace(L, '"').replace(R, '"')


def process_file(path, apply=True):
    raw = open(path, encoding='utf-8').read()
    fm, body = split_front_matter(raw)
    masked, slots = mask(body)
    fixed, odd = fix_prose(masked)
    new_body = unmask(fixed, slots)
    new = fm + new_body
    # 不变式：除引号字符外，内容必须与原文逐字节一致（可捕获占位符泄漏/内容丢失）
    if flatten(new) != flatten(raw):
        raise AssertionError('content changed beyond quotes: %s' % path)
    if '\x00' in new:
        raise AssertionError('placeholder leaked: %s' % path)
    changed = new != raw
    n = raw.count('"') - new.count('"')
    if changed and apply:
        open(path, 'w', encoding='utf-8').write(new)
    return changed, n, odd, len(raw)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    dry = '--dry-run' in sys.argv
    if args:
        files = args
    else:
        files = []
        for dp, dn, fn in os.walk('.'):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in sorted(fn):
                if f.endswith('.md'):
                    files.append(os.path.join(dp, f))
        files.sort()

    total = 0
    nfiles = 0
    all_odd = []
    for p in files:
        try:
            changed, n, odd, _ = process_file(p, apply=not dry)
        except Exception as e:
            print('ERROR', p, e)
            continue
        if changed:
            nfiles += 1
            total += n
            print('%5d  %s' % (n, p))
            for o in odd:
                all_odd.append((p, o))
    print('----')
    print('files changed: %d, quotes converted: %d, dry-run=%s' % (nfiles, total, dry))
    if all_odd:
        print('\n!! 奇数个引号的段落（配对可能错位，需人工看一眼）: %d' % len(all_odd))
        for p, o in all_odd[:40]:
            print('  ', p)
            print('     ', o.replace('\n', ' | '))


if __name__ == '__main__':
    main()
