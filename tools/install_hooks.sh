#!/usr/bin/env bash
# 启用仓库自带的 git hooks（pre-commit 会自动体检首页知识图谱）
#
#   bash tools/install_hooks.sh
#
# 原理：git config core.hooksPath .githooks —— hooks 随仓库版本化，换机器只要再跑一次。
# 卸载：git config --unset core.hooksPath
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

chmod +x .githooks/* 2>/dev/null || true
git config core.hooksPath .githooks

echo "✓ 已启用：core.hooksPath = .githooks"
echo ""
echo "用法："
echo "  正常提交           git commit -m '...'          # 有变动时自动体检，只提示不阻断"
echo "  严格模式（阻断）   GRAPH_SYNC_STRICT=1 git commit -m '...'"
echo "  指定 python        GRAPH_SYNC_PYTHON=/Users/congyuan/Software/miniconda3/bin/python3 git commit -m '...'"
echo "  临时跳过体检       git commit --no-verify"
echo "  卸载               git config --unset core.hooksPath"
