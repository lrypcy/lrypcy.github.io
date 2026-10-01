#!/usr/bin/env node
/*
 * 首页知识图谱布局体检
 * ------------------------------------------------------------------
 * 回答一个问题：assets/js/kg-layout.js 算出来的这些节点圆圈，有没有互相压在一起？
 *
 * 用法：
 *   node tools/kg_layout_check.mjs            # 用 _data/knowledge_graph.yml + _posts 的真实数据体检
 *   node tools/kg_layout_check.mjs --quiet    # 只打印结论
 *   node tools/kg_layout_check.mjs --verbose  # 额外打印每个节点坐标
 *
 * 为什么不在浏览器里量：布局是纯函数（kg-layout.js 的 compute），浏览器里那层只是把坐标写成
 * transform。所以直接 require 它跑断言即可——零依赖的回归测试，也不受「沙箱连不上 workers.dev /
 * 本地没有 Jekyll」的影响。想看真实渲染用 tools/kg_preview.mjs。
 *
 * 退出码：0 通过；1 有断言失败（节点重叠 / 越界 / 行距不足）。
 */
import path from "node:path";
import { createRequire } from "node:module";
import { ROOT, buildKgData } from "./kg_data.mjs";

const require = createRequire(import.meta.url);
const LAYOUT = path.join(ROOT, "assets", "js", "kg-layout.js");
const KGLayout = require(LAYOUT);

const argv = process.argv.slice(2);
const QUIET = argv.includes("--quiet");
const VERBOSE = argv.includes("--verbose");

let failures = 0;
function check(label, ok, detail) {
  if (!ok) { failures++; }
  if (!QUIET || !ok) {
    console.log(`  ${ok ? "✓" : "✗"} ${label}${detail ? "  " + detail : ""}`);
  }
}

function computeFor(list) {
  const nodes = list.map((n) => ({ id: n.id, layer: n.layer, order: n.order, count: n.count }));
  return { nodes, geo: KGLayout.compute(nodes) };
}

function assertGeometry(label, list) {
  const { nodes, geo } = computeFor(list);
  if (!nodes.length) { return { nodes, geo }; }
  const rMax = KGLayout.DEFAULTS.radiusMax;

  const gap = KGLayout.minGap(nodes);
  check(`${label}：节点不重叠（最小净空 ${gap === Infinity ? "—" : gap.toFixed(1) + "px"} ≥ ${KGLayout.CLEARANCE}px）`,
    gap >= KGLayout.CLEARANCE - 1e-6);

  const oob = nodes.filter((n) =>
    n.x - n.r < -0.5 || n.x + n.r > geo.W + 0.5 || n.y - n.r < -0.5 || n.y + n.r + 20 > geo.H + 0.5);
  check(`${label}：节点（含圆下方的系列名）全在画布内`,
    oob.length === 0, oob.length ? oob.map((n) => n.id).join(",") : "");

  const headClear = KGLayout.DEFAULTS.bandHead - rMax;
  check(`${label}：层标题与首行节点不打架（留白 ${headClear}px > 0）`, headClear > 0);

  const rowSpare = KGLayout.DEFAULTS.rowGap - (2 * rMax + KGLayout.CLEARANCE);
  check(`${label}：层内行距足够（${KGLayout.DEFAULTS.rowGap} − ${2 * rMax + KGLayout.CLEARANCE} = ${rowSpare}px 余量）`, rowSpare > 0);

  geo.bands.forEach((b) => {
    const ys = [...new Set(nodes.filter((n) => n.layer === b.layer).map((n) => n.y))].sort((a, c) => a - c);
    check(`${label}：第 ${b.layer + 1} 层 ${b.count} 个节点占 ${b.rowCount} 行，每行 y 互不相同`,
      ys.length === b.rowCount, `y = [${ys.map((v) => v.toFixed(0)).join(", ")}]`);
    check(`${label}：第 ${b.layer + 1} 层每行不超过 ${KGLayout.DEFAULTS.cols} 个节点（行尽量均分）`,
      ys.every((y) => nodes.filter((n) => n.y === y).length <= KGLayout.DEFAULTS.cols));
    // 同一行内相邻节点的横向净空
    const inRow = nodes.filter((n) => n.y === ys[0]).sort((a, c) => a.x - c.x);
    for (let i = 1; i < inRow.length; i++) {
      const g = inRow[i].x - inRow[i - 1].x - (inRow[i].r + inRow[i - 1].r);
      check(`${label}：第 ${b.layer + 1} 层行内相邻节点不粘连（${g.toFixed(1)}px ≥ ${KGLayout.CLEARANCE}px）`, g >= KGLayout.CLEARANCE - 1e-6);
    }
  });

  if (VERBOSE) {
    nodes.forEach((n) => console.log(
      `      ${String(n.id).padEnd(14)} L${n.layer} count=${String(n.count).padStart(2)} ` +
      `r=${n.r.toFixed(1)} → (${n.x.toFixed(1)}, ${n.y.toFixed(1)})`));
  }
  return { nodes, geo };
}

/* ---------- 主流程 ---------- */
console.log("知识图谱布局体检：" + path.relative(ROOT, LAYOUT));

const data = buildKgData();
const allSeries = data.yml.series;
const shown = allSeries.filter((s) => s.node !== false);
const hidden = allSeries.filter((s) => s.node === false);
if (!QUIET) {
  console.log(`  图谱节点 ${shown.length} 个（另有 ${hidden.length} 个 node:false 的单篇不进图谱）：` +
    shown.map((s) => s.id).join(", "));
}

console.log("\n[1] 真实数据（_data/knowledge_graph.yml + _posts）");
const realList = data.nodes;
const { nodes: realNodes, geo } = assertGeometry("真实布局", realList);

if (!QUIET) {
  console.log("  各层几何：");
  geo.bands.forEach((b) => {
    console.log(`    第 ${b.layer + 1} 层  ${String(b.count).padStart(2)} 节点 / ${b.rowCount} 行  ` +
      `行心 y = ${b.firstY}${b.rowCount > 1 ? " … " + b.lastY : ""}  标题基线 ${b.labelY}  虚线 ${b.lineY}`);
  });
  console.log(`  viewBox = 0 0 ${geo.W} ${geo.H}（列宽 ${geo.slot.toFixed(1)}px，行距 ${KGLayout.DEFAULTS.rowGap}px，` +
    `最大半径 ${Math.max(...realNodes.map((n) => n.r)).toFixed(1)}px）`);
}

console.log("\n[2] 压力测试（以后每层还会继续加系列，不能再叠到一起）");
for (const n of [1, 2, 4, 5, 6, 7, 8, 9, 12, 13, 16, 20, 40]) {
  const single = [];
  for (let i = 0; i < n; i++) { single.push({ id: `n${i}`, layer: 0, order: i, count: i + 1 }); }
  const { geo: g } = assertGeometry(`单层 ${n} 节点`, single);
  if (!QUIET) { console.log(`     → ${g.bands[0].rowCount} 行，画布高 H = ${g.H}`); }
}

for (const n of [3, 6, 7, 8, 10, 15]) {
  const multi = [];
  for (let l = 0; l < 3; l++) {
    for (let i = 0; i < n; i++) { multi.push({ id: `L${l}n${i}`, layer: l, order: i, count: i + 1 }); }
  }
  assertGeometry(`三层各 ${n} 节点`, multi);
}

console.log("\n[3] 数据完整性（喂给 SVG / 预览页的取值必须是浏览器认得的）");
{
  const badColor = data.nodes.filter((n) => !/^#[0-9a-f]{6}$/i.test(String(n.color)));
  check("每个节点的颜色都是合法的 #rrggbb（带引号或空值会让 fill 静默回落成黑色）",
    badColor.length === 0, badColor.map((n) => `${n.id}=${n.color}`).join(", "));

  const badCatColor = data.cats.filter((c) => !/^#[0-9a-f]{6}$/i.test(String(c.color)));
  check("图例里每个分类的颜色都是合法的 #rrggbb",
    badCatColor.length === 0, badCatColor.map((c) => `${c.name}=${c.color}`).join(", "));

  const dangling = data.links.filter((l) =>
    !data.nodes.some((n) => n.id === l.from) || !data.nodes.some((n) => n.id === l.to));
  check("没有指向不存在节点的依赖边（否则首页会少画一条线）", dangling.length === 0,
    dangling.map((l) => `${l.from}→${l.to}`).join(", "));

  const noDesc = data.nodes.filter((n) => !n.desc);
  check("每个节点都有 desc（悬停卡片要显示）", noDesc.length === 0, noDesc.map((n) => n.id).join(", "));

  // 图例上的数字应与 site.categories[c].size 一致，各分类之和 = 全站文章数
  const sum = data.cats.reduce((a, c) => a + c.count, 0);
  check(`图例各分类之和等于全站文章数（${sum} = ${data.postCount}）`, sum === data.postCount);
  const zero = data.cats.filter((c) => c.count === 0);
  check("没有显示为 0 篇的分类图例", zero.length === 0, zero.map((c) => c.name).join(", "));
}

console.log("\n[4] 篇数与 graph_sync.py 对齐（同一批前缀规则，两边数字必须一样）");
if (!QUIET) {
  data.nodes.forEach((n) => console.log(`    ${String(n.id).padEnd(14)} ${String(n.count).padStart(3)} 篇`));
  console.log("    ↳ 逐行对照 `python3 tools/graph_sync.py` 的「① 系列篇数」");
}

console.log("");
if (failures) {
  console.log(`✗ 布局体检失败：${failures} 项断言未通过。`);
  console.log("  节点重叠多半是把行数/行距改小了，改回 assets/js/kg-layout.js 的 DEFAULTS 再跑一遍。");
  process.exit(1);
}
console.log(`✓ 布局体检通过：${realNodes.length} 个节点互不重叠，行距、层间距、层标题留白都还有余量。`);
