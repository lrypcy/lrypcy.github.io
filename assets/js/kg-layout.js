/* 首页知识图谱：分层网格布局（纯计算，不碰 DOM）
   ------------------------------------------------------------------
   为什么单独成文件：布局是这份图里唯一「算错就会肉眼可见地坏掉」的部分
   （节点重叠），所以把它抽成纯函数，好让 tools/kg_layout_check.mjs 直接跑断言。

   加载方式：
     浏览器  <script src="/assets/js/kg-layout.js"></script> 必须排在 knowledge-graph.js 之前
     Node    require("assets/js/kg-layout.js")

   硬约束：任意两个节点的圆心距 ≥ r_a + r_b + CLEARANCE。
   只要锚点满足它，轻度松弛（relax）就是恒等变换——松弛只负责在用户拖拽后把
   图拉回一个不重叠的形态，不负责创造不重叠。改任何布局参数后请跑
   `node tools/kg_layout_check.mjs`。
   ------------------------------------------------------------------ */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) { module.exports = api; }
  if (root) { root.KGLayout = api; }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var CLEARANCE = 16; // 两圆之间的最小空隙（px）

  var DEFAULTS = {
    width: 900,
    marginLeft: 124,
    marginRight: 36,
    cols: 4,        // 每层每行最多几个节点
    rowGap: 112,    // 层内行距（圆心到圆心）；必须 > 2*radiusMax + CLEARANCE
    bandHead: 62,   // 层标题基线 → 本层首行圆心；必须 > radiusMax（否则标题压到圆上）
    bandTail: 66,   // 本层末行圆心 → 下一层标题基线；必须 > radiusMax + 节点名高度
    bandTop: 26,    // 第一个层标题基线（留出标题文字高度）
    padBottom: 82,  // 末行圆心 → 画布底边；必须 > radiusMax + 节点名高度
    radiusMin: 19,
    radiusMax: 38,
    radiusPerPost: 3.4,
    bandLabels: { 0: "① 硬件与系统底座", 1: "② 核心算法", 2: "③ 前沿专题" }
  };

  function merge(options) {
    var o = {};
    for (var k in DEFAULTS) { o[k] = DEFAULTS[k]; }
    if (options) { for (var k2 in options) { o[k2] = options[k2]; } }
    return o;
  }

  // 圆圈半径由篇数决定：唯一一篇也有 22.4px，最多的系列封顶在 radiusMax
  function radiusOf(count, o) {
    var c = (typeof count === "number" && count > 0) ? count : 1;
    return Math.min(o.radiusMax, o.radiusMin + Math.sqrt(c) * o.radiusPerPost);
  }

  function bandLabelOf(layer, o) {
    return o.bandLabels[layer] || ("第 " + (layer + 1) + " 层");
  }

  /* 把一层切成若干行：行数 = ceil(n/cols)，再把余数摊到前几行
     （摊余数而不是贪心填满，是为了避免 13 个节点被切成 4+4+4+1 这种空行）
     6 个 → [3,3]；7 个 → [4,3]；13 个 → [4,3,3,3] */
  function splitRows(list, cols) {
    var n = list.length;
    if (n === 0) { return []; }
    var nRows = Math.ceil(n / cols);
    var base = Math.floor(n / nRows), rem = n % nRows;
    var rows = [], i = 0;
    for (var r = 0; r < nRows; r++) {
      var size = base + (r < rem ? 1 : 0);
      rows.push(list.slice(i, i + size));
      i += size;
    }
    return rows;
  }

  /* 轻度松弛：把所有互相侵占的节点推开，再往锚点拉回去。
     锚点合法时它是恒等变换；用户拖动后它负责收敛回一个不重叠的形态。 */
  function relax(nodes, o, W, H) {
    for (var it = 0; it < 60; it++) {
      for (var i = 0; i < nodes.length; i++) {
        for (var j = i + 1; j < nodes.length; j++) {
          var a = nodes[i], b = nodes[j];
          var dx = b.x - a.x, dy = b.y - a.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          // 圆心完全重合时 dx=dy=0，归一化会得到 0 位移（这正是旧版重叠下来的原因）
          if (d < 0.01) { dx = 0.01; d = 0.01; }
          var minD = a.r + b.r + CLEARANCE;
          if (d >= minD) { continue; }
          var ux = dx / d, uy = dy / d;
          if (Math.abs(dy) < o.rowGap * 0.6) { ux = dx > 0 ? 1 : -1; uy = 0; } // 同一行只横向推
          var push = (minD - d) * 0.3;
          a.x -= ux * push; a.y -= uy * push;
          b.x += ux * push; b.y += uy * push;
        }
      }
      nodes.forEach(function (n) {
        n.x += (n.ax - n.x) * 0.14;
        n.y += (n.ay - n.y) * 0.30;
        n.x = Math.max(o.marginLeft - 60, Math.min(W - n.r - 8, n.x));
        n.y = Math.max(n.r + 30, Math.min(H - n.r - 26, n.y));
      });
    }
  }

  /* 入口：就地写入 n.r / n.ax / n.ay / n.x / n.y，返回几何信息
     nodes 需带 { layer, order, count }；层的纵向位置完全由「这层有几行」算出来，
     不再依赖写死的 y 数组——旧版多出来的行会退化成同一根 y，直接叠在一起。 */
  function compute(nodes, options) {
    var o = merge(options);
    var W = o.width;
    var usable = W - o.marginLeft - o.marginRight;
    var slot = usable / o.cols;

    nodes.forEach(function (n) { n.r = radiusOf(n.count, o); });

    // 层号从数据里取（新加一层不用改布局代码），按层号升序自上而下排
    var layers = [];
    nodes.forEach(function (n) { if (layers.indexOf(n.layer) < 0) { layers.push(n.layer); } });
    layers.sort(function (a, b) { return a - b; });

    var bands = [];
    var cursor = o.bandTop;
    layers.forEach(function (layer) {
      var list = nodes.filter(function (n) { return n.layer === layer; })
        .sort(function (a, b) { return (a.order || 0) - (b.order || 0); });
      var rows = splitRows(list, o.cols);
      var firstY = cursor + o.bandHead;
      rows.forEach(function (row, ri) {
        var y = firstY + ri * o.rowGap;
        // 每行按 cols 列的网格居中：行内间距恒为 slot，行越短越居中
        var xStart = o.marginLeft + (usable - row.length * slot) / 2;
        row.forEach(function (n, ci) {
          n.ax = xStart + slot * (ci + 0.5);
          n.ay = y;
        });
      });
      var lastY = firstY + (rows.length - 1) * o.rowGap;
      bands.push({
        layer: layer, label: bandLabelOf(layer, o), labelY: cursor, lineY: cursor + 9,
        firstY: firstY, lastY: lastY, rowCount: rows.length, count: list.length
      });
      cursor = lastY + o.bandTail;
    });

    var H = bands.length ? (cursor - o.bandTail + o.padBottom) : 160;
    nodes.forEach(function (n) { n.x = n.ax; n.y = n.ay; });
    relax(nodes, o, W, H);

    return { W: W, H: H, bands: bands, slot: slot, usable: usable, clearance: CLEARANCE, opts: o };
  }

  /* 最小的圆间净空（负数 = 有重叠），供校验脚本与调试使用 */
  function minGap(nodes) {
    var gap = Infinity;
    for (var i = 0; i < nodes.length; i++) {
      for (var j = i + 1; j < nodes.length; j++) {
        var a = nodes[i], b = nodes[j];
        var d = Math.sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
        gap = Math.min(gap, d - (a.r + b.r));
      }
    }
    return nodes.length < 2 ? Infinity : gap;
  }

  return {
    CLEARANCE: CLEARANCE,
    DEFAULTS: DEFAULTS,
    radiusOf: radiusOf,
    splitRows: splitRows,
    compute: compute,
    minGap: minGap
  };
});
