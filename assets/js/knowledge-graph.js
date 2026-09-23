/* 首页知识图谱：分层 DAG 布局 + 悬停高亮 + 拖拽 + 与下方「分类 × 系列」面板联动
   数据来自 #kg-data（由 _data/knowledge_graph.yml + site.posts 构建）
   ------------------------------------------------------------------ */
(function () {
  "use strict";

  var dataEl = document.getElementById("kg-data");
  var svg = document.getElementById("kg-svg");
  if (!dataEl || !svg || typeof SVGElement === "undefined") { return; }

  var data;
  try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }
  var nodes = data.nodes || [];
  var links = data.links || [];
  var cats = data.cats || [];
  if (!nodes.length) { return; }

  var W = 900, H = 566;
  var MARGIN_L = 124, MARGIN_R = 36;
  var USABLE = W - MARGIN_L - MARGIN_R;
  var SLOT = USABLE / 4;

  var BANDS = [
    { layer: 0, label: "① 硬件与系统底座", ys: [110] },
    { layer: 1, label: "② 核心算法", ys: [248, 372] },
    { layer: 2, label: "③ 前沿专题", ys: [500] }
  ];
  var ROW_CAP = [4, 4, 4]; // 每层每行的最大节点数

  var SVGNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(SVGNS, name);
    if (attrs) { for (var k in attrs) { e.setAttribute(k, attrs[k]); } }
    return e;
  }

  /* ---------- 1. 半径 ---------- */
  nodes.forEach(function (n) {
    n.r = Math.min(38, 19 + Math.sqrt(Math.max(n.count, 1)) * 3.4);
  });

  /* ---------- 2. 分层网格锚点 ---------- */
  BANDS.forEach(function (band) {
    var list = nodes.filter(function (n) { return n.layer === band.layer; })
      .sort(function (a, b) { return a.order - b.order; });
    var cap = ROW_CAP[band.layer] || 4;
    var rows = [];
    for (var i = 0; i < list.length; i += cap) { rows.push(list.slice(i, i + cap)); }
    rows.forEach(function (row, ri) {
      var y = band.ys[ri] !== undefined ? band.ys[ri] : band.ys[band.ys.length - 1];
      var totalW = row.length * SLOT;
      var xStart = MARGIN_L + (USABLE - totalW) / 2;
      row.forEach(function (n, ci) {
        n.ax = xStart + SLOT * (ci + 0.5);
        n.ay = y;
      });
    });
  });

  nodes.forEach(function (n) { n.x = n.ax; n.y = n.ay; });

  /* ---------- 3. 轻度松弛（保证不重叠又不失层感） ---------- */
  (function relax() {
    for (var it = 0; it < 40; it++) {
      for (var i = 0; i < nodes.length; i++) {
        for (var j = i + 1; j < nodes.length; j++) {
          var a = nodes[i], b = nodes[j];
          var dx = b.x - a.x, dy = b.y - a.y;
          var d = Math.sqrt(dx * dx + dy * dy) || 0.01;
          var minD = a.r + b.r + 16;
          if (d < minD && Math.abs(dy) < 70) { // 同一行：只横向推开
            var push = (minD - d) * 0.25;
            var ux = dx / d;
            a.x -= ux * push; b.x += ux * push;
          } else if (d < minD) {
            var push2 = (minD - d) * 0.25;
            a.x -= (dx / d) * push2; a.y -= (dy / d) * push2;
            b.x += (dx / d) * push2; b.y += (dy / d) * push2;
          }
        }
      }
      nodes.forEach(function (n) {
        n.x += (n.ax - n.x) * 0.14;
        n.y += (n.ay - n.y) * 0.30;
        n.x = Math.max(MARGIN_L - 60, Math.min(W - n.r - 8, n.x));
        n.y = Math.max(n.r + 30, Math.min(H - n.r - 26, n.y));
      });
    }
  })();

  /* ---------- 4. 绘制 ---------- */
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);

  var gBands = el("g", { class: "kg-bands" });
  var gEdges = el("g", { class: "kg-edges" });
  var gELab = el("g", { class: "kg-elabels" });
  var gNodes = el("g", { class: "kg-nodes" });
  svg.appendChild(gBands); svg.appendChild(gEdges); svg.appendChild(gELab); svg.appendChild(gNodes);

  // 层标签
  BANDS.forEach(function (band) {
    var t = el("text", { x: 12, y: band.ys[0] - 34, class: "kg-band-label" });
    t.textContent = band.label;
    gBands.appendChild(t);
    var line = el("line", {
      x1: 12, y1: band.ys[0] - 26, x2: W - 20, y2: band.ys[0] - 26,
      class: "kg-band-line"
    });
    gBands.appendChild(line);
  });

  function nodeById(id) {
    for (var i = 0; i < nodes.length; i++) { if (nodes[i].id === id) { return nodes[i]; } }
    return null;
  }

  // 边
  var edgeEls = [];
  links.forEach(function (l, idx) {
    var a = nodeById(l.from), b = nodeById(l.to);
    if (!a || !b) { return; }
    var g = el("g", { class: "kg-edge" });
    var path = el("path", { class: "kg-edge-line", fill: "none" });
    var head = el("polygon", { class: "kg-edge-head" });
    g.appendChild(path); g.appendChild(head);
    gEdges.appendChild(g);

    var lab = el("g", { class: "kg-elabel" });
    var txt = el("text", { class: "kg-elabel-text" });
    txt.textContent = l.label || "";
    var bg = el("rect", { class: "kg-elabel-bg", rx: 5 });
    lab.appendChild(bg); lab.appendChild(txt);
    gELab.appendChild(lab);

    edgeEls.push({ a: a, b: b, g: g, path: path, head: head, lab: lab, bg: bg, txt: txt, idx: idx });
  });

  // 节点
  nodes.forEach(function (n, i) {
    var g = el("g", { class: "kg-node", tabindex: "0", role: "button" });
    g.setAttribute("aria-label", n.name + "，" + n.count + " 篇，分类：" + n.cat);
    var fx = el("g", { class: "kg-node-fx" });
    fx.style.animationDelay = (i * 55) + "ms";
    g.appendChild(fx);

    var c = el("circle", { r: n.r, fill: n.color, class: "kg-node-circle" });
    var halo = el("circle", { r: n.r + 6, fill: "none", class: "kg-node-halo" });
    var num = el("text", { class: "kg-node-num", "text-anchor": "middle", "dominant-baseline": "central" });
    num.textContent = String(n.count);
    var name = el("text", { class: "kg-node-name", "text-anchor": "middle" });
    name.textContent = n.short || n.name;

    fx.appendChild(halo); fx.appendChild(c); fx.appendChild(num); fx.appendChild(name);
    gNodes.appendChild(g);
    n._g = g; n._c = c; n._halo = halo; n._num = num; n._name = name;
  });

  function placeNode(n) {
    n._g.setAttribute("transform", "translate(" + n.x.toFixed(1) + "," + n.y.toFixed(1) + ")");
    n._name.setAttribute("y", (n.r + 17).toFixed(1));
  }
  nodes.forEach(placeNode);

  function drawEdges() {
    edgeEls.forEach(function (e) {
      var a = e.a, b = e.b;
      var dx = b.x - a.x, dy = b.y - a.y;
      var d = Math.sqrt(dx * dx + dy * dy) || 1;
      var ux = dx / d, uy = dy / d;
      var sx = a.x + ux * (a.r + 3), sy = a.y + uy * (a.r + 3);
      var ex = b.x - ux * (b.r + 9), ey = b.y - uy * (b.r + 9);
      // 轻微弧线，避免直线重叠
      var mx = (sx + ex) / 2, my = (sy + ey) / 2;
      var px = -uy, py = ux;
      var bow = (e.idx % 2 === 0 ? 1 : -1) * Math.min(16, d * 0.06);
      var cx = mx + px * bow, cy = my + py * bow;
      e.path.setAttribute("d", "M" + sx.toFixed(1) + "," + sy.toFixed(1) +
        " Q" + cx.toFixed(1) + "," + cy.toFixed(1) + " " + ex.toFixed(1) + "," + ey.toFixed(1));

      var t = 1;
      var tipX = ex, tipY = ey;
      var dirX = tipX - cx, dirY = tipY - cy;
      var dl = Math.sqrt(dirX * dirX + dirY * dirY) || 1;
      dirX /= dl; dirY /= dl;
      var w = 4.6, len = 8.5;
      var nx = -dirY, ny = dirX;
      e.head.setAttribute("points", [
        (tipX + dirX * len).toFixed(1) + "," + (tipY + dirY * len).toFixed(1),
        (tipX + nx * w).toFixed(1) + "," + (tipY + ny * w).toFixed(1),
        (tipX - nx * w).toFixed(1) + "," + (tipY - ny * w).toFixed(1)
      ].join(" "));

      // 边标签落在弧线中点
      var lx = mx + px * bow, ly = my + py * bow;
      e.txt.setAttribute("x", lx.toFixed(1));
      e.txt.setAttribute("y", (ly - 5).toFixed(1));
      var wpx = (e.txt.textContent.length * 11 + 14);
      e.bg.setAttribute("x", (lx - wpx / 2).toFixed(1));
      e.bg.setAttribute("y", (ly - 18).toFixed(1));
      e.bg.setAttribute("width", wpx.toFixed(1));
      e.bg.setAttribute("height", 19);
      e.txt.setAttribute("text-anchor", "middle");
    });
  }
  drawEdges();

  /* ---------- 5. 图例 ---------- */
  var legend = document.getElementById("kg-legend");
  if (legend) {
    cats.forEach(function (c) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "kg-legend-item";
      b.setAttribute("data-cat", c.name);
      var dot = document.createElement("span");
      dot.className = "cat-dot";
      dot.style.background = c.color;
      b.appendChild(dot);
      b.appendChild(document.createTextNode(c.name + " "));
      var em = document.createElement("em");
      em.textContent = c.count;
      b.appendChild(em);
      b.addEventListener("click", function () { showCategory(c.name, true); });
      legend.appendChild(b);
    });
  }

  /* ---------- 6. 面板联动 ---------- */
  var panels = [].slice.call(document.querySelectorAll("#cat-panels .cat-panel"));
  var tabs = [].slice.call(document.querySelectorAll("#cat-tabs .cat-tab"));

  function showCategory(cat, scroll) {
    panels.forEach(function (p) {
      p.hidden = !(cat === "__all__" || p.getAttribute("data-cat") === cat);
    });
    tabs.forEach(function (t) {
      t.classList.toggle("is-active", t.getAttribute("data-cat") === cat);
    });
    if (scroll) {
      var map = document.getElementById("cat-panels");
      if (map && map.scrollIntoView) { map.scrollIntoView({ behavior: "smooth", block: "start" }); }
    }
  }

  function focusSeries(id) {
    var target = document.getElementById("series-" + id);
    if (!target) { return false; }
    var panel = target.closest ? target.closest(".cat-panel") : null;
    if (panel && panel.hidden) { showCategory(panel.getAttribute("data-cat"), false); }
    target.classList.remove("is-flash");
    void target.offsetWidth;
    target.classList.add("is-flash");
    if (target.scrollIntoView) { target.scrollIntoView({ behavior: "smooth", block: "center" }); }
    return true;
  }

  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      showCategory(t.getAttribute("data-cat"), false);
    });
  });

  // 学习路线跳转
  [].slice.call(document.querySelectorAll(".road-step")).forEach(function (a) {
    a.addEventListener("click", function (e) {
      var id = a.getAttribute("data-target");
      if (focusSeries(id)) { e.preventDefault(); }
    });
  });

  // 系列内文章展开/收起
  [].slice.call(document.querySelectorAll(".series-toggle")).forEach(function (btn) {
    btn.setAttribute("data-label", btn.textContent);
    btn.addEventListener("click", function () {
      var blk = btn.closest(".series-block");
      var open = blk.classList.toggle("is-open");
      btn.textContent = open ? "收起 ↑" : btn.getAttribute("data-label");
    });
  });

  var expandBtn = document.getElementById("kg-expand");
  if (expandBtn) {
    var expanded = false;
    expandBtn.addEventListener("click", function () {
      expanded = !expanded;
      panels.forEach(function (p) {
        [].slice.call(p.querySelectorAll(".series-block")).forEach(function (b) {
          b.classList.toggle("is-open", expanded);
          var tb = b.querySelector(".series-toggle");
          if (tb) { tb.textContent = expanded ? "收起 ↑" : tb.getAttribute("data-label"); }
        });
      });
      expandBtn.textContent = expanded ? "收起全部文章" : "展开全部文章";
      expandBtn.classList.toggle("is-active", expanded);
    });
  }

  /* ---------- 7. 悬停 / 点击 / 拖拽 ---------- */
  var tip = document.getElementById("kg-tip");
  var card = svg.parentNode ? svg.parentNode.parentNode : null;

  function showTip(n, evt) {
    if (!tip || !card) { return; }
    tip.innerHTML = "";
    var h = document.createElement("div");
    h.className = "kg-tip-head";
    var dot = document.createElement("span");
    dot.className = "cat-dot";
    dot.style.background = n.color;
    h.appendChild(dot);
    var strong = document.createElement("b");
    strong.textContent = n.name;
    h.appendChild(strong);
    var em = document.createElement("em");
    em.textContent = n.count + " 篇";
    h.appendChild(em);
    var meta = document.createElement("div");
    meta.className = "kg-tip-meta";
    meta.textContent = n.cat + " · 第 " + (n.layer + 1) + " 层";
    var desc = document.createElement("p");
    desc.className = "kg-tip-desc";
    desc.textContent = n.desc || "";
    tip.appendChild(h); tip.appendChild(meta); tip.appendChild(desc);
    tip.hidden = false;
    moveTip(evt);
  }
  function moveTip(evt) {
    if (!tip || !card || tip.hidden) { return; }
    var r = card.getBoundingClientRect();
    var x = evt.clientX - r.left + 14;
    var y = evt.clientY - r.top + 14;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > r.width - 8) { x = r.width - tw - 8; }
    if (y + th > r.height - 8) { y = y - th - 28; }
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { if (tip) { tip.hidden = true; } }

  function highlight(n) {
    var near = {};
    near[n.id] = true;
    edgeEls.forEach(function (e) {
      var on = (e.a.id === n.id || e.b.id === n.id);
      e.g.classList.toggle("is-hl", on);
      e.g.classList.toggle("is-dim", !on);
      e.lab.classList.toggle("show", on);
      if (on) { near[e.a.id] = true; near[e.b.id] = true; }
    });
    nodes.forEach(function (m) {
      var on = !!near[m.id];
      m._g.classList.toggle("is-hl", m.id === n.id);
      m._g.classList.toggle("is-near", on && m.id !== n.id);
      m._g.classList.toggle("is-dim", !on);
    });
  }
  function clearHighlight() {
    edgeEls.forEach(function (e) {
      e.g.classList.remove("is-hl", "is-dim");
      e.lab.classList.remove("show");
    });
    nodes.forEach(function (m) { m._g.classList.remove("is-hl", "is-near", "is-dim"); });
  }

  nodes.forEach(function (n) {
    n._g.addEventListener("mouseenter", function (e) { highlight(n); showTip(n, e); });
    n._g.addEventListener("mousemove", moveTip);
    n._g.addEventListener("mouseleave", function () { clearHighlight(); hideTip(); });
    n._g.addEventListener("focus", function () { highlight(n); });
    n._g.addEventListener("blur", function () { clearHighlight(); });
    n._g.addEventListener("click", function (e) {
      if (n._dragMoved) { return; }
      e.preventDefault();
      highlight(n);
      focusSeries(n.id);
    });
    n._g.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); focusSeries(n.id); }
    });
    n._g.style.cursor = "grab";

    // 拖拽
    var dragging = false, startX = 0, startY = 0, ox = 0, oy = 0;
    n._dragMoved = false;
    n._g.addEventListener("pointerdown", function (evt) {
      dragging = true; n._dragMoved = false;
      startX = evt.clientX; startY = evt.clientY; ox = n.x; oy = n.y;
      n._g.style.cursor = "grabbing";
      try { n._g.setPointerCapture(evt.pointerId); } catch (err) {}
    });
    n._g.addEventListener("pointermove", function (evt) {
      if (!dragging) { return; }
      var rect = svg.getBoundingClientRect();
      var scale = rect.width / W;
      var dx = (evt.clientX - startX) / scale;
      var dy = (evt.clientY - startY) / scale;
      if (Math.abs(dx) + Math.abs(dy) > 3) { n._dragMoved = true; }
      n.x = Math.max(n.r + 6, Math.min(W - n.r - 6, ox + dx));
      n.y = Math.max(n.r + 20, Math.min(H - n.r - 18, oy + dy));
      placeNode(n); drawEdges();
    });
    function endDrag() {
      if (!dragging) { return; }
      dragging = false;
      n._g.style.cursor = "grab";
      setTimeout(function () { n._dragMoved = false; }, 0);
    }
    n._g.addEventListener("pointerup", endDrag);
    n._g.addEventListener("pointercancel", endDrag);
  });

  svg.addEventListener("mouseleave", function () { clearHighlight(); hideTip(); });

  var resetBtn = document.getElementById("kg-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      nodes.forEach(function (n) { n.x = n.ax; n.y = n.ay; placeNode(n); });
      drawEdges();
      clearHighlight();
    });
  }

  /* ---------- 8. URL hash 直接定位到系列 ---------- */
  if (location.hash && location.hash.indexOf("#series-") === 0) {
    var sid = location.hash.slice("#series-".length);
    setTimeout(function () { focusSeries(decodeURIComponent(sid)); }, 120);
  }
})();
