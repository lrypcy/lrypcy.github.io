/* AI Infra 知识全景 · 首页交互 */
(function(){
const ROOT = document.querySelector(".atlas");
if (!ROOT || !ROOT.querySelector("#heroStats")) return;

const ATLAS = JSON.parse(document.getElementById("atlas-data").textContent);
const LIVE = JSON.parse(document.getElementById("atlas-series").textContent);

function hydrate(atlas, live){
  const byId = new Map(live.series.map(s => [s.id, s]));
  const total = {2:0, 1:0, 0:0};
  let subs = 0, topics = 0;
  atlas.domains.forEach(d => {
    const ids = new Set();
    let dTopics = 0;
    const cov = {2:0, 1:0, 0:0};
    d.subs.forEach(s => {
      subs++;
      s.series = s.series.map(id => byId.get(id)).filter(Boolean);
      s.series.forEach(x => ids.add(x.id));
      s.topics.forEach(t => {
        topics++; dTopics++;
        total[t.coverage]++; cov[t.coverage]++;
      });
    });
    const postSet = new Set();
    ids.forEach(id => (byId.get(id)?.posts || []).forEach(p => postSet.add(p.url)));
    d.stats = {
      subs: d.subs.length, topics: dTopics,
      coveredSubs: d.subs.filter(s => s.coverage > 0).length,
      fullSubs: d.subs.filter(s => s.coverage === 2).length,
      series: ids.size, posts: postSet.size, cov,
    };
    d.seriesIds = [...ids];
  });
  atlas.series = live.series;
  atlas.meta.stats = {
    domains: atlas.domains.length, subs, topics,
    series: live.series.length, nodeSeries: live.series.filter(s => s.node).length,
    posts: live.posts, cov: total,
    coveragePct: Math.round(((total[2] + total[1] * 0.5) / (topics || 1)) * 100),
  };
}
hydrate(ATLAS, LIVE);
const SITE = "https://lrypcy.github.io";
const COV = {2:"站内专题",1:"部分涉及",0:"站外补齐"};
const COV_ORDER = [2,1,0];
const INK = {
  "#22d3ee":"#0e7490", "#2dd4bf":"#0f766e", "#34d399":"#047857",
  "#60a5fa":"#2563eb", "#818cf8":"#4f46e5", "#a78bfa":"#7c3aed",
  "#c084fc":"#9333ea", "#f472b6":"#db2777", "#fbbf24":"#b45309",
  "#facc15":"#a16207", "#fb923c":"#c2410c", "#94a3b8":"#64748b"
};
const ink = c => INK[String(c).toLowerCase()] || c;

const domainMap = new Map(ATLAS.domains.map(d => [d.id, d]));
const subMap = new Map();
const topicMap = new Map();
const seriesMap = new Map(ATLAS.series.map(s => [s.id, s]));
ATLAS.domains.forEach(d => d.subs.forEach(s => {
  subMap.set(s.id, {...s, domainId:d.id});
  s.topics.forEach(t => topicMap.set(t.id, {...t, subId:s.id, domainId:d.id, subName:s.name, domainName:d.name, subDesc:s.desc}));
}));

const $ = (sel, el=ROOT) => el.querySelector(sel);
const $$ = (sel, el=ROOT) => [...el.querySelectorAll(sel)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (a,b) => b ? Math.round((a/b)*100) : 0;
const fmt = n => n.toLocaleString("zh-CN");

function highlight(text, q){
  const i = text.toLowerCase().indexOf(q);
  if (i < 0) return esc(text);
  return esc(text.slice(0,i)) + "<mark>" + esc(text.slice(i, i+q.length)) + "</mark>" + esc(text.slice(i+q.length));
}
function covPill(c){ return `<span class="cov-pill" data-cov="${c}"><span class="cdot cov${c}"></span>${COV[c]}</span>`; }
function covBar(stats){
  const n = stats.topics || 1;
  return `<div class="covbar"><i style="width:${pct(stats.cov[2],n)}%;background:#10b981"></i><i style="width:${pct(stats.cov[1],n)}%;background:#f59e0b"></i><i style="width:${pct(stats.cov[0],n)}%;background:#cbd5e1"></i></div>`;
}
function covPercent(stats){ return pct(stats.cov[2] + stats.cov[1]*0.5, stats.topics); }

/* ---------------- hero ---------------- */
const S = ATLAS.meta.stats;
$("#heroStats").innerHTML = [
  [S.domains, "知识域"], [S.subs, "子领域"], [S.topics, "主题"],
  [S.series, "站内系列"], [fmt(S.posts), "站内文章"], [S.coveragePct + "%", "加权覆盖度"],
].map(([n,l], i) => `<div class="stat${i===5?" accent":""}"><b>${n}</b><span>${l}</span></div>`).join("");
$("#covPct").textContent = `${S.coveragePct}% 加权覆盖 · ${S.cov[2]} / ${S.topics} 主题为站内专题`;
$("#heroCovBar").innerHTML =
  `<i style="width:${pct(S.cov[2],S.topics)}%;background:#10b981"></i>` +
  `<i style="width:${pct(S.cov[1],S.topics)}%;background:#f59e0b"></i>` +
  `<i style="width:${pct(S.cov[0],S.topics)}%;background:#cbd5e1"></i>`;
$("#heroCovNote").innerHTML = COV_ORDER.map(c =>
  `<span><span class="cdot cov${c}"></span>${COV[c]} <b>${S.cov[c]}</b></span>`).join("");
$("#fc2").textContent = S.cov[2]; $("#fc1").textContent = S.cov[1]; $("#fc0").textContent = S.cov[0];

/* ---------------- graph ---------------- */
const POS = {
  model:[50,10], frontier:[84,10],
  obs:[10,27], dist:[33,27], data:[67,27], gov:[90,27],
  compress:[16,45], serve:[50,45], cluster:[84,45],
  comp:[33,63], eco:[67,63],
  hw:[50,81],
};

const graphEl = $("#graph");
const svg = $("#edges");
const nodeEls = new Map();
const edgeEls = [];
const labelEls = [];

ATLAS.domains.forEach((d,i) => {
  const p = POS[d.id];
  if (!p) { console.warn("no position", d.id); return; }
  const el = document.createElement("div");
  el.className = "gnode";
  el.style.setProperty("--c", d.color);
  el.style.setProperty("--c-ink", ink(d.color));
  el.style.left = p[0] + "%";
  el.style.top = p[1] + "%";
  el.dataset.id = d.id;
  el.tabIndex = 0;
  el.setAttribute("role","button");
  el.setAttribute("aria-label", `${d.name}：${d.tagline}`);
  const cp = covPercent(d.stats);
  el.innerHTML = `
    <div class="gnode-top"><span class="gicon">${d.icon}</span><span class="gname">${esc(d.name)}</span></div>
    <div class="gtag">${esc(d.tagline)}</div>
    <div class="gmeta"><span><i>${d.stats.subs}</i> 子域</span><span><i>${d.stats.topics}</i> 主题</span><span><i>${d.stats.posts}</i> 篇</span></div>
    <div class="gbar" title="站内覆盖 ${cp}%"><span style="width:${pct(d.stats.cov[2],d.stats.topics)}%;background:#10b981"></span><span style="width:${pct(d.stats.cov[1],d.stats.topics)}%;background:#f59e0b"></span><span style="width:${pct(d.stats.cov[0],d.stats.topics)}%;background:#cbd5e1"></span></div>`;
  graphEl.appendChild(el);
  nodeEls.set(d.id, el);
  el.addEventListener("mouseenter", () => { if (!state.selected) focusGraph(d.id); });
  el.addEventListener("mouseleave", () => { if (!state.selected) clearGraphFocus(); else focusGraph(state.selected); });
  el.addEventListener("focus", () => focusGraph(d.id));
  el.addEventListener("blur", () => { if (!state.selected) clearGraphFocus(); });
  el.addEventListener("click", () => { state.selected = d.id; focusGraph(d.id); openFresh({type:"domain",id:d.id}); });
  el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); } });
});

// edges
function curve(a,b){
  const [x1,y1] = a, [x2,y2] = b;
  const dx = x2-x1, dy = y2-y1;
  let c1, c2;
  if (Math.abs(dy) >= Math.abs(dx)) {
    c1 = [x1, y1 + dy*0.42]; c2 = [x2, y2 - dy*0.42];
  } else {
    c1 = [x1 + dx*0.42, y1]; c2 = [x2 - dx*0.42, y2];
  }
  return {d:`M ${x1} ${y1} C ${c1[0]} ${c1[1]}, ${c2[0]} ${c2[1]}, ${x2} ${y2}`,
          mid:[(x1 + 3*c1[0] + 3*c2[0] + x2)/8, (y1 + 3*c1[1] + 3*c2[1] + y2)/8]};
}
const NS = "http://www.w3.org/2000/svg";
ATLAS.edges.forEach(e => {
  const from = domainMap.get(e.from), to = domainMap.get(e.to);
  const g = curve(POS[e.from], POS[e.to]);
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", g.d);
  path.setAttribute("class", "edge");
  path.setAttribute("stroke", ink(from.color));
  path.setAttribute("marker-end", "url(#arrow)");
  if (e.dashed) path.setAttribute("stroke-dasharray", "4 5");
  path.dataset.from = e.from;
  path.dataset.to = e.to;
  const title = document.createElementNS(NS, "title");
  title.textContent = `${from.name} → ${to.name}：${e.label}`;
  path.appendChild(title);
  svg.appendChild(path);

  const hit = document.createElementNS(NS, "path");
  hit.setAttribute("d", g.d);
  hit.setAttribute("class", "edge-hit");
  svg.appendChild(hit);

  const lab = document.createElement("div");
  lab.className = "elabel";
  lab.textContent = e.label;
  lab.style.left = g.mid[0] + "%";
  lab.style.top = g.mid[1] + "%";
  lab.dataset.from = e.from;
  lab.dataset.to = e.to;
  graphEl.appendChild(lab);

  const rec = { ...e, path, lab, fromName:from.name, toName:to.name };
  edgeEls.push(rec);
  labelEls.push(lab);

  const on = () => {
    lab.classList.add("on");
    path.classList.add("hovered");
  };
  const off = () => {
    lab.classList.remove("on");
    path.classList.remove("hovered");
    if (state.selected) focusGraph(state.selected);
  };
  hit.addEventListener("mouseenter", on);
  hit.addEventListener("mouseleave", off);
});

function focusGraph(id){
  graphEl.classList.add("focusing");
  const near = new Set([id]);
  edgeEls.forEach(e => {
    const on = e.from === id || e.to === id;
    e.path.classList.toggle("on", on);
    if (on) { near.add(e.from); near.add(e.to); }
  });
  nodeEls.forEach((el, nid) => {
    el.classList.toggle("near", near.has(nid));
    el.classList.toggle("active", nid === id);
  });
}
function clearGraphFocus(){
  graphEl.classList.remove("focusing");
  nodeEls.forEach(el => el.classList.remove("near","active"));
  edgeEls.forEach(e => e.path.classList.remove("on"));
  labelEls.forEach(l => l.classList.remove("on"));
}

/* ---------------- atlas index + panels ---------------- */
function stickyStackHeight(){
  let h = 0;
  const hd = document.querySelector(".header");
  if (hd && getComputedStyle(hd).position === "sticky") h += hd.getBoundingClientRect().height;
  [ROOT.querySelector(".topbar"), ROOT.querySelector(".atlas-index")].forEach(el => {
    if (el && getComputedStyle(el).position === "sticky") h += el.getBoundingClientRect().height;
  });
  return h + 14;
}
function scrollUnderSticky(el){
  if (!el) return;
  window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - stickyStackHeight(), behavior: "smooth" });
}
$("#atlasIndex").innerHTML = ATLAS.domains.map(d =>
  `<button class="idx" data-jump="${d.id}" style="--c:${d.color}"><i></i>${esc(d.name)}</button>`).join("");
$("#atlasIndex").addEventListener("click", e => {
  const b = e.target.closest("[data-jump]");
  if (!b) return;
  const panel = ROOT.querySelector(`.domain-panel[data-domain="${b.dataset.jump}"]`);
  if (panel) {
    setDomainOpen(panel, true);
    syncDisclosure();
    scrollUnderSticky(panel);
  }
});

function topicChip(t, q){
  return `<button class="topic" data-topic="${t.id}" data-cov="${t.coverage}" data-name="${esc(t.name)}" title="${esc(t.name)} · ${COV[t.coverage]}">${esc(t.name)}</button>`;
}

$("#atlasGrid").innerHTML = ATLAS.domains.map(d => {
  const cp = covPercent(d.stats);
  const subs = d.subs.map(s => {
    const series = s.series.map(x =>
      `<button class="spill" data-series="${x.id}" title="${esc(x.name)}，站内 ${x.count} 篇">${esc(x.short)}<b>${x.count}</b></button>`).join("");
    return `<div class="sub-card cov${s.coverage}" data-sub="${s.id}" id="sub-${s.id}">
      <div class="sub-head">
        <span class="cdot cov${s.coverage}"></span>
        <h4 class="sub-name" data-open-sub="${s.id}">${esc(s.name)}</h4>
        ${covPill(s.coverage)}
        <button class="sub-toggle" data-sub-toggle="${s.id}" aria-expanded="false"><span class="sub-tlabel">展开 ${s.topics.length} 主题</span><i class="chev">▾</i></button>
      </div>
      <p class="sub-desc">${esc(s.desc)}</p>
      ${series ? `<div class="series-row">${series}</div>` : ""}
      <div class="topics">${s.topics.map(t => topicChip(t)).join("")}</div>
    </div>`;
  }).join("");

  return `<article class="domain-panel" id="domain-${d.id}" data-domain="${d.id}" style="--c:${d.color};--c-ink:${ink(d.color)}">
    <div class="dhead">
      <div>
        <div class="dtitle">
          <div class="dicon">${d.icon}</div>
          <div>
            <h3 data-open-domain="${d.id}" style="cursor:pointer">${esc(d.name)}</h3>
            <div class="dtag">${esc(d.tagline)}</div>
          </div>
        </div>
        <p class="ddesc">${esc(d.desc)}</p>
      </div>
      <div class="dside">
        <div class="ring" style="--p:${cp}" title="加权覆盖 ${cp}%"><span>${cp}%</span></div>
        <div class="dstats">
          <span><b>${d.stats.subs}</b> 子域</span>
          <span><b>${d.stats.topics}</b> 主题</span>
          <span><b>${d.stats.series}</b> 系列</span>
          <span><b>${d.stats.posts}</b> 篇</span>
        </div>
        <button class="panel-toggle" data-domain-toggle="${d.id}" aria-expanded="false">展开 ${d.subs.length} 子域 <i class="chev">▾</i></button>
      </div>
    </div>
    <div class="dbar-row">
      ${covBar(d.stats)}
      <div class="dbar-txt">
        <span class="s2">● 专题 ${d.stats.cov[2]}</span>
        <span class="s1">◐ 部分 ${d.stats.cov[1]}</span>
        <span class="s0">○ 站外 ${d.stats.cov[0]}</span>
      </div>
    </div>
    <div class="subs">${subs}</div>
  </article>`;
}).join("");

/* ---------------- atlas interactions ---------------- */
/* ---------------- progressive disclosure ---------------- */
function setDomainOpen(panel, on){
  if (!panel) return;
  panel.classList.toggle("domain-open", on);
  const btn = $("[data-domain-toggle]", panel);
  if (!btn) return;
  const n = $$(".sub-card", panel).length;
  btn.setAttribute("aria-expanded", on ? "true" : "false");
  btn.innerHTML = `${on ? "收起子域" : `展开 ${n} 子域`} <i class="chev">▾</i>`;
}
function setTopicsOpen(card, on){
  if (!card) return;
  card.classList.toggle("topics-open", on);
  const btn = $("[data-sub-toggle]", card);
  if (!btn) return;
  const n = $$(".topic", card).length;
  btn.setAttribute("aria-expanded", on ? "true" : "false");
  btn.innerHTML = `<span class="sub-tlabel">${on ? "收起主题" : `展开 ${n} 主题`}</span><i class="chev">▾</i>`;
  btn.setAttribute("aria-label", `${on ? "收起" : "展开"} ${n} 个主题`);
}
function allTopicsOpen(){
  const panels = $$(".domain-panel");
  return panels.length > 0 && panels.every(p =>
    p.classList.contains("domain-open") &&
    $$(".sub-card", p).every(c => c.classList.contains("topics-open")));
}
function syncDisclosure(){
  $("#toggleAll").textContent = $$(".domain-panel").some(p => p.classList.contains("domain-open"))
    ? "收起领域" : "展开领域";
  $("#toggleAllTopics").textContent = allTopicsOpen() ? "收起全部主题" : "展开全部主题";
}

$("#atlasGrid").addEventListener("click", e => {
  const chip = e.target.closest(".topic");
  if (chip) { openFresh({type:"topic", id:chip.dataset.topic}); return; }

  const spill = e.target.closest(".spill");
  if (spill) { openFresh({type:"series", id:spill.dataset.series}); return; }

  const subOpen = e.target.closest("[data-open-sub]");
  if (subOpen) { openFresh({type:"sub", id:subOpen.dataset.openSub}); return; }

  const domOpen = e.target.closest("[data-open-domain]");
  if (domOpen) { openFresh({type:"domain", id:domOpen.dataset.openDomain}); return; }

  const domToggle = e.target.closest("[data-domain-toggle]");
  if (domToggle) {
    const panel = domToggle.closest(".domain-panel");
    setDomainOpen(panel, !panel.classList.contains("domain-open"));
    syncDisclosure();
    return;
  }

  const subToggle = e.target.closest("[data-sub-toggle]");
  if (subToggle) {
    const card = subToggle.closest(".sub-card");
    setTopicsOpen(card, !card.classList.contains("topics-open"));
    syncDisclosure();
    return;
  }
});

/* ---------------- filter ---------------- */
const state = { q:"", cov:"all", selected:null };
const searchEl = $("#search");
let searchTimer = 0;

function applyFilter(){
  const q = state.q.trim().toLowerCase();
  const cov = state.cov;
  const filtering = !!q || cov !== "all";
  let topicN = 0, subN = 0, domainN = 0;

  $$(".domain-panel").forEach(panel => {
    const d = domainMap.get(panel.dataset.domain);
    const dHit = !!q && (d.name + " " + d.tagline + " " + d.desc).toLowerCase().includes(q);
    let panelTopicN = 0, panelSubN = 0;

    $$(".sub-card", panel).forEach(subEl => {
      const s = subMap.get(subEl.dataset.sub);
      const sHit = !!q && !dHit && (
        (s.name + " " + s.desc).toLowerCase().includes(q) ||
        s.series.some(x => (x.name + " " + x.short).toLowerCase().includes(q))
      );
      let n = 0;
      $$(".topic", subEl).forEach(chip => {
        const t = topicMap.get(chip.dataset.topic);
        const nameHit = !!q && (dHit || sHit || t.name.toLowerCase().includes(q));
        const covOk = cov === "all" || String(t.coverage) === cov;
        const show = covOk && (!q || nameHit);
        chip.classList.toggle("is-hidden", !show);
        chip.classList.toggle("hit", !!q && t.name.toLowerCase().includes(q));
        chip.innerHTML = q && t.name.toLowerCase().includes(q) ? highlight(t.name, q) : esc(t.name);
        if (show) n++;
      });
      const showSub = n > 0;
      subEl.classList.toggle("is-hidden", !showSub);
      subEl.classList.toggle("filter-open", showSub && filtering);
      const nameEl = $(".sub-name", subEl);
      if (nameEl) nameEl.innerHTML = q && s.name.toLowerCase().includes(q) ? highlight(s.name, q) : esc(s.name);
      if (showSub) { panelSubN++; panelTopicN += n; }
    });

    const active = panelTopicN > 0;
    panel.classList.toggle("is-dim", !active && filtering);
    panel.classList.toggle("filter-open", active && filtering);
    if (active) { domainN++; subN += panelSubN; topicN += panelTopicN; }
  });

  $("#toggleAll").disabled = filtering;
  $("#toggleAllTopics").disabled = filtering;
  $("#resultCount").classList.toggle("on", filtering);
  if (filtering) {
    $("#resultCount").textContent = `筛出 ${topicN} 主题 · ${subN} 子域 · ${domainN} 域`;
  }
}

searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.q = searchEl.value;
    if (state.q.trim()) { state.selected = null; clearGraphFocus(); }
    applyFilter();
  }, 70);
});

$("#covFilter").addEventListener("click", e => {
  const b = e.target.closest("button[data-cov]");
  if (!b) return;
  state.cov = b.dataset.cov;
  $$("#covFilter button").forEach(x => x.classList.toggle("on", x === b));
  applyFilter();
});

$("#toggleAll").addEventListener("click", () => {
  const panels = $$(".domain-panel");
  const anyOpen = panels.some(p => p.classList.contains("domain-open"));
  panels.forEach(p => setDomainOpen(p, !anyOpen));
  syncDisclosure();
});

$("#toggleAllTopics").addEventListener("click", () => {
  const panels = $$(".domain-panel");
  const open = !allTopicsOpen();
  panels.forEach(p => {
    setDomainOpen(p, open);
    $$(".sub-card", p).forEach(c => setTopicsOpen(c, open));
  });
  syncDisclosure();
});

$("#reset").addEventListener("click", () => {
  clearFilters();
  window.scrollTo({top:0, behavior:"smooth"});
});

/* ---------------- paths ---------------- */
const domainById = new Map(ATLAS.domains.map(d => [d.id, d]));
$("#paths").innerHTML = ATLAS.paths.map(p => {
  const steps = p.steps.map((step,i) => {
    const s = typeof step === "string" ? domainById.get(step) : step;
    if (!s) return "";
    return `<button class="step" data-open-path="${s.id}" style="--c:${s.color};--c-ink:${ink(s.color)}"><em>${String(i+1).padStart(2,"0")}</em><i>${s.icon}</i>${esc(s.name)}</button>`;
  }).join("");
  return `<div class="path-card"><h3>${esc(p.name)}</h3><p>${esc(p.desc)}</p><div class="steps">${steps}</div><div class="path-meta">${p.steps.length} 个领域 · 按顺序或按需跳读</div></div>`;
}).join("");
$("#paths").addEventListener("click", e => {
  const b = e.target.closest("[data-open-path]");
  if (b) openFresh({type:"domain", id:b.dataset.openPath});
});

/* ---------------- drawer ---------------- */
const drawer = $("#drawer"), backdrop = $("#drawerBackdrop"), drawerBody = $("#drawerBody"), backBtn = $("#drawerBack");
let stack = [];

function postList(posts, limit){
  if (!posts.length) return `<div class="empty">该系列当前没有匹配文章。</div>`;
  const shown = limit ? posts.slice(0, limit) : posts;
  return `<ul class="article-list">${shown.map(p =>
    `<li><a href="${SITE}${esc(p.url)}" target="_blank" rel="noopener"><span>${esc(p.title)}</span><time>${esc(p.date)}</time></a></li>`).join("")}</ul>`;
}

function seriesBlock(id, limit){
  const s = seriesMap.get(id);
  if (!s) return "";
  return `<div class="series-block">
    <button class="series-item" data-open="series|${s.id}"><b>${esc(s.name)}</b><span>${s.count} 篇</span></button>
    ${postList(s.posts, limit || 4)}
    ${s.count > (limit || 4) ? `<button class="more-link" data-open="series|${s.id}">查看全部 ${s.count} 篇 →</button>` : ""}
  </div>`;
}

function renderDomain(id){
  const d = domainMap.get(id);
  const cp = covPercent(d.stats);
  const edges = ATLAS.edges.filter(e => e.from === id || e.to === id).map(e => {
    const currentIsFrom = e.from === id;
    const other = domainMap.get(currentIsFrom ? e.to : e.from);
    return `<button class="edge-item" data-open="domain|${other.id}">
      <span class="cdot" style="background:${other.color}"></span>
      <b>${esc(other.name)}</b>
      <span class="edge-dir">${currentIsFrom ? "支撑下游" : "受上游支撑"}</span>
      <span class="edge-label">${esc(e.label)}</span>
    </button>`;
  }).join("");

  const subs = d.subs.map(s => `<button class="sub-row" data-open="sub|${s.id}">
    <span class="cdot cov${s.coverage}"></span><b>${esc(s.name)}</b>
    <span>${s.topics.length} 主题 · ${s.series.length} 系列</span>
  </button>`).join("");

  const series = d.seriesIds.map(sid => seriesMap.get(sid)).filter(Boolean)
    .map(s => `<button class="series-item" data-open="series|${s.id}"><b>${esc(s.name)}</b><span>${s.count} 篇</span></button>`).join("");

  return `<div style="--c:${d.color};--c-ink:${ink(d.color)}">
    <div class="d-kicker">${d.icon} 知识域</div>
    <h2 class="d-title">${esc(d.name)}</h2>
    <p class="d-sub">${esc(d.tagline)}</p>
    <div class="d-badges"><span class="mini">覆盖 ${cp}%</span><span class="mini">站内专题 ${d.stats.cov[2]} / ${d.stats.topics}</span><span class="mini">${d.stats.series} 个站内系列</span></div>
    <button class="locate-btn" data-locate="domain|${d.id}">在版图中定位 →</button>
    <p class="d-desc">${esc(d.desc)}</p>
    <div class="meta-grid">
      <div class="meta"><b>${d.stats.subs}</b><span>子领域</span></div>
      <div class="meta"><b>${d.stats.topics}</b><span>主题</span></div>
      <div class="meta"><b>${d.stats.posts}</b><span>站内文章</span></div>
      <div class="meta"><b>${d.stats.cov[2]}</b><span>站内专题</span></div>
      <div class="meta"><b>${d.stats.cov[1]}</b><span>部分涉及</span></div>
      <div class="meta"><b>${d.stats.cov[0]}</b><span>站外补齐</span></div>
    </div>
    <div class="d-section">
      <h5>主题覆盖构成 <b>${d.stats.cov[2]} / ${d.stats.topics}</b></h5>
      ${covBar(d.stats)}
    </div>
    <div class="d-section"><h5>子领域 <b>${d.stats.subs}</b></h5><div class="drawer-subs">${subs}</div></div>
    <div class="d-section"><h5>上下游关系 <b>${ATLAS.edges.filter(e=>e.from===id||e.to===id).length}</b></h5><div class="edge-list">${edges || '<div class="empty">无显式依赖边。</div>'}</div></div>
    <div class="d-section"><h5>站内系列 <b>${d.seriesIds.length}</b></h5><div class="series-list">${series || '<div class="empty">该领域暂无站内系列，属于站外补齐范围。</div>'}</div></div>
  </div>`;
}

function renderSub(id){
  const s = subMap.get(id), d = domainMap.get(s.domainId);
  const covCount = {2:0,1:0,0:0};
  s.topics.forEach(t => covCount[t.coverage]++);
  const series = s.series.map(x => seriesBlock(x.id, 5)).join("");
  const topics = s.topics.map(t =>
    `<button class="topic" data-open="topic|${t.id}" data-cov="${t.coverage}" style="position:static">${esc(t.name)}</button>`).join("");
  return `<div style="--c:${d.color};--c-ink:${ink(d.color)}">
    <div class="d-kicker">${d.icon} ${esc(d.name)} / 子领域</div>
    <h2 class="d-title">${esc(s.name)}</h2>
    <p class="d-sub">${s.topics.length} 个主题 · ${s.series.length} 个站内系列</p>
    <div class="d-badges">${covPill(s.coverage)}<span class="mini" style="display:inline-flex;align-items:center;gap:5px"><span class="cdot cov2"></span>${covCount[2]}<span class="cdot cov1"></span>${covCount[1]}<span class="cdot cov0"></span>${covCount[0]}</span></div>
    <button class="locate-btn" data-locate="sub|${s.id}">在版图中展开 →</button>
    <p class="d-desc">${esc(s.desc)}</p>
    <div class="d-section"><h5>主题清单 <b>${s.topics.length}</b></h5><div class="topic-list">${topics}</div></div>
    <div class="d-section"><h5>站内对应系列</h5>${series || '<div class="empty">该子领域暂无站内系列，主题以站外补齐为主。</div>'}</div>
    <div class="d-section"><h5>所属领域</h5><button class="sub-row" data-open="domain|${d.id}"><span class="cdot" style="background:${d.color}"></span><b>${esc(d.name)}</b><span>${esc(d.tagline)}</span></button></div>
  </div>`;
}

function renderTopic(id){
  const t = topicMap.get(id), d = domainMap.get(t.domainId), s = subMap.get(t.subId);
  const covered = s.topics.filter(x => x.coverage > 0).slice(0, 12);
  const note = t.coverage === 2
    ? "该主题在站内有系统专题覆盖，可直接按对应系列顺序阅读。"
    : t.coverage === 1
      ? "该主题在站内有文章触及，但尚未形成完整专题；适合作为延伸阅读，而不是系统学习入口。"
      : "站外补齐：当前博客尚未覆盖该主题。它被保留在版图中，是为了标出知识缺口，而不是假装已经写过。建议按论文、官方文档、开源实现三条线补。";
  const rel = s.series.map(x => `<button class="series-item" data-open="series|${x.id}"><b>${esc(x.name)}</b><span>${x.count} 篇</span></button>`).join("");
  const others = covered.length
    ? `<div class="topic-list">${covered.map(x => `<button class="topic" data-open="topic|${x.id}" data-cov="${x.coverage}" style="position:static">${esc(x.name)}</button>`).join("")}</div>`
    : '<div class="empty">同子域暂无已覆盖主题。</div>';
  return `<div style="--c:${d.color};--c-ink:${ink(d.color)}">
    <div class="d-kicker">${d.icon} ${esc(d.name)} / ${esc(s.name)}</div>
    <h2 class="d-title">${esc(t.name)}</h2>
    <p class="d-sub">主题 · ${esc(s.name)}</p>
    <div class="d-badges">${covPill(t.coverage)}</div>
    <button class="locate-btn" data-locate="topic|${t.id}">在版图中定位 →</button>
    <div class="explain">${esc(note)}</div>
    <div class="d-section"><h5>所属子领域</h5>
      <p class="d-desc" style="margin:0 0 10px">${esc(s.desc)}</p>
      <button class="sub-row" data-open="sub|${s.id}"><span class="cdot cov${s.coverage}"></span><b>${esc(s.name)}</b><span>${s.topics.length} 主题</span></button>
    </div>
    <div class="d-section"><h5>同子域站内系列</h5><div class="series-list">${rel || '<div class="empty">暂无对应系列。</div>'}</div></div>
    <div class="d-section"><h5>同子域已覆盖主题</h5>${others}</div>
  </div>`;
}

function renderSeries(id){
  const s = seriesMap.get(id);
  const linked = ATLAS.domains.filter(d => d.subs.some(x => x.series.some(y => y.id === id)));
  const links = linked.map(d => `<button class="sub-row" data-open="domain|${d.id}"><span class="cdot" style="background:${d.color}"></span><b>${esc(d.name)}</b><span>${esc(d.tagline)}</span></button>`).join("");
  return `<div style="--c:#67e8f9;--c-ink:#0e7490">
    <div class="d-kicker">◫ 站内系列</div>
    <h2 class="d-title">${esc(s.name)}</h2>
    <p class="d-sub">${esc(s.category || "")} · ${s.count} 篇文章</p>
    <div class="d-badges"><span class="mini">${s.node ? "知识图谱节点" : "单篇 / 未入图"}</span><span class="mini">${linked.length} 个领域</span></div>
    <p class="d-desc">${esc(s.desc || "该系列暂无简介。")}</p>
    <div class="d-section"><h5>文章列表 <b>${s.count}</b></h5>${postList(s.posts)}</div>
    <div class="d-section"><h5>所属领域</h5><div class="drawer-subs">${links || '<div class="empty">暂无关联领域。</div>'}</div></div>
  </div>`;
}

function renderEntry(entry){
  if (entry.type === "domain") return renderDomain(entry.id);
  if (entry.type === "sub") return renderSub(entry.id);
  if (entry.type === "topic") return renderTopic(entry.id);
  if (entry.type === "series") return renderSeries(entry.id);
  return "";
}
function crumbOf(entry){
  if (entry.type === "domain") return domainMap.get(entry.id)?.name || "";
  if (entry.type === "sub") { const s = subMap.get(entry.id); return `${domainMap.get(s.domainId)?.name} / ${s.name}`; }
  if (entry.type === "topic") { const t = topicMap.get(entry.id); return `${t.domainName} / ${t.subName} / ${t.name}`; }
  if (entry.type === "series") return seriesMap.get(entry.id)?.name || "";
  return "";
}
function paintDrawer(){
  const entry = stack[stack.length-1];
  if (!entry) return;
  drawerBody.innerHTML = renderEntry(entry);
  drawerBody.scrollTop = 0;
  backBtn.hidden = stack.length < 2;
  $("#drawerCrumb").textContent = crumbOf(entry);
}
function openFresh(entry){
  stack = [entry];
  paintDrawer();
  drawer.classList.add("open");
  backdrop.classList.add("open");
  drawer.setAttribute("aria-hidden","false");
  document.body.classList.add("no-scroll");
}
function navigate(entry){ stack.push(entry); paintDrawer(); }
function closeDrawer(){
  drawer.classList.remove("open");
  backdrop.classList.remove("open");
  drawer.setAttribute("aria-hidden","true");
  document.body.classList.remove("no-scroll");
  stack = [];
  state.selected = null;
  clearGraphFocus();
}
$("#drawerClose").addEventListener("click", closeDrawer);
backdrop.addEventListener("click", closeDrawer);
backBtn.addEventListener("click", () => { if (stack.length > 1) { stack.pop(); paintDrawer(); } });
function scrollToTarget(el){
  if (!el) return;
  el.scrollIntoView({behavior:"smooth", block:"center"});
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1600);
}
function clearFilters(){
  state.q = ""; state.cov = "all"; state.selected = null;
  searchEl.value = "";
  $$("#covFilter button").forEach(x => x.classList.toggle("on", x.dataset.cov === "all"));
  clearGraphFocus();
  applyFilter();
}
function locateInAtlas(type, id){
  if (state.q || state.cov !== "all") clearFilters();
  let panel = null, target = null;
  if (type === "domain") {
    panel = ROOT.querySelector(`.domain-panel[data-domain="${id}"]`);
    target = panel;
    setDomainOpen(panel, true);
  } else if (type === "sub" || type === "topic") {
    const s = type === "sub" ? subMap.get(id) : subMap.get(topicMap.get(id).subId);
    if (!s) return;
    panel = ROOT.querySelector(`.domain-panel[data-domain="${s.domainId}"]`);
    const card = panel && $(`.sub-card[data-sub="${s.id}"]`, panel);
    setDomainOpen(panel, true);
    setTopicsOpen(card, true);
    target = type === "sub" ? card : $(`.topic[data-topic="${id}"]`, card);
  }
  syncDisclosure();
  requestAnimationFrame(() => scrollToTarget(target));
}
drawerBody.addEventListener("click", e => {
  const loc = e.target.closest("[data-locate]");
  if (loc) {
    const [type, id] = loc.dataset.locate.split("|");
    closeDrawer();
    setTimeout(() => locateInAtlas(type, id), 60);
    return;
  }
  const el = e.target.closest("[data-open]");
  if (!el) return;
  const [type, id] = el.dataset.open.split("|");
  navigate({type, id});
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    if (drawer.classList.contains("open")) closeDrawer();
    else if (document.activeElement === searchEl && searchEl.value) {
      searchEl.value = ""; state.q = ""; applyFilter();
    }
  }
  if (e.key === "/" && !drawer.classList.contains("open") && document.activeElement !== searchEl && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
    e.preventDefault();
    searchEl.focus();
  }
});

const topbarEl = ROOT.querySelector('.topbar');
function syncTopbar(){ document.documentElement.style.setProperty('--topbar-h', topbarEl.offsetHeight + 'px'); }
// 小屏下把可横向滚动的依赖图初始定位到主栈居中
if (window.matchMedia('(max-width:1000px)').matches) {
  const gw = ROOT.querySelector('.graph-wrap');
  requestAnimationFrame(() => { gw.scrollLeft = (gw.scrollWidth - gw.clientWidth) / 2; });
}
if (window.ResizeObserver) new ResizeObserver(syncTopbar).observe(topbarEl);
syncTopbar();

applyFilter();
syncDisclosure();

// anchors are resolved after the panels are rendered
function applyHash(){
  if (!location.hash) return;
  const el = ROOT.querySelector(location.hash);
  if (!el) return;
  const card = el.closest(".sub-card");
  if (card) {
    setDomainOpen(card.closest(".domain-panel"), true);
    setTopicsOpen(card, true);
    syncDisclosure();
    requestAnimationFrame(() => scrollUnderSticky(el));
    return;
  }
  if (el.classList.contains("domain-panel")) {
    setDomainOpen(el, true);
    syncDisclosure();
  }
  requestAnimationFrame(() => scrollUnderSticky(el));
}
applyHash();
window.addEventListener("hashchange", applyHash);
})();
