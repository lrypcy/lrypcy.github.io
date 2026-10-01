/*
 * 知识图谱数据装载（供 tools/kg_*.mjs 共用）
 * ------------------------------------------------------------------
 * 复刻 index.html 里那段 Liquid 做的事：拿 _data/knowledge_graph.yml 的系列前缀规则
 * 去 _posts/*.md 的文件名上做匹配，数出每个系列有几篇。
 *
 * 为什么要自己写一个 YAML 子集解析器：这些脚本得能在裸 node 下跑（站点构建依赖 GitHub Pages，
 * 本地没有 Jekyll/bundler），不能为了读 3 个键去装第三方依赖。只认下面这几段：
 *   categories: name / icon / color / desc
 *   series:     id / name / short / category / layer / order / prefixes / exclude / node / desc
 *   links:      - { from, to, label }
 * 新增别的键不受影响（会被读进来但不使用）；改这几段的写法时要同步 kg_layout_check.mjs 的断言。
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const YML_PATH = path.join(ROOT, "_data", "knowledge_graph.yml");
export const POSTS_DIR = path.join(ROOT, "_posts");

export function parseScalar(raw) {
  if (raw === undefined) { return undefined; }
  const v = String(raw).trim();
  if (v === "false") { return false; }
  if (v === "true") { return true; }
  if (/^-?\d+$/.test(v)) { return Number(v); }
  return v.replace(/^["']|["']$/g, "");
}

export function parseList(raw) {
  if (raw === undefined) { return []; }
  const v = String(raw).trim();
  if (!v.startsWith("[") || !v.endsWith("]")) { return [parseScalar(v)]; }
  return v.slice(1, -1).split(",").map((s) => parseScalar(s)).filter((s) => s !== undefined && s !== "");
}

/* 取出某个顶层键（categories / series）下的列表项；遇到下一个顶层键就停 */
function readListBlock(lines, key) {
  const start = lines.findIndex((l) => new RegExp(`^${key}:\\s*$`).test(l));
  if (start < 0) { throw new Error(`_data/knowledge_graph.yml 里找不到 ${key}: 段`); }
  const items = [];
  let cur = null;
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i];
    if (/^[A-Za-z_]/.test(line)) { break; }               // 下一个顶层键
    if (/^\s*(#|$)/.test(line)) { continue; }             // 注释与空行
    const mitem = /^\s*-\s*(\w+):\s*(.*)$/.exec(line);
    if (mitem) { cur = { [mitem[1]]: mitem[2].trim() }; items.push(cur); continue; }
    const mkv = /^\s+([\w-]+):\s*(.*)$/.exec(line);
    if (mkv && cur) { cur[mkv[1]] = mkv[2].trim(); }
  }
  return items;
}

function readLinks(lines) {
  const start = lines.findIndex((l) => /^links:\s*$/.test(l));
  if (start < 0) { return []; }
  const out = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i];
    if (/^[A-Za-z_]/.test(line)) { break; }
    const m = /from:\s*([\w-]+)\s*,\s*to:\s*([\w-]+)\s*,\s*label:\s*(.*?)\s*\}?\s*$/.exec(line);
    if (m) { out.push({ from: m[1], to: m[2], label: parseScalar(m[3]) }); }
  }
  return out;
}

export function loadYml() {
  const text = fs.readFileSync(YML_PATH, "utf8");
  const lines = text.split(/\r?\n/);
  /* 所有标量都要过 parseScalar：yml 里 color: "#2563eb" 是带引号的，
     漏掉这一步就会把 `"#2563eb"`（含引号）当成颜色塞给 SVG fill，
     浏览器判为非法取值、静默回落到黑色——节点全变黑且看不出报错。 */
  return {
    categories: readListBlock(lines, "categories").map((c) => ({
      name: parseScalar(c.name), icon: parseScalar(c.icon),
      color: parseScalar(c.color), desc: parseScalar(c.desc)
    })),
    series: readListBlock(lines, "series").map((s) => ({
      id: parseScalar(s.id), name: parseScalar(s.name), short: parseScalar(s.short),
      category: parseScalar(s.category),
      layer: parseScalar(s.layer), order: parseScalar(s.order),
      node: s.node === undefined ? true : parseScalar(s.node),
      prefixes: parseList(s.prefixes), exclude: parseList(s.exclude), desc: parseScalar(s.desc)
    })),
    links: readLinks(lines)
  };
}

export function listPosts() {
  return fs.readdirSync(POSTS_DIR).filter((f) => f.endsWith(".md")).sort();
}

/* Liquid 的命中规则：文件名 contains 任一前缀；再被任一 exclude 命中就踢掉 */
export function matchPosts(series, posts) {
  return posts.filter((f) => {
    if (!series.prefixes.some((p) => f.includes(p))) { return false; }
    if (series.exclude.some((e) => f.includes(e))) { return false; }
    return true;
  });
}

/* 首页图例上的数字来自 Liquid 的 site.categories[c].size，是「这个分类有几篇文章」，
   与系列篇数不是一回事：node:false 的单篇系列不进图谱，但它照样给所属分类计数。
   这里按 front matter 的 categories 复刻一遍（含 `- A` 列表与 `[A, B]` 行内两种写法）。 */
export function countPostsByCategory(posts) {
  const counter = {};
  posts.forEach((f) => {
    const raw = fs.readFileSync(path.join(POSTS_DIR, f), "utf8");
    const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(raw);
    if (!m) { return; }
    const fm = m[1];
    if (!fm.includes("categories:")) { return; }
    const block = fm.split("categories:")[1].split(/^tags:/m)[0];
    const inline = /\[(.*?)\]/.exec(block);
    const names = inline
      ? inline[1].split(",").map((s) => parseScalar(s)).filter(Boolean)
      : [...block.matchAll(/^\s*-\s*(.+?)\s*$/gm)].map((mm) => parseScalar(mm[1]));
    names.forEach((c) => { counter[c] = (counter[c] || 0) + 1; });
  });
  return counter;
}

/* 与 index.html 输出到 #kg-data 的结构一致（node:false 的系列不进图谱） */
export function buildKgData() {
  const yml = loadYml();
  const posts = listPosts();
  const colorOf = {};
  yml.categories.forEach((c) => { colorOf[c.name] = c.color; });
  const catCount = countPostsByCategory(posts);

  const nodes = [];
  yml.series.forEach((s) => {
    if (s.node === false) { return; }
    nodes.push({
      id: s.id, name: s.name, short: s.short, cat: s.category,
      count: matchPosts(s, posts).length,
      layer: s.layer, order: s.order, desc: s.desc,
      color: colorOf[s.category] || "#3b82f6"
    });
  });

  const cats = yml.categories.map((c) => ({
    name: c.name, color: c.color, icon: c.icon, count: catCount[c.name] || 0
  }));

  return { nodes, links: yml.links, cats, yml, postCount: posts.length, catCount };
}
