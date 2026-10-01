#!/usr/bin/env node
/*
 * 首页知识图谱本地预览（不需要 Jekyll）
 * ------------------------------------------------------------------
 * 背景：本站由 GitHub Pages 构建，本地跑不动 Jekyll（缺 bundler），
 * 所以想看一眼知识图谱长什么样、是不是又叠在一起了，以前只能 push 完等线上。
 * 这个脚本把真实的 kg-data + assets/css/custom.css + 两个图谱脚本拼成一个
 * 自包含 HTML（样式与脚本全部内联，file:// 直接打开就行），排版与首页一致。
 *
 * 用法：
 *   node tools/kg_preview.mjs                 # 写到 /tmp 并打印路径
 *   node tools/kg_preview.mjs --out=preview.html
 *   node tools/kg_preview.mjs --width=360     # 窄屏（手机）预览，配合截图用
 *
 * 截图（macOS 自带 Chrome）：
 *   node tools/kg_preview.mjs --out=/tmp/kg.html
 *   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --no-proxy-server \
 *     --screenshot=/tmp/kg.png --window-size=1080,900 file:///tmp/kg.html
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ROOT, buildKgData } from "./kg_data.mjs";

const argv = process.argv.slice(2);
const argOf = (name, dflt) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};
const OUT = argOf("out", path.join(os.tmpdir(), "kg-preview.html"));
const WIDTH = argOf("width", "1080");

const data = buildKgData();
const css = fs.readFileSync(path.join(ROOT, "assets", "css", "custom.css"), "utf8");
const layoutJs = fs.readFileSync(path.join(ROOT, "assets", "js", "kg-layout.js"), "utf8");
const graphJs = fs.readFileSync(path.join(ROOT, "assets", "js", "knowledge-graph.js"), "utf8");

/* 内联脚本的老坑：被内联的 JS 里只要出现 `</script>`（哪怕在注释里），
   浏览器就会认为这个 script 块到此结束，后面的代码全变成正文文本。
   kg-layout.js 的头部注释里就写了这行示例，所以必须转义。 */
const escapeInline = (s) => s.replace(/<\/script/gi, "<\\/script").replace(/<!--/g, "<\\!--");

/* 图例不在这里写死：knowledge-graph.js 会往 #kg-legend 里追加按钮，
   预渲染一份会变成两排（踩过一次）。 */
const legend = "";

const html = `<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>知识图谱本地预览（离线拼装，非线上产物）</title>
<style>${css}
body { padding: 20px; }
.preview-note { max-width: 900px; margin: 0 auto 12px; font-size: 12.5px; color: var(--text-light); }
</style></head>
<body>
<p class="preview-note">离线预览：kg-data 由 tools/kg_data.mjs 从 _data/knowledge_graph.yml + _posts 现算，
样式与脚本内联自 assets/。与线上唯一的差别是外层没有站点 header / 分页。</p>
<script>
// 预览脚本出错时别静默画出一张空图——直接在页面顶部甩出来
window.addEventListener("error", function (e) {
  var d = document.createElement("pre");
  d.style.cssText = "color:#b91c1c;background:#fff1f2;border:1px solid #fecdd3;padding:10px 12px;" +
    "border-radius:8px;font-size:13px;white-space:pre-wrap;max-width:900px;margin:0 auto 12px";
  d.textContent = "图谱脚本报错：" + (e.message || e) +
    "\n" + (e.filename || "inline") + (e.lineno ? ":" + e.lineno : "");
  document.body.insertBefore(d, document.body.firstChild);
});
</script>
<section class="home-section" style="max-width:900px;margin:0 auto;">
  <div class="kg-card">
    <div class="kg-toolbar">
      <div class="kg-legend" id="kg-legend">${legend}</div>
      <div class="kg-actions">
        <button class="kg-btn" id="kg-expand" type="button">展开全部文章</button>
        <button class="kg-btn" id="kg-reset" type="button">重置布局</button>
      </div>
    </div>
    <div class="kg-scroll">
      <svg id="kg-svg" viewBox="0 0 ${900} 650" role="img" aria-label="站点知识图谱"></svg>
    </div>
    <div class="kg-tip" id="kg-tip" hidden></div>
    <p class="kg-hint">提示：图谱按「硬件与系统底座 → 核心算法 → 前沿专题」三层排布，纵向即学习顺序。</p>
  </div>
</section>
<script type="application/json" id="kg-data">${JSON.stringify({ nodes: data.nodes, links: data.links, cats: data.cats })}</script>
<script>${escapeInline(layoutJs)}</script>
<script>${escapeInline(graphJs)}</script>
</body></html>
`;

fs.writeFileSync(OUT, html, "utf8");
console.log(`✓ 预览已生成：${OUT}`);
console.log(`  节点 ${data.nodes.length} 个 / 依赖边 ${data.links.length} 条 / 分类 ${data.cats.length} 个（宽度按 ${WIDTH}px 看更接近线上）`);
