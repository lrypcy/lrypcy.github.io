# AGENTS.md

> 本文件实体在 `.agents/AGENTS.md`，仓库根目录的 `AGENTS.md` 是指向它的软链
> （部分 Agent 工具只认根目录的 `AGENTS.md`）。**改就改 `.agents/` 里这份，别改软链。**
>
> 分工：本文件只讲**工程与环境**；面向内容的写作规范在同目录 **`WRITING.md`**。

## 1. 项目是什么

`lrypcy.github.io` —— Jekyll + GitHub Pages 个人技术博客，主题自维护，无第三方 gem 主题。

- 站名 `一身骄傲`，作者字段 `lrypcy`，仓库 `https://github.com/lrypcy/lrypcy.github.io`
- 构建：**只由 GitHub Pages 远程构建**，本地没有可用的 Jekyll（缺 bundler 2.5.4）
- 生成物：`archives / categories / tags / index.html / search-index.json` 全 Liquid 自动出，**不手工维护**
- `_site/` 是旧构建产物、不被 git 跟踪；`_config.yml` 的 `exclude` 已含 `.agents`、`tools` 等

## 2. 规范文件导航

| 文件 | 看什么 |
| --- | --- |
| `.agents/WRITING.md` | **写作规范**：骨架、语言视角、引用与源码锁定、元叙述不入正文、符号、实验数字、四步校验。动笔前读 |
| `.agents/AGENTS.md`（本文件） | 工程约定：目录、代码示例归属、环境坑、提交门禁、红线 |
| `.workbuddy/memory/topics/*.md` | 子系统与踩坑细节：`analytics`、`knowledge-graph`、`series-boundaries`、`writing-checks` |

> ⚠️ `.workbuddy/` 在 `.gitignore` 里（不随仓库走）。**对外可见的约定一律以 `.agents/` 为准**，
> 记忆目录只放踩坑细节与历史。

## 3. 目录职责

| 路径 | 内容 |
| --- | --- |
| `_posts/` | 正文，130+ 篇，命名 `YYYY-MM-DD-<slug>.md` |
| `_data/knowledge_graph.yml` | 首页知识图谱数据（改它必跑布局校验） |
| `assets/js/` | 图谱布局等前端脚本 |
| `tools/` | **只留**校验脚本 + Cloudflare 访问统计后端；**不再新增随文实验脚本**（存量保留，见 §5） |
| `deep-dive-*/` `wiki/` `Clippings/` | 素材与草稿，不发布 |
| `.agents/` | 仓库内规范（随 git 走） |
| `.workbuddy/memory/` | 项目记忆，append-only 日志 + `topics/` 子话题（不随 git 走） |

## 4. 动笔前 / 改任何东西前

1. 写新长文：先读 `.workbuddy/memory/topics/series-boundaries.md`（系列边界 + 已核正事实）
2. 写公式：先读 user-level skill `kramdown-math-render-fix`
3. 写完：跑 `WRITING.md` §8 的四步校验，**全过才可提交**
4. 改 `_data/knowledge_graph.yml` 或 `assets/js/kg-layout.js`：
   `node tools/kg_layout_check.mjs`（pre-commit 已挂，失败阻断提交）
5. 改访问统计：跑 `node --test tools/analytics/test/*.test.mjs`（**必须带 glob**）+
   `ruby tools/analytics/check_liquid.rb`（改 include 必跑，写错挂整站）

## 5. 代码放哪：ipynbs 仓库（本仓库不再新增代码）

本机所有 git 项目清单在 `~/AGENTS.md`，其中 **`ipynbs` 仓库专门放代码**
（路径见那份清单里 `ipynbs` 一行）。**新写的随文实验脚本与代码示例一律放 `ipynbs`，
不再写进本仓库 `tools/`。**

- **随文实验脚本** → `ipynbs` 的 `experiments/<topic>/<name>/`，结构沿用该仓库既有约定：
  `<name>.ipynb` 或 `<name>.py` + `README.md` + `results/`（输出固化，读者不必自己跑）。
  README 写清**对应文章（带链接）+ 如何运行 + 结论**
- **可独立运行 / 可交互的示例** → 同上，notebook 形态优先（遵守该仓库自己的开发规范）
- 改完 **push 到 GitHub**，然后在本文仓库**引用**：正文给 permalink（锁 commit，
  见 `WRITING.md` §3）。不要把大 notebook 拷进本站仓库
- 本仓库 `tools/` 只剩**校验脚本 + Cloudflare 访问统计后端**；已存在的实验脚本
  （`fm_lab` `megatron_bench` `profiling_bench` `sampling_bench` `rl_*_lab.py` 等）
  **留在原地不动**，等哪天要改动时再顺势迁走

为什么：`tools/` 在 `_config.yml` 的 `exclude` 里，站点上访问不到；留在 GitHub 上读者要
下载上千行脚本才能读懂。ipynbs 的 notebook 代码与输出并排，且能被正文 permalink 直接指过去。

## 6. 环境事实（会静默吃掉结果的坑）

- **macOS 的 `grep` 是 BSD 版，`\|` 交替不生效**，会静默返回空 —— 一律 `grep -E`
- 解释器：只用 numpy → miniconda base `python3`；需 torch →
  `~/Software/miniconda3/envs/torch/bin/python`；纯校验脚本 → managed python 3.13
- 沙箱**直连不通 `*.workers.dev`**（代理拦成 502），curl / WebFetch 都不行；
  `wrangler`（走代理）能通，用 `wrangler d1 execute --remote` 验证；部署 `bash tools/analytics/deploy.sh`
- `git push` / `wrangler deploy` 偶发被沙箱静默拦下（无输出、不报错、版本号不变）；
  必要时 `dangerouslyDisableSandbox` 重跑，再核 `git log origin/master`
- 长实验脚本会被沙箱 SIGTERM（exit 137）掐断且输出截断 —— 给耗时脚本加 `sys.argv` 分组跑

## 7. 提交与发布

- 提交前跑对应门禁（§4）；pre-commit 钩子默认提示不阻断，`GRAPH_SYNC_STRICT=1` 切严格
- 站点改动 push 后由 GitHub Pages 构建，**本地看不到效果**：前端改动先
  `node tools/kg_preview.mjs` 出离线预览再 push
- 交付要求：最终改动 push 到远程 `master`

## 8. 硬性红线

- **任何位置都不能暴露真名**：页脚用 `site.title`，`_config.yml` 的 author 恒为 `lrypcy`
- **仓库里不写主目录绝对路径**（`/Users/<name>/...`）：一律写成 `~/...` 或相对路径。
  凡不在 `_config.yml` 的 `exclude` 里的 `.md` 都会被 Jekyll 原样发布到站点，
  绝对路径等于把账号名挂出去；**新增草稿 / 素材目录也要同步加进 `exclude`**
- **不删 `.workbuddy/`**（项目数据，不是缓存）；个人目录操作一律先只读扫描再确认
- **不自造数学符号**，一律用来源文献的通行写法
- **引用源码必须锁版本 + commit**，行号给 permalink；**但「为什么这么做」的解释不进正文**
- 写作视角类红线（不用第一人称叙述生成过程、不写硬件向内容、没实验就不写、没出处不写实测数）
  见 `WRITING.md` §2 / §6
