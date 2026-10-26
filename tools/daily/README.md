# AI 日报流水线

每天 08:00 从 **24 个信源**采集**全人工智能 / 超级智能**方向的进展，按
**9 个分类**自动打标、归组，在 `_daily/` 写一篇资讯条目并推送。
**完整手册见 `.agents/DAILY-PIPELINE.md`。**

研究兴趣：具身智能、强化学习与后训练、世界模型 / WAM、AI Infra、自动驾驶端到端、
大模型与前沿实验室、Agent / 超级智能、安全与对齐。

## 四步跑完

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
python3 tools/daily/fetch_all.py --days 7                   # 1) 24 源采集
python3 tools/daily/pick_topics.py --top 20 --by-category   # 2) 筛选 + 分类打标
python3 tools/daily/rate.py --top 20 --by-category          # 3) 评分建议
python3 tools/daily/scaffold.py --top 12                    # 4) 生成按分类归组的骨架
# 精读润色 → 校验 → 提交推送
```

| 脚本 | 干什么 |
| --- | --- |
| `fetch_all.py` | 24 源统一采集，顺带抓候选配图（含中文 alt） |
| `categories.py` | **分类词表读取入口**（数据在 `_data/daily_categories.yml`） |
| `pick_topics.py` | 关键词加权初筛 **+ 自动打主分类**，输出候选清单 |
| `rate.py` | 给候选打 1–5 星建议分（值得读 / 影响力两个维度） |
| `scaffold.py` | 按分类归组的日报**骨架生成器**（半自动写作） |
| `fetch_qingke.py` | 只抓青稞（调试用，`fetch_all` 已含） |

## 三个特性

**分类**：词表在 **`_data/daily_categories.yml`**，脚本与前端（`news.md` /
`_layouts/daily.html`）**共用同一份**，顺序即展示顺序。改分类只改那个文件。
每条候选输出 `category`（主分类 key）+ `category_label` + `category_scores`（九类得分，
便于复核打错没）。日报正文**同一分类写在一起**，分节用
`<h2 id="cat-<key>"><label></h2>` —— 别用 kramdown 的 `{: #id}`（实测会把 `{: #id}`
当字面文字留在标题里）。front matter 写 `cats: [embodied, infra]`（ASCII key）。

**配图**：采集器把候选图（架构图、结果表、真机对比）抓进 `raw-*.json` 的 `images` 字段。
默认**外链**源站图片，特别有价值的才下载到 `assets/img/daily/`。没有合适的图就不加。

**评分**：每期 front matter 有 `rating`（1–5，显示在列表页标题后）；正文每条给
`值得读` 和 `影响力` 两个星。`rate.py` 给建议分，但最终由人定 —— 它只看得懂证据信号
（数字 / 论文 / 代码 / 热度），看不懂「这个结果重不重要」。

口径见 `.agents/DAILY-PIPELINE.md` §2 / §5.2 / §5.3。

## 信源（24 个）

- **中文 / 论文 / 代码**：青稞社区、HuggingFace Daily Papers、arXiv（**9 条检索式**，
  覆盖具身 / 世界模型 / RL / AI Infra / 自动驾驶 / agentic / RLHF）、GitHub 搜索（7 组）、量子位
- **前沿实验室 / 厂商官方**：OpenAI、Google Research、Microsoft Research、DeepMind、
  NVIDIA（2 feed）、BAIR、HuggingFace Blog、AWS ML、PyTorch、vLLM、Together AI
- **媒体 / 深度观点**：MarkTechPost、IEEE Spectrum Robotics、Robohub、TechCrunch AI、
  Ars Technica AI、Interconnects、Sebastian Raschka、LessWrong

英文源补中文源覆盖不到的官方一手发布与欧美实验室动态。
`pick_topics.py` 里给媒体源（qbitai / techcrunch_ai / aws_ml / lesswrong）压了分，
一手论文与官方博客给正分。

**代理 `127.0.0.1:7890`（ClashX）是硬依赖** —— 除青稞和 arXiv 外全都要走代理。
脚本启动会先探测，挂了直接报错。不带代理跑：`DAILY_PROXY= python3 tools/daily/fetch_all.py`。

**抓不到的源**：知乎（匿名 403）、Anthropic（404）、机器之心 RSS（返回 HTML）、
ai.meta.com（400）、Synced / SemiAnalysis（feed 停在 2025）—— 都实测过，别再加。
完整清单见 `.agents/DAILY-PIPELINE.md` §3.1。

## 常用参数

- `--days N` 采集窗口（默认 7 天）
- `--only qingke,hf` 只跑指定信源（**默认合并**上次其他源的数据，不冲掉）
- `--no-merge` 配合 `--only`，强制只留本次抓的
- `--date YYYY-MM-DD` 补跑历史某天
- `pick_topics.py --by-category` 按分类分组打印 + 分类分布
- `pick_topics.py --categories embodied,infra` 只看指定分类
- `pick_topics.py --source hf,arxiv` 只看某些信源
- `pick_topics.py --dump-body` 打印摘要，方便直接阅读
- `--no-body` 跳过青稞正文抓取（快，但拿不到细节和配图）
- `rate.py --all` 给 raw 全部条目打分（不只看候选）
- `scaffold.py --publish --force` 直接把骨架写进 `_daily/`（默认只写 `out/draft-*.md`）

## 注意

- `out/` 在 `.gitignore` 里，是抓取产物，不入库
- **`pick_topics.py` 不能改名成 `select.py`** —— 会遮蔽标准库 `select`，所有网络脚本都会挂
- 筛选器只做初筛，直播预告 / 招聘 / 广告要人工丢掉；`rate.py` 的星级也只是建议
- **分类是建议不是判决**：跨领域条目（具身 + RL、世界模型 + Infra）常打偏，
  写的时候 `cats` 以人工判断为准
- **别把「量化 / 编译 / GPU / 芯片」当噪声降权** —— AI Infra 现在是正经兴趣方向
- 同一工作常被多源重复命中，合并成一条写，链接给全
- 外链图片前先 curl 确认能访问（有的站防盗链）
- 当天没有符合主题的新内容就跳过，不凑数
