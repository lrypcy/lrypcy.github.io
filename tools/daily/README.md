# 每日资讯流水线

每天 08:00 从多个信源采集具身智能 / 世界动作模型 / 强化学习方向的进展，
在 `_daily/` 写一篇资讯条目并推送。**完整手册见 `.agents/DAILY-PIPELINE.md`。**

## 三步跑完

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
python3 tools/daily/fetch_all.py --days 7            # 11 源采集
python3 tools/daily/pick_topics.py --top 12 --by-source   # 筛选，看候选
python3 tools/daily/rate.py --top 12                # 评分建议（可选）
# 精读后写 _daily/YYYY-MM-DD-<slug>.md，校验，提交推送
```

| 脚本 | 干什么 |
| --- | --- |
| `fetch_all.py` | 11 源统一采集，顺带抓候选配图（含中文 alt） |
| `pick_topics.py` | 关键词加权打分做主题初筛，输出候选清单 |
| `rate.py` | 给候选打 1–5 星建议分（值得读 / 影响力两个维度） |
| `fetch_qingke.py` | 只抓青稞（调试用，`fetch_all` 已含） |

## 两个特性

**配图**：采集器把候选图（架构图、结果表、真机对比）抓进 `raw-*.json` 的 `images` 字段。
默认**外链**源站图片，特别有价值的才下载到 `assets/img/daily/`。没有合适的图就不加。

**评分**：每期 front matter 有 `rating`（1–5，显示在列表页标题后）；正文每条给
`值得读` 和 `影响力` 两个星。`rate.py` 给建议分，但最终由人定 —— 它只看得懂证据信号
（数字 / 论文 / 代码 / 热度），看不懂「这个结果重不重要」。

口径见 `.agents/DAILY-PIPELINE.md` §4.2 / §4.3。

## 信源（11 个）

`fetch_all.py` 统一采集，`fetch_qingke.py` 是青稞单源调试用的（`fetch_all` 已含）。

**中文 / 论文**：青稞社区、HuggingFace Daily Papers、arXiv（4 条检索式）、GitHub 搜索

**英文官方博客**：NVIDIA（2 个 feed）、DeepMind、BAIR、HuggingFace Blog

**英文机器人媒体**：IEEE Spectrum Robotics、Robohub、MarkTechPost

英文源用来补中文源覆盖不到的官方一手发布与欧美实验室动态。实测当天靠 MarkTechPost
捞到 Kyutai CADRE、靠 NVIDIA Blog 捞到 Cosmos 工具链，这两条中文源完全没有。

**代理 `127.0.0.1:7890`（ClashX）是硬依赖** —— 除青稞和 arXiv 外全都要走代理。
脚本启动会先探测，挂了直接报错。不带代理跑：`DAILY_PROXY= python3 tools/daily/fetch_all.py`。

**知乎抓不了**：匿名访问全部 403（网页、API、问答接口都需要登录态）。
当前靠青稞正文里附的作者知乎原文链接间接覆盖。

## 常用参数

- `--days N` 采集窗口（默认 7 天）
- `--only qingke,hf` 只跑指定信源（**默认合并**上次其他源的数据，不冲掉）
- `--no-merge` 配合 `--only`，强制只留本次抓的
- `--date YYYY-MM-DD` 补跑历史某天
- `--source hf,arxiv` 只看某些信源的候选
- `--dump-body` 打印摘要，方便直接阅读
- `--no-body` 跳过青稞正文抓取（快，但拿不到细节和配图）
- `rate.py --all` 给 raw 全部条目打分（不只看候选）

## 注意

- `out/` 在 `.gitignore` 里，是抓取产物，不入库
- **`pick_topics.py` 不能改名成 `select.py`** —— 会遮蔽标准库 `select`，所有网络脚本都会挂
- 筛选器只做初筛，直播预告 / 招聘 / 广告要人工丢掉；`rate.py` 的星级也只是建议
- 同一工作常被多源重复命中，合并成一条写，链接给全
- 外链图片前先 curl 确认能访问（有的站防盗链）
- 当天没有符合主题的新内容就跳过，不凑数
