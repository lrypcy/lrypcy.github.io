# 每日资讯流水线：具身智能 / 世界动作模型 / 强化学习

> 每天 08:00 采集 → 筛选 → 在 `_daily/` 写一篇资讯条目 → 提交推送。
> 本文件是**唯一的操作手册**，跑流水线前先读它。

---

## 1. 目录与文件

| 路径 | 作用 |
| --- | --- |
| `tools/daily/fetch_qingke.py` | 青稞社区采集器（列表 API + 正文抓取 + RSS 兜底） |
| `tools/daily/select.py` | 主题筛选与打分，输出候选清单 |
| `tools/daily/out/raw-<date>.json` | 当日原始抓取（含正文全文） |
| `tools/daily/out/brief-<date>.json` | 当日候选清单（不含正文） |
| `_daily/YYYY-MM-DD-<slug>.md` | 产出的资讯条目，自动进 `/news/` 列表 |
| `tools/daily/README.md` | 给人类看的快速上手 |

`out/` 已加进 `_config.yml` 的 `exclude`，不会被 Jekyll 发布。

---

## 2. 信源与实测状态（2026-10-26 核）

| 信源 | 状态 | 说明 |
| --- | --- | --- |
| 青稞 `qingkeai.online` 列表 API | ✅ | `/apis/api.content.halo.run/v1alpha1/posts?page=1&size=N`。**带 `sort` 或 `fieldSelector` 参数会 500**，只能取默认序再本地按 `publishTime` 倒排 |
| 青稞正文 | ✅ | `GET /blog/<slug>`，取 `<article>` 片段去标签。实测 18 条全部达标 |
| 青稞 RSS | ✅ 兜底 | `/feed.xml`，20 条。列表 API 挂时自动回退 |
| arXiv API | ✅ | `export.arxiv.org/api/query`，偶发读超时（已内置 3 次重试）。补充论文一手来源用 |
| 知乎 API | ❌ | 匿名调用 403 风控。需要时改为 Bing 检索命中 `zhuanlan.zhihu.com` |
| HuggingFace papers | ❌ | 沙箱内超时，不依赖 |

**青稞正文里常带作者的知乎原文链接**，写条目时优先引用**知乎原文 + 论文 arXiv 链接**，
青稞链接作为「经青稞社区转载」的出处一起给。

---

## 3. 每天 08:00 的执行步骤

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io

# 1. 采集：抓最近 7 天、最多 40 条
python3 tools/daily/fetch_qingke.py --size 40 --days 7

# 2. 筛选：看候选清单（--dump-body 可打印正文摘要）
python3 tools/daily/select.py --top 8

# 3. 精读入选条目正文，写 _daily/YYYY-MM-DD-<slug>.md（格式见 §4）
# 4. 校验（§5）
# 5. git add 该条目 && git commit && git push
```

筛选器只是**初筛**，不是终判。以下情况人工介入：

- 同一主题多条重复 → 合并成一条写
- 分数高但是直播预告 / 招聘 / 广告 → 直接丢弃（`NEGATIVE` 已降权，但可能漏）
- 分数低但确实是本方向的重磅 → 手动捞回来

---

## 4. 条目格式（照 `_daily/2026-10-24-daily-demo.md`）

```markdown
---
title: 具身智能日报 · <月日>：<一句话点出当日主线>
date: YYYY-MM-DD
summary: 列表页显示的那一句话，40 字内，说清今天有什么
---

## <条目标题>

**来源**：[青稞社区](青稞链接) ｜ [知乎原文](链接) ｜ 作者：<署名>

<3–5 句：解决什么问题、方法是什么、关键数字>

- 关键数字照抄原文，**不换算、不估算**
- 有论文就给 arXiv 链接
- 没有论文（纯公众号/知乎观点文）就写清「未见公开论文」

## <第二条>
...

---

本期 N 条，来源：青稞社区。
```

硬性要求：

- **数字必须有出处**：原文里的数字照抄并给链接；自己不能凭空推算
- **不确定的标「待验证」**，不许把作者观点当结论
- **中文引号用全角**
- 一条 200–400 字，一期 4–6 条，别写成流水账
- 不写元叙述（「本文由 XX 整理」这类不进正文，见 `WRITING.md` §4）

---

## 5. 校验（写完必跑）

```bash
python3 tools/check_liquid.py _daily/<file>.md    # Liquid 冲突，必须过
python3 tools/scan_garbled.py _daily/<file>.md    # 乱码伪词
python3 tools/check_links.py                      # 站内链接（不传参数扫全站）
```

`_daily` 条目不需要跑 `check_tldr.py`（那是长文的 TL;DR 体检）和 `check_post.py`
（那是公式表格渲染检查，资讯条目一般没有公式）。**如果条目里写了公式，就得跑 `check_post.py`。**

pre-commit 会自动跑 `check_liquid`（硬阻断）+ 图谱同步 + 链接检查（提示）。

---

## 6. 定时

**DSH 会话提醒**（`schedule_create`，每天 08:00 Asia/Shanghai）驱动本流水线。
提醒触发后按 §3 执行。GUI/会话需在线；关机或会话关闭会漏，漏了可手动补跑。

补跑历史某天：`python3 tools/daily/fetch_qingke.py --date 2026-10-27 --days 2`。

---

## 7. 踩坑记录

- **Halo 列表 API 带 `sort` 参数返回 500**（`spec.publish` 的 fieldSelector 同样 500）。
  只能 `?page=1&size=N` 裸取，本地排序。默认序基本是时间倒序但不保证。
- **青稞正文抓取要带浏览器 UA**，否则可能拿到不同渲染结果。
- **抓取间隔 0.4s**，别把站点打疼。
- 站点是 Halo 2.x，`/apis/api.content.halo.run/v1alpha1/posts/<slug>` 单条查询返回 404，
  只能走列表接口按 `slug` 过滤。
