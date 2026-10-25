# 每日资讯流水线：具身智能 / 世界动作模型 / 强化学习

> 每天 08:00 多信源采集 → 筛选 → 在 `_daily/` 写一篇资讯条目 → 提交推送。
> 本文件是**唯一的操作手册**，跑流水线前先读它。

---

## 1. 目录与文件

| 路径 | 作用 |
| --- | --- |
| `tools/daily/fetch_all.py` | **统一采集器**（11 源，主力入口） |
| `tools/daily/pick_topics.py` | 主题筛选与打分，输出候选清单 |
| `tools/daily/rate.py` | 给候选打 1–5 星建议分（两个维度），写作时参考 |
| `tools/daily/fetch_qingke.py` | 青稞单源采集器（fetch_all 已含，留着调试用） |
| `tools/daily/out/raw-<date>.json` | 当日原始抓取（含摘要全文） |
| `tools/daily/out/brief-<date>.json` | 当日候选清单 |
| `_daily/YYYY-MM-DD-<slug>.md` | 产出的资讯条目，自动进 `/news/` 列表 |

`out/` 在 `.gitignore` 里，不入库也不发布。
**`pick_topics.py` 不能改名回 `select.py`** —— 会遮蔽标准库 `select`，导致所有网络脚本导入失败。

---

## 2. 信源（2026-10-26 实测）

| 信源 | 状态 | 拿到什么 | 依赖代理 |
| --- | --- | --- | --- |
| 青稞 `qingkeai.online` | ✅ | Halo 列表 API + 正文全文。中文解读主力，每篇自带 arXiv / 代码 / 署名 | 否 |
| HuggingFace Daily Papers | ✅ | 论文标题、摘要、AI 摘要、arXiv id、github 仓库、upvotes、关键词 | **是** |
| arXiv API | ✅ | 4 条检索式覆盖 WAM / VLA+具身 / RL+机器人 / sim-to-real。**间歇性抽风**，见 §7 | 否（走代理更稳） |
| GitHub 仓库搜索 | ✅ | 按 push 时间筛活跃项目，带 star 数与 topics | **是** |
| NVIDIA Blog（两个 feed） | ✅ | 官方发布，世界模型 / 物理 AI 一手信息 | **是** |
| DeepMind Blog | ✅ | 官方研究发布（机器人相关较少，Gemini Robotics 系列会出） | **是** |
| BAIR Blog（Berkeley） | ✅ | 官方研究博客。**更新很慢**（实测最新停在 2026-07），多数周期为空，正常 | **是** |
| HuggingFace Blog | ✅ | 官方发布与教程，与 `hf` 论文流不同 | **是** |
| IEEE Spectrum Robotics | ✅ | 机器人垂直媒体，含 Video Friday 等周更栏目 | **是** |
| Robohub | ✅ | 机器人学术社区转载，学术内容占比高 | **是** |
| MarkTechPost | ✅ | 论文速讯，覆盖中文源常漏掉的欧洲实验室（如 Kyutai） | **是** |
| 知乎 | ❌ | 匿名访问 403（API 与网页都是）。见 §2.1 | — |

英文源的价值：中文源（青稞/HF 论文）覆盖不到的**官方一手发布**与**欧美实验室动态**。
实测当天就靠 MarkTechPost 捞到了 Kyutai CADRE、靠 NVIDIA Blog 捞到了 Cosmos 工具链，
这两条中文源完全没有。

**代理是硬依赖**：`127.0.0.1:7890`（ClashX，系统代理设置里已启用）。
HF / GitHub / NVIDIA 直连全不通。脚本启动时会先探代理，挂了直接报错退出。
不带代理跑：`DAILY_PROXY= python3 tools/daily/fetch_all.py`（青稞 + arXiv 仍可用）。

### 2.1 知乎为什么没进采集器

匿名访问全部 403：网页 `/hot`、话题页、`api/v4/search_v3`、问答接口都返回
「您当前请求存在异常」。需要登录态，而本机没装 `agent-browser`。

**当前替代路径**：青稞正文里普遍附有作者的知乎原文链接，等于间接拿到了知乎内容。
需要主动找知乎讨论时用 Bing（`site:zhihu.com`），但中文检索质量不稳定，
实测返回的常是无关结果 —— 只当补充，不当主力。

---

## 3. 每天 08:00 的执行步骤

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io

# 1. 采集（11 源，7 天窗口）
python3 tools/daily/fetch_all.py --days 7

# 2. 筛选
python3 tools/daily/pick_topics.py --top 12 --by-source
python3 tools/daily/pick_topics.py --top 6 --dump-body   # 看正文再定

# 2b. 评分建议（可选，写之前看一眼心里有数）
python3 tools/daily/rate.py --top 12

# 3. 精读入选条目，写 _daily/YYYY-MM-DD-<slug>.md（格式见 §4）
# 4. 校验（§5）
# 5. git add 该条目 && git commit && git push
```

某个源间歇性 reset 时重跑它**不会冲掉其他源**（默认按 source 合并）：

```bash
python3 tools/daily/fetch_all.py --only arxiv        # 只重跑 arxiv，其余保留
python3 tools/daily/fetch_all.py --only arxiv --no-merge  # 强制只留这一源
```

筛选器只是**初筛**。以下情况人工介入：

- 同一工作被多个源重复命中（青稞解读 + arXiv 原文 + HF）→ **合并成一条**，以解读为主、链接给全
- 直播预告 / 招聘 / 广告 → 直接丢弃（`NEGATIVE` 已降权，但会漏）
- 分数低但确实是本方向重磅 → 手动捞回
- **优先照顾多样性**：一期里别全是 arXiv 论文，也别全是青稞解读
- 星级参考 `rate.py` 的建议，但**自己定**（见 §4.3）

---

## 4. 条目格式

```markdown
---
title: 具身智能日报 · <月日>：<一句话点出当日主线>
date: YYYY-MM-DD
summary: 列表页显示的那一句话，40 字内
rating: 4
---

**评分说明**：每条给两个星——`值得读`是读者花几分钟看这条能拿回什么，
`影响力`是这条对领域的冲击。两者不等价。

## <条目标题>

`值得读 ★★★★☆` ｜ `影响力 ★★★★☆`

**来源**：[青稞社区](链接) ｜ 论文：[arXiv:xxxx](链接) ｜ 代码：[repo](链接)

<3–5 句：解决什么问题、方法是什么、关键数字>

![<图的中文说明，用原文 alt>](<图片 URL>)

## <第二条>
...

---

本期 N 条，来源：青稞社区 / HuggingFace Daily Papers / arXiv / GitHub / NVIDIA Blog。
```

硬性要求：

- **数字必须有出处**：照抄原文并给链接；不换算、不估算
- **不确定的标「待验证」**，不把作者观点当结论
- **中文引号用全角**
- 一条 200–400 字，一期 4–6 条
- 不写元叙述（「本文由 XX 整理」不进正文，见 `WRITING.md` §4）
- 纯观点类内容写清「未见公开论文」

### 4.1 front matter 的 `rating`

本期**综合评分**（1–5），显示在 `/news/` 列表页标题后面。取本期各条目星级的大致均值，
再按「今天这期整体含金量」微调。没有 `rating` 字段就不显示星，不报错。

### 4.2 配图

**默认外链**源站图片（零体积、零版权风险，只是引用）。采集器已把候选图抓进
`raw-*.json` 的 `images` 字段（含中文 alt），直接用：

```bash
python3 tools/daily/pick_topics.py --dump-body   # 候选里带 images
```

规则：

- 选**能说明问题**的图（架构图、结果表、真机对比），不选装饰性配图
- alt 用原文的中文说明，或自己写一句「这张图在说什么」，别只写「图 1」
- **没有合适的图就不加**——不为了有图而放图
- 一张条目最多 2 张
- 外链前先用 curl 确认能访问且是图片（防盗链的站会返 403）：

  ```bash
  curl -sI -x http://127.0.0.1:7890 -A "Mozilla/5.0" <图片 URL> | head -3
  ```
- **特别有价值的图才下载到仓库** `assets/img/daily/`（与既有文章一致，永不裂图）。
  单张控制在 200 KB 内，用 `![说明](/assets/img/daily/xxx.png)`

### 4.3 每条的星级

`值得读` 与 `影响力` 各 1–5 星，写在标题下一行。`rate.py` 会给建议分，但**它只看得懂
证据信号**（有没有数字 / 论文 / 代码 / 热度），看不懂「这个结果重不重要」——
**最终星级由人定**。参考口径：

| 星 | 值得读 | 影响力 |
|:--:|:--|:--|
| 5 | 必读，有硬数字 + 新路线 | 方向级突破或重磅开源 |
| 4 | 值得看，有对照实验数字 | 明显推进，别人会接着做 |
| 3 | 一般，有增量 | 增量改进 |
| 2 | 边缘相关，扫一眼即可 | 小修小补 |
| 1 | 凑数 | 几乎无影响 |

立场论文 / 综述默认 `值得读` 低（没实验数字）、`影响力` 未必低——这两个维度就是为
区分这种情况而设的。

---

## 5. 校验（写完必跑）

```bash
python3 tools/check_liquid.py _daily/<file>.md    # Liquid 冲突，必须过
python3 tools/scan_garbled.py _daily/<file>.md    # 乱码伪词
python3 tools/check_links.py                      # 站内链接（不传参数扫全站）
```

条目里若写了公式，还要跑 `python3 tools/check_post.py <file>`。
`check_tldr.py` 不需要跑（那是长文的 TL;DR 体检）。

---

## 6. 定时

**DSH 会话提醒**（`schedule_create`，每天 08:00 Asia/Shanghai）驱动本流水线。
GUI / 会话需在线；关机或会话关闭会漏，漏了手动补跑：

```bash
python3 tools/daily/fetch_all.py --date 2026-10-27 --days 2
```

---

## 7. 踩坑记录

- **`select.py` 会遮蔽标准库 `select`**。文件名改成 `pick_topics.py` 才正常。
  症状是 `AttributeError: module 'select' has no attribute 'select'`。
- **`export.arxiv.org` 会间歇性 Connection reset / 502**，而且一挂就是全部四条检索式。
  不是我们的错，等几秒重试通常能恢复。已改为 4 次重试 + 指数退避（1.5/3/6s）。
  真跑不出来时 HF 源已覆盖大部分论文，不必死等。
- **arXiv RSS（`rss.arxiv.org/rss/<cat>`）是空壳** —— 只有 channel 头没有 item，别用。
- **BAIR feed 更新极慢**（实测最新 2026-07），7 天窗口内返回 0 条是常态，不是故障。
- **Halo 列表 API 带 `sort` / `fieldSelector` 参数返回 500**。只能 `?page=1&size=N` 裸取，本地排序。
- **arXiv 官方要求请求间隔**，四条检索式之间 sleep 3s，别并发。
- **`therobotreport` 403 / `venturebeat` 429 / `ai.meta.com` 无 feed** —— 试过，别再加。
- **代理挂掉的症状**是 HF / GitHub / 博客类源集体返回 0 条。脚本已加启动探测。
- 青稞正文抓取要带浏览器 UA；抓取间隔 0.4s。
