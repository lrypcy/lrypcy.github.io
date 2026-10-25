# 每日资讯流水线：具身智能 / 世界动作模型 / 强化学习

> 每天 08:00 多信源采集 → 筛选 → 在 `_daily/` 写一篇资讯条目 → 提交推送。
> 本文件是**唯一的操作手册**，跑流水线前先读它。

---

## 1. 目录与文件

| 路径 | 作用 |
| --- | --- |
| `tools/daily/fetch_all.py` | **统一采集器**（五源，主力入口） |
| `tools/daily/pick_topics.py` | 主题筛选与打分，输出候选清单 |
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
| arXiv API | ✅ | 4 条检索式覆盖 WAM / VLA+具身 / RL+机器人 / sim-to-real | 否（走代理更稳） |
| GitHub 仓库搜索 | ✅ | 按 push 时间筛活跃项目，带 star 数与 topics | **是** |
| NVIDIA Developer Blog | ✅ | 两个 Atom feed，去重后约 28 条/周 | **是** |
| 知乎 | ❌ | 匿名访问 403（API 与网页都是）。见 §2.1 | — |

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

# 1. 采集（五源，7 天窗口）
python3 tools/daily/fetch_all.py --days 7

# 2. 筛选
python3 tools/daily/pick_topics.py --top 12 --by-source
python3 tools/daily/pick_topics.py --top 6 --dump-body   # 看正文再定

# 3. 精读入选条目，写 _daily/YYYY-MM-DD-<slug>.md（格式见 §4）
# 4. 校验（§5）
# 5. git add 该条目 && git commit && git push
```

筛选器只是**初筛**。以下情况人工介入：

- 同一工作被多个源重复命中（青稞解读 + arXiv 原文 + HF）→ **合并成一条**，以解读为主、链接给全
- 直播预告 / 招聘 / 广告 → 直接丢弃（`NEGATIVE` 已降权，但会漏）
- 分数低但确实是本方向重磅 → 手动捞回
- **优先照顾多样性**：一期里别全是 arXiv 论文，也别全是青稞解读

---

## 4. 条目格式

```markdown
---
title: 具身智能日报 · <月日>：<一句话点出当日主线>
date: YYYY-MM-DD
summary: 列表页显示的那一句话，40 字内
---

## <条目标题>

**来源**：[青稞社区](链接) ｜ 论文：[arXiv:xxxx](链接) ｜ 代码：[repo](链接)

<3–5 句：解决什么问题、方法是什么、关键数字>

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
- **Halo 列表 API 带 `sort` / `fieldSelector` 参数返回 500**。只能 `?page=1&size=N` 裸取，本地排序。
- **arXiv 官方要求请求间隔**，四条检索式之间 sleep 3s，别并发。
- **代理挂掉的症状**是 HF / GitHub / NVIDIA 三个源集体返回 0 条。脚本已加启动探测。
- 青稞正文抓取要带浏览器 UA；抓取间隔 0.4s。
