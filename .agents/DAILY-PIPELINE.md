# 每日资讯流水线：AI 日报（全 AI / 超级智能）

> 每天 08:00 多信源采集 → 分类筛选 → 评分 → 在 `_daily/` 写一篇资讯条目 → 提交推送。
> 本文件是**唯一的操作手册**，跑流水线前先读它。

**主题范围**：全部人工智能与超级智能方向，不再局限于具身智能。研究兴趣包括（不限于）：
具身智能、强化学习与后训练、世界模型 / 世界动作模型、AI Infra（训练与推理系统）、
自动驾驶端到端算法、大模型与前沿实验室发布、Agent / 超级智能、安全与对齐。
判定标准只有一条：**是不是 AI 方向的研究进展或一手发布**，是就收，不是就丢。

---

## 1. 目录与文件

| 路径 | 作用 |
| --- | --- |
| `_data/daily_categories.yml` | **分类词表唯一事实源**（9 类，key + 中文 label + 顺序） |
| `tools/daily/fetch_all.py` | **统一采集器**（24 源，主力入口） |
| `tools/daily/pick_topics.py` | 全主题筛选 + **自动分类打标**，输出候选清单 |
| `tools/daily/rate.py` | 给候选打 1–5 星建议分（两个维度），写作时参考 |
| `tools/daily/scaffold.py` | 按分类归组的日报**骨架生成器**（半自动写作） |
| `tools/daily/fetch_qingke.py` | 青稞单源采集器（fetch_all 已含，留着调试用） |
| `tools/daily/out/raw-<date>.json` | 当日原始抓取（含摘要全文） |
| `tools/daily/out/brief-<date>.json` | 当日候选清单（带分类） |
| `tools/daily/out/rated-<date>.json` | 当日评分结果（带分类） |
| `_daily/YYYY-MM-DD-ai-daily.md` | 产出的资讯条目，自动进 `/news/` 列表 |

`out/` 在 `.gitignore` 里，不入库也不发布。
**`pick_topics.py` 不能改名回 `select.py`** —— 会遮蔽标准库 `select`，导致所有网络脚本导入失败。

---

## 2. 分类词表（唯一事实源：`_data/daily_categories.yml`）

**改分类要动这个文件，别在脚本或模板里另写一份。** 脚本读它做打标，
`news.md` / `_layouts/daily.html` 用 `site.data.daily_categories` 读它做展示。

| key | label | 覆盖什么 |
| --- | --- | --- |
| `embodied` | 具身智能 | VLA、操作与抓取、人形与灵巧手、locomotion、sim-to-real、模仿学习 |
| `world-model` | 世界模型 | world model / WAM、视频生成即模拟、JEPA、神经仿真器、dynamics model |
| `rl` | 强化学习与后训练 | RL 算法、GRPO / PPO / RLVR / RLHF、奖励建模、信用分配、post-training |
| `llm` | 大模型与前沿实验室 | 模型发布与评测、预训练与推理技术、一手官方发布 |
| `agents` | 智能体与超级智能 | agent 与多智能体、工具使用、自我改进、long-horizon、AGI / 超级智能路线 |
| `infra` | AI Infra | 训练与推理系统、并行与显存、kernel 与量化部署、集群、吞吐与成本 |
| `auto-driving` | 自动驾驶 | 端到端、驾驶 VLA、BEV / occupancy、数据闭环与仿真 |
| `safety` | 安全与对齐 | interpretability、alignment、red teaming、评测与治理 |
| `misc` | 交叉与观点 | 立场论文、综述、开源工具、其他 AI 进展 |

约定：

- **顺序即展示顺序**：日报正文的分节顺序、`/news/` 上 chip 的顺序都按这个文件来。
  新增类别加在末尾，别插中间（历史条目的锚点依赖顺序）。
- 打标：`pick_topics.py` 按关键词给每条算各类得分，`category` 取最高分的类（**主分类**）。
  脚本给的是**建议**——人工精读时该改就改。
- front matter 里写 **ASCII key**：`cats: [embodied, world-model, rl]`，中文显示由模板映射。

### 2.1 为什么分节标题用 raw HTML

```markdown
<h2 id="cat-embodied">具身智能</h2>
```

**不要**写 `## 具身智能 {: #c-embodied}`。实测本机 kramdown 2.5.2 会把 `{: #id}` 当字面文字
留在标题里（渲染出「具身智能 {: #c-embodied}」），远程 GitHub Pages 同样不可靠。
raw HTML 的 `<h2>` id 100% 稳定，且 CSS 的 `.post-body h2` 照样命中。
站内长文里已有的 `<a id="ref-xxx"></a>` 也是同一套路。

---

## 3. 信源（24 个，2026-10-26 实测）

### 中文 / 论文 / 代码

| 信源 | 状态 | 拿到什么 | 依赖代理 |
| --- | --- | --- | --- |
| 青稞 `qingkeai.online` | ✅ | Halo 列表 API + 正文全文。中文解读主力，每篇自带 arXiv / 代码 / 署名 | 否 |
| HuggingFace Daily Papers | ✅ | 论文标题、摘要、AI 摘要、arXiv id、github 仓库、upvotes、关键词 | **是** |
| arXiv API | ✅ | 9 条检索式：WAM / VLA+具身 / RL+机器人 / sim-to-real / **AI Infra（推理 + 服务化两条）/ 自动驾驶 / agentic / RLHF**。间歇性抽风，见 §8 | 否（走代理更稳） |
| GitHub 仓库搜索 | ✅ | 按 push 时间筛活跃项目，带 star 数与 topics（7 组，含 Infra / 自驾 / agent） | **是** |
| 量子位 `qbitai.com` | ✅ | 中文 AI 媒体，覆盖国内发布与产业动态 | **是** |

### 前沿实验室 / 厂商官方博客（RSS）

| 信源 | 状态 | 依赖代理 |
| --- | --- | --- |
| OpenAI News | ✅ | **是** |
| Google Research Blog | ✅ | **是** |
| Microsoft Research | ✅ | **是** |
| Google DeepMind Blog | ✅ | **是** |
| NVIDIA Blog（两个 feed） | ✅ | **是** |
| BAIR Blog（Berkeley） | ✅（更新很慢，多数周期为空属正常） | **是** |
| HuggingFace Blog | ✅ | **是** |
| AWS Machine Learning Blog | ✅ | **是** |
| PyTorch Blog | ✅ | **是** |
| vLLM Blog | ✅ | **是** |
| Together AI Blog | ✅ | **是** |

### AI Infra / 观点 / 媒体（RSS）

| 信源 | 状态 | 备注 |
| --- | --- | --- |
| MarkTechPost | ✅ | 论文速讯，覆盖中文源常漏掉的欧洲实验室 |
| IEEE Spectrum Robotics | ✅ | 机器人垂直媒体 |
| Robohub | ✅ | 机器人学术社区转载 |
| Interconnects | ✅ | RL / post-training 方向时事 |
| Sebastian Raschka Magazine | ✅ | LLM / 研究解读 |
| LessWrong | ✅ | AGI / 超级智能 / 对齐讨论 |
| TechCrunch AI | ✅ | 产业动态，噪声较大（SOURCE_BONUS 偏低） |
| Ars Technica AI | ✅ | 产业动态，噪声较大 |

英文源的价值：中文源（青稞 / HF 论文）覆盖不到的**官方一手发布**与**欧美实验室动态**。
实测当天就靠 MarkTechPost 捞到 Kyutai CADRE、靠 NVIDIA Blog 捞到 Cosmos 工具链。

**代理是硬依赖**：`127.0.0.1:7890`（ClashX，系统代理设置里已启用）。
HF / GitHub / 各官方博客直连全不通。脚本启动时会先探代理，挂了直接报错退出。
不带代理跑：`DAILY_PROXY= python3 tools/daily/fetch_all.py`（青稞 + arXiv 仍可用）。

### 3.1 试过不行的源（别再加）

| 源 | 结果 |
| --- | --- |
| Anthropic News | 404，无 RSS |
| ai.meta.com | 400 / 无 feed |
| 机器之心 `jiqizhixin.com/rss` | 返回 HTML，不是 feed |
| Synced Review | feed 停在 2025，条目过期 |
| SemiAnalysis | feed 停在 2025 |
| Anyscale | feed 0 条 |
| Modal / xinzhiyuan / deargps / ImportAI | 不通 |
| InfoQ AI | 405 |
| 知乎 | 匿名访问 403（见 §3.2） |
| `therobotreport` / `venturebeat` / `ai.meta.com` | 403 / 429 / 无 feed |

### 3.2 知乎为什么没进采集器

匿名访问全部 403：网页 `/hot`、话题页、`api/v4/search_v3`、问答接口都返回
「您当前请求存在异常」。需要登录态，而本机没装 `agent-browser`。

**当前替代路径**：青稞正文里普遍附有作者的知乎原文链接，等于间接拿到了知乎内容。

---

## 4. 每天 08:00 的执行步骤

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io

# 1. 采集（24 源，7 天窗口）
python3 tools/daily/fetch_all.py --days 7

# 2. 筛选 + 分类打标（先看分类分布，别只看分数）
python3 tools/daily/pick_topics.py --top 20 --by-category
python3 tools/daily/pick_topics.py --top 8 --dump-body      # 看正文再定
python3 tools/daily/pick_topics.py --categories infra,auto-driving --top 10   # 只看某几类

# 3. 评分建议（可选）
python3 tools/daily/rate.py --top 20 --by-category

# 4. 生成按分类归组的骨架（可选，省掉排版）
python3 tools/daily/scaffold.py --top 12            # 默认写到 out/draft-<date>.md
python3 tools/daily/scaffold.py --top 12 --publish  # 直接生成 _daily/...（已存在需 --force）

# 5. 精读入选条目，写 _daily/YYYY-MM-DD-ai-daily.md（格式见 §5）
# 6. 校验（§6）
# 7. git add 该条目 && git commit && git push
```

某个源间歇性 reset 时重跑它**不会冲掉其他源**（默认按 source 合并）：

```bash
python3 tools/daily/fetch_all.py --only arxiv        # 只重跑 arxiv，其余保留
python3 tools/daily/fetch_all.py --only arxiv --no-merge  # 强制只留这一源
```

筛选器只是**初筛**。以下情况人工介入：

- 同一工作被多个源重复命中（青稞解读 + arXiv 原文 + HF）→ **合并成一条**，以解读为主、链接给全
- 直播预告 / 招聘 / 广告 → 直接丢弃（`NEGATIVE` 已降权，但会漏）
- 分数低但确实是重磅 → 手动捞回
- **分类打错要改**：脚本按关键词打分，跨领域的条目（比如具身 + RL）常常分错，`cats` 以人工判断为准
- **优先照顾多样性**：一期里别全是论文，也别全是某一类；每期尽量覆盖 2 个以上分类
- 星级参考 `rate.py` 的建议，但**自己定**（见 §5.3）

---

## 5. 条目格式

````markdown
---
title: AI 日报 · <月日>：<一句话点出当日主线>
date: YYYY-MM-DD
summary: 列表页显示的那一句话，40 字内
rating: 4
cats: [embodied, world-model, rl]
---

<h2 id="cat-embodied">具身智能</h2>

### <条目标题>

`值得读 ★★★★☆` ｜ `影响力 ★★★★☆`

**来源**：[青稞社区](链接) ｜ 论文：[arXiv:xxxx](链接) ｜ 代码：[repo](链接)

<3–5 句：解决什么问题、方法是什么、关键数字>

![<图的中文说明>](<图片 URL>)

<h2 id="cat-world-model">世界模型</h2>

### <第二条>
...

---

本期 N 条，来源：青稞社区 / HuggingFace Daily Papers / arXiv / NVIDIA Blog / 量子位。
````

硬性要求：

- **同一分类写在一起**：一个分类一节，节内是该分类的条目；分类顺序按 `_data/daily_categories.yml`
- `cats` 必须是词表里**出现过的 key**，按词表顺序排；没用到的分类不要写进去
- **数字必须有出处**：照抄原文并给链接；不换算、不估算
- **不确定的标「待验证」**，不把作者观点当结论
- **中文引号用全角**；一条 200–400 字，一期 8–12 条
- 不写元叙述（「本文由 XX 整理」不进正文，见 `WRITING.md` §4）
- 纯观点类内容写清「未见公开论文」

### 5.1 front matter 的 `rating`

本期**综合评分**（1–5），显示在 `/news/` 列表页标题后面。取本期各条目星级的大致均值，
再按「今天这期整体含金量」微调。没有 `rating` 字段就不显示星，不报错。

### 5.2 配图

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
- **特别有价值的图才下载到仓库** `assets/img/daily/`。单张控制在 200 KB 内

### 5.3 每条的星级

`值得读` 与 `影响力` 各 1–5 星，写在标题下一行。`rate.py` 会给建议分，但**它只看得懂
证据信号**（有没有数字 / 论文 / 代码 / 热度），看不懂「这个结果重不重要」——**最终星级由人定**。

| 星 | 值得读 | 影响力 |
|:--:|:--|:--|
| 5 | 必读，有硬数字 + 新路线 | 方向级突破或重磅开源 |
| 4 | 值得看，有对照实验数字 | 明显推进，别人会接着做 |
| 3 | 一般，有增量 | 增量改进 |
| 2 | 边缘相关，扫一眼即可 | 小修小补 |
| 1 | 凑数 | 几乎无影响 |

立场论文 / 综述默认 `值得读` 低（没实验数字）、`影响力` 未必低——这两个维度就是为区分
这种情况而设的。

---

## 6. 校验（写完必跑）

```bash
python3 tools/check_liquid.py _daily/<file>.md    # Liquid 冲突，必须过
python3 tools/scan_garbled.py _daily/<file>.md    # 乱码伪词
python3 tools/check_links.py                      # 站内链接（不传参数扫全站）
```

条目里若写了公式，还要跑 `python3 tools/check_post.py <file>`。
`check_tldr.py` 不需要跑（那是长文的 TL;DR 体检）。

另外自查两项（远程构建挂了才发现就晚了）：

- `cats` 里的 key 都在 `_data/daily_categories.yml` 里
- 分节用的是 `<h2 id="cat-xxx">`，不是 `{: #xxx}`

---

## 7. 定时

**DSH 会话提醒**（`schedule_create`，每天 08:00 Asia/Shanghai）驱动本流水线。
GUI / 会话需在线；关机或会话关闭会漏，漏了手动补跑：

```bash
python3 tools/daily/fetch_all.py --date 2026-10-27 --days 2
```

提醒的 prompt 就是 §4 那串步骤的精简版，改流水线记得同步改它。

---

## 8. 踩坑记录

- **`select.py` 会遮蔽标准库 `select`**。文件名改成 `pick_topics.py` 才正常。
  症状是 `AttributeError: module 'select' has no attribute 'select'`。
- **kramdown 的 `{: #id}` 不生效**，会当字面文字留在标题里。分节锚点一律写 raw `<h2 id="...">`。
- **`export.arxiv.org` 会间歇性 Connection reset / 502**。不是我们的错，等几秒重试通常能恢复。
  已改为多次重试 + 指数退避。真跑不出来时 HF 源已覆盖大部分论文，不必死等。
- **arXiv 检索式之间要 sleep 3s**（官方要求请求间隔），别并发。现在 9 条检索式，全量跑约 30s+。
- **arXiv RSS（`rss.arxiv.org/rss/<cat>`）是空壳** —— 只有 channel 头没有 item，别用。
- **BAIR feed 更新极慢**（实测最新 2026-07），7 天窗口内返回 0 条是常态，不是故障。
- **Halo 列表 API 带 `sort` / `fieldSelector` 参数返回 500**。只能 `?page=1&size=N` 裸取，本地排序。
- **代理挂掉的症状**是 HF / GitHub / 博客类源集体返回 0 条。脚本已加启动探测。
- **某些源「返回 0 条」是正常的**：`together`（实测最新停在 10-19，7 天窗口常为空）、
  `bair`（停在 2026-07）。采集器现在会区分两种 0 条并打到 stderr：
  `[info] xxx：feed 正常，N 天窗口内无新条目` 与
  `[warn] xxx 返回的不是可解析的 feed（可能反爬 / 登录墙…）`。
  看到前者别去动源，看到后者才说明该源被挡了。
- **arXiv 深夜容易被 reset**（`[Errno 54] Connection reset by peer` / SSL 握手超时），
  08:00 跑一般没问题；真全军覆没就 `--only arxiv` 重试一次，HF 源已覆盖大部分论文。
- **`pick_topics.py` 的 total 用九类命中分之和**，不是「取最高分类分」。
  改成只取最大值会让跨领域条目（具身 + RL）总分掉下阈值，被 `--min-score` 静默筛掉 ——
  分类是**打标**，不该顺手降低召回。
- 青稞正文抓取要带浏览器 UA；抓取间隔 0.4s。
- **分类关键词里别再把「量化 / 编译 / GPU / 芯片」当噪声**——AI Infra 现在是正经兴趣方向，
  降权只留给广告 / 招聘 / 直播预告。
