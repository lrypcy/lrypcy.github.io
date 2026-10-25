---
title: "Agent 技术剖析：那个 1.3 GB 的库里到底装了什么"
date: 2026-10-25 14:00:00 +0800
categories:
  - Agent技术剖析
tags: [agent, sqlite, fts5, query-plan, index, storage-cost]
layout: post
mathjax: true
---

> **系列导航** ｜ **Agent 技术剖析** ｜ 第 2 篇 / 共 2 篇已发
>
> [← 第 1 篇：会话状态放哪里——七个 agent 的存储设计横评](/2026/10/25/agent-anatomy-where-state-lives/)
>
> 上一篇回答「状态存在哪」（文件 / 库 / 混合）。这一篇往下一层：**这些库到底把字节花在了什么地方**，
> 以及写进去的每一条数据、建的每一个索引，在真实查询里有没有兑现。

> **TL;DR**
>
> * **核心问题与结论**：agent 库的体积不由消息条数决定，由两个选择决定——要不要把流式事件逐条落库，要不要为内容建全文索引。前者让 OpenCode 的 `event` 表独占全库 80%（§1）；后者让 Hermes 的双 FTS5 达到主表 `messages` 的 2.85 倍（§2）。
> * **关键证据**：`dbstat` 逐表实测（§1/§2）。OpenCode 13.4 万条事件 = 1.04 GB / 1.3 GB，1.16 万条消息只有 53.9 MB；Hermes 1.07 万条消息 48.8 MB，检索设施 139 MB。消息数几乎相同，两库体积差 6.6 倍（§4.1）。
> * **适用边界与代价**：全文索引的代价是约 3 倍存储（§2）加每次写入两遍 tokenize（§2.3）；事件溯源的代价是每次交互写十几行而非一行。不建索引就没有历史检索，不落事件就没有重放能力。这两笔钱只在产品真需要检索或重放时才值。
> * **系列定位**：承接上一篇的存储形态分类，本篇量化成本并用查询计划验证；下一篇转向上下文压缩的运行时代价。

---

## 0. 先把账摊开

上一篇列了七个 agent 各存了多少表、多大体量。但「1.3 GB」这种数字本身没有解释力——它可能是十万条对话，也可能是一张表在吞。

`dbstat` 能把字节归到具体的表和索引上。这是这一篇的全部方法：不问「多大」，问「**花在哪**」。

两个库的实测（皆为单次快照，见文末取证基准）：

| 库 | 文件大小 | `dbstat` 聚合 | 最大项 | 占比 |
| --- | --- | --- | --- | --- |
| OpenCode | 1.3 GB | 1287.4 MB | `event` 1040.0 MB | 80.8% |
| Hermes | 199 MB | 193.6 MB | FTS 设施合计 139.0 MB | 71.8% |

**两个数字都要报，因为含义不同**：文件大小含空闲页与未被 `dbstat` 计入的元数据页（Hermes 有 280 个空闲页），`dbstat` 聚合只算被表与索引真实占用的页。算「谁在吃磁盘」用后者，报「这个库多大」用前者。

这两行数字直接推翻了「agent 数据库主要用来存对话」的直觉——**两家都不是主要在存对话**。一家主要存事件，一家主要存索引。

---

## 1. OpenCode：事件溯源拿了 80%

逐表拆开（`dbstat`，单位 MB，只列前 12）：

```text
event                          1040.0
part                            161.5
message                          53.9
event_aggregate_type_seq_idx      9.5
event_aggregate_seq_idx           6.3
sqlite_autoindex_event_1          5.7
part_message_id_id_idx            3.7
part_session_idx                  2.3
sqlite_autoindex_part_1           2.1
message_session_time_created_id_idx 1.1
```

`event` 一张表 1.04 GB，是第二名的 6.4 倍。而 `event` 表只有 13.4 万行——**平均每行 7.9 KB**。

这个数字值得单独想一下。一条事件的 `data` 列存的是 JSON，8 KB 意味着里面装的不只是「用户发了一句话」，还包括流式输出的中间片段、工具调用的完整参数、工具结果的截断版本。**事件比消息细，所以事件比消息多，也就比消息占得多**。

### 1.1 为什么事件会比消息多十倍

上一篇提过这个比例，这里给出机制。

一条助手消息在用户眼里是「一个回复」。但系统要落库的是它的**全部状态迁移**：

```text
turn.start          -> 用户提交
step.start          -> 模型开始生成
part.created        -> 开始一个新 part
part.updated        -> part 内容增长（多次）
part.updated        -> ...
step.finish         -> 这一步结束，带 token 用量
step.start          -> 工具执行后继续生成
...
turn.finish
```

一次带两轮工具调用的交互，事件数轻松到十几条，而用户看到的「消息」只有两三条。实测 13.4 万事件对 1.16 万消息，约 **11.6 : 1**。

这就是事件溯源的核心成本：**它记录的是「发生了什么」，不是「现在是什么」**。前者天然比后者细一个数量级。

### 1.2 索引只占事件的 1.6%

`event` 相关的索引（两个显式索引 + 主键自动索引）合计约 21.5 MB，占全库 1.6%。这是个很健康的比例——说明**索引不是这个库膨胀的原因**，数据本身才是。

对比 Hermes 会很有意思：那边索引是主因。

### 1.3 `part` 与 `message` 的比例说明了拆分粒度

`part` 161.5 MB，`message` 53.9 MB，约 3 : 1。上一篇用行数算过这个比值（4.2 : 1），按字节算是 3.0 : 1。

两个比值接近，说明**平均每个 part 的体积和平均每条 message 的体积在同一个量级**——这验证了拆分的合理性：如果把 part 内联进 message，每条消息的体积会变成现在的 4 倍，而流式更新时每次都要重写这 4 倍大小的 JSON。**拆开是为了写入放大，不是为了查询**。

### 1.4 一个没被利用的索引

`event_aggregate_type_seq_idx`（9.5 MB）是 `(aggregate_id, type, seq)`。9.5 MB 说明它被真实维护着，每一行事件都进了两遍索引。

但它的服务对象——「按类型取某聚合的连续事件」——只在重放时才会用到，而重放是罕见操作。**为罕见查询付了 9.5 MB 常驻成本加每次写入的维护成本**，这是事件溯源实现里容易被忽略的一笔。

---

## 2. Hermes：索引比数据大三倍

Hermes 的账完全是另一种形状（单位 MB）：

```text
messages_fts_trigram_data      64.8
messages                       48.8
messages_fts_trigram_content   31.5
messages_fts_content           31.5
messages_fts_data              10.7
sessions                        4.4
idx_messages_session_active     0.5
```

拆开看：

- 主表 `messages`：**48.8 MB**，1.07 万条，平均每条 4.7 KB；
- trigram FTS（`data` + `content` + `idx` + `docsize`）：约 **97.5 MB**；
- 默认分词 FTS（`data` + `content` + `idx` + `docsize`）：约 **41.5 MB**；
- 两套合计 **139 MB**，是主表的 **2.85 倍**。

**「让历史可检索」这件事，花了数据本身近三倍的空间。**

### 2.1 为什么是两份 `content`

FTS5 有两种模式：内容存 FTS 表内部，或用 `content=` 指向外部表（外部内容模式，避免重复存储）。Hermes 用的是前者——所以 `messages_fts_content` 和 `messages_fts_trigram_content` 各存一份完整正文，各 31.5 MB。

外面那两份 31.5 MB 是纯冗余：正文在 `messages` 表里已经有了。**改用外部内容模式可以省掉这 63 MB**，代价是删除/更新时要靠 `content_rowid` 回主表校验，实现略复杂。

这是一个典型的取舍点：63 MB（全库 32%）换实现简单。对 199 MB 的库来说，这个决定值不值取决于机器——但至少现在知道了这笔账在哪。

### 2.2 trigram 比默认分词贵 2.35 倍

trigram 索引 97.5 MB vs 默认分词 41.5 MB。原因很直接：trigram 把文本切成三字符滑窗，**每个字符都参与 3 个 token**，索引项数量约为字符数的 3 倍。而默认分词按词切，token 数远少于字符数。

对一个 4.7 KB 的平均消息（中文按字算约几千字），trigram 的 token 数就是上万。**2.35 倍这个数字，基本就是「三字符滑窗 vs 按词切分」的 token 密度比**。

值不值？取决于产品要不要「中文子串搜索」。按词分词对中文几乎不可用（没有空格，整句当一个词），所以对一个中文用户为主的产品，trigram 不是可选项。

### 2.3 触发器：每次插入写三次

两个 FTS 表、六个触发器（insert / delete / update 各三个）。一条消息插入时实际执行：

```sql
-- 1. 主表
INSERT INTO messages (...) VALUES (...);
-- 2. 默认分词索引（触发器）
INSERT INTO messages_fts(rowid, content) VALUES (new.id, content || tool_name || tool_calls);
-- 3. trigram 索引（触发器）
INSERT INTO messages_fts_trigram(rowid, content) VALUES (new.id, same_concat);
```

**一次逻辑写入变成三次物理写入，而且第 2、3 次都要对全文做分词与索引构建**。这是 FTS 写入成本的来源——不在磁盘 IO，在 tokenize 的 CPU。

长消息上这个成本很可观：4.6 KB 的正文要过两遍 tokenizer（一遍按词，一遍每字符生成三个 trigram）。**这就是为什么「每条工具结果都进 FTS」的代价远比想象高**。

### 2.4 触发器把 `tool_name` 拼进正文，是个聪明的决定

```sql
COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
  || ' ' || COALESCE(new.tool_calls, '')
```

被索引的不是「助手说的话」，而是「说的话 + 工具名 + 工具参数」。这把「哪个会话跑过 pytest」变成一次 MATCH，而不是全表扫 JSON 解 `tool_calls`。

代价是索引体积：`tool_calls` 常常比 `content` 还长（一次工具调用带完整参数），拼进去后索引的语料变大。**这是用空间换查询能力的又一笔。**

---

## 3. 查询计划：索引有没有兑现

字节花出去了，接下来要问：**建了索引，优化器真的用了吗**。`EXPLAIN QUERY PLAN` 是唯一判据——它直接报出优化器选了哪条索引、要不要额外建临时 B 树。

### 3.1 OpenCode：两条主力路径干净，一条有缺口

```text
-- 取会话消息（agent 最热的路径，每轮循环都跑）
SELECT * FROM message WHERE session_id=? ORDER BY time_created, id
  → SEARCH message USING INDEX message_session_time_created_id_idx (session_id=?)

-- 重放事件
SELECT * FROM event WHERE aggregate_id=? AND seq > ? ORDER BY seq
  → SEARCH event USING INDEX event_aggregate_seq_idx (aggregate_id=? AND seq>?)
```

两条都只有 `SEARCH`、没有 `TEMP B-TREE`——索引完整覆盖了过滤和排序。这是设计到位的证据。

第三条露了缺口：

```text
SELECT * FROM session_input WHERE session_id=? AND promoted_seq IS NULL ORDER BY admitted_seq
  → SEARCH session_input USING INDEX session_input_session_promoted_seq_idx
  → USE TEMP B-TREE FOR ORDER BY
```

索引是 `(session_id, promoted_seq, delivery, admitted_seq)`，查询按 `admitted_seq` 排序。**排序键被 `delivery` 挡住了**——SQLite 只有在索引的前缀列上才能免排序，`admitted_seq` 排在第四位，前面有个不在 `WHERE` 里的 `delivery`，于是排序只能另外建临时 B 树。

修法有两种，各有代价：把 `admitted_seq` 提到 `delivery` 前面（但那样按 `delivery` 过滤就没法免排序），或者改查询按 `promoted_seq` 排序（但语义可能要求按提交顺序）。**索引列顺序不是随便排的，它编码了「哪些查询免排序」这个决定。**

### 3.2 Hermes：四条全中，包括那个部分索引

```text
-- 取活跃历史
SELECT * FROM messages WHERE session_id=? AND active=1 ORDER BY timestamp
  → SEARCH messages USING INDEX idx_messages_session_active (session_id=? AND active=?)

-- FTS 检索
SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'pytest'
  → SCAN messages_fts VIRTUAL TABLE INDEX 0:M1

-- 多通道路由（六列索引）
SELECT id FROM sessions WHERE source=? AND user_id=? AND chat_id=?
  AND chat_type=? AND thread_id=? ORDER BY started_at DESC LIMIT 1
  → SEARCH sessions USING INDEX idx_sessions_gateway_peer (source=? ... thread_id=?)

-- 部分索引
SELECT id FROM messages WHERE active IS NULL
  → SCAN messages USING COVERING INDEX idx_messages_active_null
```

第四条是关键。那个 `CREATE INDEX idx_messages_active_null ON messages(active) WHERE active IS NULL` 不只是「建着好看」：

- 优化器**认出查询谓词 `active IS NULL` 蕴含索引谓词**，于是选了它而不是全表扫；
- 因为查询只取 `id`（在主键里），走了 **`COVERING INDEX`——不需要回表**。

这两点纯读 schema 推不出来。部分索引能不能被选中，取决于优化器能否证明蕴含关系；能不能不回表，取决于查询列是否都在索引里。**都必须实测。**

顺带解释了那个索引存在的真实理由：主键 `AUTOINCREMENT` 的 `id` 在索引里，所以「找出所有 `active IS NULL` 的行」是一次纯索引扫描。这正好服务于迁移期的数据修复语句：

```sql
UPDATE messages SET active = 1 WHERE active IS NULL
```

### 3.3 WorkBuddy：一个真实的全表扫

```text
-- 轮询待投递（状态机核心）
WHERE status=? AND next_run_at<=? ORDER BY next_run_at
  → SEARCH automation_delivery_outbox USING INDEX idx_automation_delivery_outbox_status_next_run

-- 会话列表（最常打开的界面之一）
SELECT id,title FROM sessions WHERE cwd=? AND deleted_at IS NULL ORDER BY updated_at DESC
  → SCAN sessions
  → USE TEMP B-TREE FOR ORDER BY
```

第二条是**全表扫 + 临时排序**。原因：`sessions` 表上只有主键自动索引和一个 `buddy_snapshot_id` 索引，**没有任何覆盖 `cwd` 或 `updated_at` 的索引**。

现在表里 67 行，全扫的代价看不见。到几千行这个界面就会开始卡。

有意思的是它自己的迁移历史已经演示过这个问题一次——`0006_add_automation_runs_indexes.sql` 专门给 `automation_runs` 补了两个索引：

```sql
CREATE INDEX IF NOT EXISTS idx_automation_runs_automation_id ON automation_runs(automation_id);
CREATE INDEX IF NOT EXISTS idx_automation_runs_created_at     ON automation_runs(created_at);
```

**索引是事后补的，说明查询是在有了真实数据之后才变慢的**。按同一套逻辑，`(cwd, deleted_at, updated_at)` 是下一个该补的——`deleted_at` 也要进索引，因为软删除的过滤条件同样参与选择率。

---

## 4. 三种存储的真实价格

把字节账换算成可比的单位。同一个「存对话」的需求：

| 方案 | 体量 | 每条成本 | 能检索 | 能重放 |
| --- | --- | --- | --- | --- |
| Pi 纯 JSONL | 21 会话，数 MB 量级 | 最低 | 否（要扫全文件） | 否 |
| Hermes 全量 + 双 FTS | 1.07 万条 → 194 MB | 约 18 KB/条 | **是**（中文子串） | 否 |
| OpenCode 事件溯源 | 13.4 万事件 → 1.3 GB | 约 8 KB/事件 | 否（JSON 列） | **是** |

三个数字放在一起，结论很清楚：

**Hermes 花 19 KB/条买到了检索，OpenCode 花 8 KB/事件买到了重放，Pi 什么都没买所以最便宜。** 没有哪个「更优」——只有「你的产品要不要那个能力」。

但有一件事需要警惕：**OpenCode 花了 1.3 GB，却没买到检索**。它的正文全在 JSON 列里，没有任何 FTS。所以它那 1.3 GB 买的是「重放 + 统一读写路径」，不是「可查」。如果产品有搜索需求，这 1.3 GB 还得再加一遍 FTS 的钱。

### 4.1 一个反直觉的比较

Hermes 存 1.07 万条消息花 194 MB（`dbstat` 口径）；OpenCode 存 1.16 万条消息加 13.4 万事件花 1287 MB。**消息数量几乎一样，体积差 6.6 倍。**

差异全部来自两个选择：
1. OpenCode 把每条消息拆成多个事件（11.5 倍放大）；
2. Hermes 为内容建了两套索引（3 倍附加）。

**两个选择的方向相反**：一个把钱花在「记录的粒度」，一个花在「检索的深度」。这也解释了为什么两家的表结构几乎没有相似之处——它们优化的是不同的东西。

---

## 5. 选型：先问三个问题

前面的账可以压缩成三个判据。

### 5.1 会不会有人搜索历史？

不会 → 不要 FTS。省下 3 倍空间和每次写入的 tokenize 成本。

会，而且用户说中文 / 搜代码片段 → **必须要 trigram**。按词分词的 FTS5 对中文基本无效（整句当一个词），对代码片段也搜不到（`kv_cache_utils` 和 `kv cache utils` 在按词分词下是不同 token）。代价是 trigram 比默认分词贵 2.35 倍。

会，但只是英文自然语言 → 默认分词够用，别为 trigram 付钱。

### 5.2 需不需要按时刻重放？

不需要 → **不要事件溯源**。它会让你每次交互写十几行而不是一行，并且那 80% 的体积是纯记录成本。

需要（审计、调试、精确复现 bug、多人协作的冲突消解）→ 事件溯源是对的，但要注意：**重放能力不等于查询能力**。事件表的 `data` 是 JSON，想按内容查还得另建索引。OpenCode 就是这个状态——有重放、无检索。

### 5.3 单机会不会跑成服务？

会（多通道、后台任务、定时投递）→ 必须入库，因为你需要的不是存储而是**事务**：谁在跑、跑完了没、消息发了没。这部分数据量极小（WorkBuddy 整个库 208 KB），但离了事务就不对。

不会（一个进程一次运行）→ 文件足够。别忘了加锁（DSH 的 `session.lock` 就是这个），否则两个进程在同一目录会互相覆盖。

---

## 6. 三个可以立刻用的结论

**① 体积失控的两个源头是「事件粒度」和「全文索引」，不是「消息条数」。** 看到库大了，先 `dbstat` 看哪张表在吃，别猜。

**② 建的索引要用 `EXPLAIN QUERY PLAN` 验一遍。** 本文实测的三个缺口（OpenCode 排序键被挡、WorkBuddy 全表扫、以及 Hermes 部分索引确实被选中）都是纯读 schema 看不出来的。索引列顺序直接编码了「哪条查询免排序」，排错了不会报错，只会慢。

**③ 落地前想清楚检索需求，因为它是 3 倍空间的开关。** 决定不建 FTS 之后再加，意味着一次全库重建索引；决定建了却没人用，就是常驻的 3 倍成本加每次写入的 tokenize。

---

## 7. 下一篇

存储层的账算完了。下一篇转向**运行时**：上下文是怎么被挑选、压缩、再注入的——Pi 的摘要记录、Hermes 的 `active`/`compacted` 双标记、OpenCode 的 `context_epoch` 区间替换，三种方式在每一轮循环里各自要付出什么。

---

上一篇：[Agent 技术剖析：会话状态放哪里——七个 agent 的存储设计横评](/2026/10/25/agent-anatomy-where-state-lives/)

---

## 参考

**取证基准（各数字的来源版本）**

| 项目 | 版本 | 引用位置 |
| --- | --- | --- |
| OpenCode | `1.18.35`，commit `b7e4f1ef7433f83ba009eefa2997aeb81017f6ed` | `packages/core/src/session/sql.ts` L22–L176、`packages/core/src/event/sql.ts` L4–L25 |
| Hermes | `0.19.0` | `hermes_state.py` L155（`SCHEMA_VERSION`）、L178（`_WAL_INCOMPAT_MARKERS`）、L4456（压缩软删）、L1981（会话 upsert）；上游 `NousResearch/hermes-agent` |
| WorkBuddy | `5.5.6` | 迁移 SQL `~/.workbuddy/.workbuddy-sqlite-migrations/0006_add_automation_runs_indexes.sql`；库 `~/.workbuddy/workbuddy.db` |
| Pi | `0.84.2` | `~/.pi/agent/sessions/` |

**测量方法**：`dbstat` 虚表按表/索引聚合 `pgsize`；`EXPLAIN QUERY PLAN` 在对应实库上直接执行。两个库的 `page_size` 均为 4096，OpenCode 页数 329583、Hermes 页数 49834。所有数字为单次快照——会话在持续增长，**稳定的是表间比例，不是绝对字节**。
