---
title: "Agent 技术剖析：会话状态放哪里——七个 agent 的存储设计横评"
date: 2026-10-25 09:00:00 +0800
categories:
  - Agent技术剖析
tags: [agent, storage, sqlite, event-sourcing, context-compaction, jsonl]
layout: post
mathjax: true
---

> **系列导航** ｜ **Agent 技术剖析** ｜ 第 1 篇
>
> 本系列读真代码、真库、真文件：把 coding agent 的内部实现拆开看。这一篇只回答一个问题——
> 一个 agent 每轮循环到底要把什么东西写下来、写到哪里去。

> **TL;DR**
>
> * **核心问题与结论**：agent 持久状态分五类：会话树、消息与工具记录、压缩结果、用量计费、需事务的任务投递。七家给出四种形态——纯 JSONL（Pi、Claude Code）、压缩 JSONL（DSH）、元数据入库正文出库（WorkBuddy、Codex）、全量入库（OpenCode、Hermes）；没有一家把五类都放进数据库。
> * **关键证据**：分界线是「要不要跨会话查询与事务」。OpenCode 用 13.4 万事件 + 1.16 万消息支撑事件溯源，库 1.3 GB（均见 §2 实测表）；Hermes 用 1.07 万条消息撑出 199 MB 并挂两套 FTS5；WorkBuddy 库仅 208 KB，只放会话索引与投递 outbox，正文全在 JSONL。
> * **适用边界与代价**：入库换来跨会话检索、事务与崩溃后可恢复的任务状态，代价是 JSON 列让关系查询退化、库随转录线性膨胀；文件方案零依赖、可 diff、可分支，代价是无索引、无并发写保护、压缩有损。库不是更专业，是更贵但更可查。
> * **系列定位**：本篇建立存储层坐标系，后续篇在同一套状态上继续。

---

## 0. 一个循环，五类状态

所有 coding agent 的主循环都是同一件事：把上下文喂给模型，模型回工具调用，执行工具，结果写回上下文，重复。这个循环本身不需要存储——它需要的是**崩溃之后还能接着跑**，以及**用户下次打开还能看见上次的内容**。

于是问题变成：循环里有哪些东西一旦丢失就无法重建？逐条数下来只有五类：

| # | 状态 | 丢失后果 | 是否可重建 |
| --- | --- | --- | --- |
| 1 | 会话与消息树 | 找不到上次的对话，无法续跑 | 不可 |
| 2 | 消息与工具调用记录 | 上下文无法复原，工具输出丢失 | 不可（工具执行有副作用） |
| 3 | 上下文压缩结果 | 压缩后必须重算，token 白花 | 可重算但很贵 |
| 4 | 用量与计费 | 账单对不上，预算控制失效 | 不可 |
| 5 | 需要事务的任务与投递 | 定时任务重复执行、消息漏发 | 不可 |

第 1、2 类构成**转录**（transcript），体量占绝对大头；第 4、5 类各自只有几十行到几千行，却恰恰是最需要 ACID 的部分。这个体量差异，几乎解释了后面所有的设计分歧。

---

## 1. 判据：什么时候该上数据库

在展开七个实现之前，先立一个判据，否则「谁的设计更好」无从谈起。存储方案要回答四个问题，成本依次上升：

**能不能追加写就够？** 转录是只追加的：一条消息写进去就不再改。只追加的数据用追加文件最省——没有锁、没有事务、没有 schema 迁移，写到一半崩了最多丢最后一行。

**需要按会话随机读吗？** 需要。所以至少要能把「哪个会话的哪些行」高效取出来。文件方案在这里退化：按目录分片（一个会话一个文件）能解决 90% 的读取，剩下的 10% 是所有「跨会话」操作——搜索、统计、列表排序。

**需要跨会话查询吗？** 这是分水岭。只要用户问过「我上周在哪个项目里改过这段代码」，就必须要索引；只要有索引，就必须有事务来维护它。FTS5 是这条路上的典型产物。

**需要事务性副作用吗？** 定时任务的「正在运行」标志、外部消息的「恰好发一次」、预算的「已经花了多少」——这三类必须在崩溃后保持一致，文件做不到（除非自己实现 WAL，那就是在重造数据库）。

四个问题的答案从「是是是否」到「是是是是」，对应从纯文件到全量入库的整个谱系。七家 agent 恰好把这个谱系铺满了。

---

## 2. 七家的存储清单

先看全貌。下表每一行的数字都来自本机安装的真实数据，版本钉死（统计为单次快照：会话数在持续增长，绝对行数会变，**稳定的是量级与比值**）：

| Agent | 版本 | 主存储 | 库/目录体积 | 转录形态 |
| --- | --- | --- | --- | --- |
| OpenCode | 1.18.35 | SQLite（WAL） | 1.3 GB | 全量入库 + 事件溯源 |
| Hermes | 0.19.0 | SQLite（WAL）×3 | 199 MB | 全量入库 + 双 FTS5 |
| WorkBuddy | 5.5.6 | SQLite（元数据）+ 文件 | 208 KB + 数百 MB | 元数据入库，正文出库 |
| Codex CLI | 本机 rollout | SQLite（goals）+ 文件 | 数十 KB + 数十 MB | 目标入库，转录出库 |
| Pi | 0.84.2 | 无数据库 | 纯文件 | 追加 JSONL |
| Claude Code | 本机 project 目录 | 无数据库 | 纯文件 | 追加 JSONL |
| DSH | 0.2.0-rc.2 | 无数据库 | 纯文件 | 压缩 JSONL + 锁文件 |

七家里只有四家用了 SQLite，而用 SQLite 的四家里两家只存元数据。**「数据库」在这个领域不是默认选项，是一个需要论证的选择**。

### 2.1 全量入库：OpenCode

OpenCode 把一切都塞进 `~/.local/share/opencode/opencode.db`。库里 21 张表，真实的分布是这样的：

```text
event                134353
part                  48923
message               11629
todo                    551
session                 386
session_message         130
migration                38
__drizzle_migrations     20
project                  12
project_directory        11
data_migration            1
permission / workspace / session_input / session_context_epoch / credential  ...  0
```

`event` 表 13.4 万行、`part` 表 4.9 万行——**事件数量是消息数量的十倍以上**。这不是巧合，是事件溯源的必然结果：一条助手消息由「开始」「若干 part 更新」「结束」多个事件拼成，每个事件一行。

库旁边还并存着文件存储：`storage/`（1.3 MB）、`snapshot/`（62 MB，以 git sha 命名的目录）、`log/`（13 MB）。数据库不是唯一存储，只是**权威**存储。

### 2.2 全量入库：Hermes

Hermes 的 `~/.hermes/state.db` 是 199 MB，但里面只有约 1.07 万条消息：

```text
sessions               430
messages             10701
session_model_usage    302
gateway_routing         19
delivery_obligations     7
state_meta               2
schema_version           1
compression_locks        0
async_delegations        0
```

**1.07 万条消息撑出 199 MB**，平均每条约 19 KB——因为它存的是带完整工具调用参数、推理内容、API 原始载荷的多通道助手历史，且几乎所有列都是 TEXT。这个数字本身就说明了「全量入库」的真实成本结构：不是行数贵，是每行的 JSON 贵。

### 2.3 元数据入库：WorkBuddy 与 Codex

WorkBuddy 走的是完全相反的路线。`workbuddy.db` 只有 208 KB、10 张表：

```text
sessions                                       67
session_usage                                  66
buddy_snapshots                                 1
workspaces                                      1
automations / automation_runs / automation_runtime_state / automation_delivery_outbox  ... 0
migration_meta                                 13
__workbuddy_drizzle_migrations                 10
```

**67 条会话索引、66 条用量记录，一句对话正文都没有**。正文在 `~/.workbuddy/projects/<压平的项目路径>/<sessionId>.jsonl`，也是 67 个文件，一一对应。此外还有 `.meta.json` 边车、`.file-rollback.ndjson` 回滚记录、`<sessionId>/tool-results/` 目录和 `subagents/` 子目录。

转录 JSONL 的记录类型分布（本机 67 个文件合计）：

```text
function_call            8315
function_call_result     8291
reasoning                4770
file-history-snapshot    2778
message                  2545
ai-title                   34
```

工具调用与工具结果各有 8300 余条，而「消息」只有 2545 条——**agent 的转录里，工具往返是消息的三倍以上**。谁把每条工具结果都当成一行消息入库，库就会先被工具输出撑爆。

Codex CLI 是同构的更小版本：`~/.codex/sessions/` 下 55 个 rollout JSONL，配一个 `goals_1.sqlite`——库里只有三张表（`_sqlx_migrations`、`thread_goals`、`thread_goal_continuation_deferrals`），只管「目标与预算」这一小块需要状态机的数据。

### 2.4 纯文件：Pi、Claude Code、DSH

Pi 完全不用数据库。`~/.pi/agent/sessions/<压平的项目路径>/<ISO 时间戳>_<uuidv7>.jsonl`，21 个会话文件，记录类型只有五种：

```text
message                1811
model_change             24
thinking_level_change    21
session                  21
compaction                2
```

Claude Code 同样是 `~/.claude/projects/<压平路径>/*.jsonl`，记录类型是 `assistant` / `user` / `attachmentqueue`-类操作 / `file-history-snapshot` / `system` / `last-prompt`。

DSH 多走了一步：`~/.dsh/sessions/<压平路径>/<sessionId>/` 下是 `session.v4.jsonl.zstd` 加一个 0 字节的 `session.lock`。**压缩 + 锁文件**，格式自带版本号（`v4`），首行是自描述的会话头。

三家的共同点是：**零依赖、人可读、可 diff**。Pi 的会话文件可以直接进 git，可以手工改，可以被另一个 agent 读。

---

## 3. 会话树：同一件事的三种表示

agent 的对话不是线性的。用户会回退、会分支、会从中间某个点重新开始，子代理还会挂出一个独立的分支。于是「消息列表」其实是一棵树，七家用了三种表示。

### 3.1 父指针链（文件派）

Pi 的每条记录都带 `id` 与 `parentId`：

```json
{"type":"message","id":"407e7b7e","parentId":"ca4a8cc7",
 "timestamp":"...","message":{...}}
```

会话首行是一个不含 `parentId` 的 `session` 记录，它就是根。当前叶子节点由外部状态记录，模型看到的上下文 = 从叶子沿 `parentId` 回溯到根的路径。**分支的成本只是多一个 `parentId` 指向同一个祖先**，不需要任何额外结构。

这个设计的好处是它天然支持三种操作，且都不需要改历史：

- 回退：把叶子移回祖先，旧分支原地保留；
- 分支：新消息的 `parentId` 指向旧祖先；
- fork 会话：复制文件、改叶子。

代价是「列出这个会话的所有分支」必须扫全文件建索引，文件里没有任何索引。

WorkBuddy 的 JSONL 用的是同一套，但多一个 `logicalParentId`——**逻辑父**。物理 `parentId` 维持文件内的写入顺序，`logicalParentId` 表达语义上的对话结构。两个指针并存，说明它的写入路径里有「一条记录物理上追加在末尾、逻辑上属于更早的位置」的情况，典型来源是异步工具结果回填。

### 3.2 父键列（库派）

入库的两家把树写成列。OpenCode 的 `session` 表：

```sql
CREATE TABLE `session` (
  `id` text PRIMARY KEY,
  `project_id` text NOT NULL,
  `parent_id` text,
  `slug` text NOT NULL,
  ...
  CONSTRAINT `fk_session_project_id_project_id_fk`
    FOREIGN KEY (`project_id`) REFERENCES `project`(`id`) ON DELETE CASCADE
);
CREATE INDEX `session_project_idx`  ON `session` (`project_id`);
CREATE INDEX `session_parent_idx`   ON `session` (`parent_id`);
CREATE INDEX `session_workspace_idx` ON `session` (`workspace_id`);
```

树的层级只到**会话级**（`parent_id` 指向父会话，用于子会话/分支），消息级另有一套 `session_message.seq` 全序。这跟文件派的区别是根本性的：文件派用树表达「上下文是什么」，库派用树表达「会话之间什么关系」，而消息顺序用一个单调序号 `seq` 表达。

Hermes 同理，`sessions.parent_session_id` 配 `idx_sessions_parent`，但它把「谁是谁的延续」扩展成了多通道语义：`session_key`、`chat_id`、`chat_type`、`thread_id`、`parent_session_id` 五个字段共同定位一个对话线程，对应索引是六列的

```sql
CREATE INDEX idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
```

六列索引说明它的典型查询是「这个平台上这个用户这个群这个话题里最近一次会话」——**这是多通道助手的查询形状，不是 CLI 的**。

### 3.3 事件流里的树（事件溯源派）

OpenCode 还有第三种：树不落在消息上，落在事件上。`event_sequence` 给每个聚合根分配单调 `seq`：

```sql
CREATE TABLE `event_sequence` (
  `aggregate_id` text PRIMARY KEY,
  `seq` integer NOT NULL,
  `owner_id` text
);
CREATE TABLE `event` (
  `id` text PRIMARY KEY,
  `aggregate_id` text NOT NULL,
  `seq` integer NOT NULL,
  `type` text NOT NULL,
  `data` text NOT NULL,
  CONSTRAINT `fk_event_aggregate_id_event_sequence_aggregate_id_fk`
    FOREIGN KEY (`aggregate_id`) REFERENCES `event_sequence`(`aggregate_id`) ON DELETE CASCADE
);
CREATE UNIQUE INDEX `event_aggregate_seq_idx`      ON `event` (`aggregate_id`,`seq`);
CREATE INDEX        `event_aggregate_type_seq_idx` ON `event` (`aggregate_id`,`type`,`seq`);
```

注意 `seq` 的分配者是 `event_sequence` 表而不是消息表——**事件序号由聚合根统一发放，这就是并发写的串行化点**：两个进程同时写同一个会话，抢的是 `event_sequence` 里那一行。`WITHOUT ROWID` 没用到，但 `aggregate_id` 做主键已经让这行极小。

第二个索引 `(aggregate_id, type, seq)` 是给「读某类事件的连续区间」用的——重放时按 type 过滤，正是这个形状。

---

## 4. 消息模型：拆开、整块、还是索引化

### 4.1 拆开：message + part

OpenCode 的 v1 模型把一条助手消息拆成 `message` 和 `part`：

```sql
CREATE TABLE `message` (
  `id` text PRIMARY KEY,
  `session_id` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL,
  CONSTRAINT `fk_message_session_id_session_id_fk`
    FOREIGN KEY (`session_id`) REFERENCES `session`(`id`) ON DELETE CASCADE
);
CREATE TABLE `part` (
  `id` text PRIMARY KEY,
  `message_id` text NOT NULL,
  `session_id` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL,
  ...
);
CREATE INDEX `message_session_time_created_id_idx` ON `message` (`session_id`,`time_created`,`id`);
CREATE INDEX `part_message_id_id_idx`              ON `part` (`message_id`,`id`);
CREATE INDEX `part_session_idx`                    ON `part` (`session_id`);
```

`message` 存元信息（角色、模型、用量），`part` 存内容片段（文本、工具调用、文件 diff、推理）。拆分的动机是**流式更新**：助手回复是逐块到达的，每块是独立的 `part`，写入是「插入一行」而不是「改一个不断增长的字符串」。

三个索引各自对应一条真实查询：

- `(session_id, time_created, id)`：按会话取全量消息并保持顺序；带上 `id` 是为了在 `time_created` 相同的毫秒内仍有确定顺序；
- `(message_id, id)`：取一条消息的所有 part；
- `(session_id)`：直接按会话删 part、按会话统计。

注意 `data` 是 `text` 存 JSON，**不是结构化列**。这意味着「查所有调用过某个工具的消息」无法走索引，只能扫 `part` 全表再解 JSON。这是全文 JSON 列的固有代价，后面 §8 会展开。

### 4.2 整块：文件派的一行一个 JSON

文件派没有这个问题，因为它们根本不打算查询——一条记录就是一个完整 JSON 对象，字段全部内联。代价是**结构随版本漂移无人管**：Pi 用首行的 `version: 3`、DSH 用文件名里的 `v4` 声明格式版本，读取端必须分支处理多个版本。

Pi 的首行：

```json
{"type":"session","version":3,"id":"<uuidv7>","timestamp":"...","cwd":"<项目路径>"}
```

### 4.3 索引化：Hermes 的双 FTS5

Hermes 走的是第三极：消息入库，且**为内容建两套全文索引**。

```sql
CREATE VIRTUAL TABLE messages_fts USING fts5(content);
CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
```

两套索引都由触发器维护，而不是应用层代码：

```sql
CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content,'') || ' ' || COALESCE(new.tool_name,'')
          || ' ' || COALESCE(new.tool_calls,'')
    );
END;
-- delete / update 触发器的形状相同：先 DELETE 再 INSERT
```

三个细节值得记住：

**为什么两套**。`fts5` 默认分词器按词切分，对英文和空格分隔的语言有效；`trigram` 分词器按三字符滑窗切分，对中文这种没有空格的文字，以及「找一段代码片段」这种子串 search 是必需的。**一套给语言，一套给子串**——这不是冗余，是两个不同的检索需求。

**为什么用触发器**。索引与主表的一致性交给了数据库，应用层任何写入路径（包括开发者随手写的插入语句）都不会漏掉索引。反过来，`content=` 外部内容表语法可以避免正文存两份，这里没用——所以 `content` 在两套 FTS 表里各存一份，这是 199 MB 里相当大的一块。

**为什么把 `tool_name` 与 `tool_calls` 拼进被索引的正文**。因为用户搜的往往是工具名或命令片段（「哪个会话跑过 pytest」），而不是助手说的话。这个决定把「检索工具调用」变成一次 MATCH 而不是全表扫 JSON。

### 4.4 事件溯源与投影：两套模型并存

OpenCode 的库里同时存在 v1（`message` + `part`）与 v2（`session_message`）：

```sql
CREATE TABLE `session_message` (
  `id` text PRIMARY KEY,
  `session_id` text NOT NULL,
  `type` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL,
  `seq` integer NOT NULL,
  ...
);
CREATE UNIQUE INDEX `session_message_session_seq_idx`       ON `session_message` (`session_id`,`seq`);
CREATE INDEX        `session_message_session_type_seq_idx`  ON `session_message` (`session_id`,`type`,`seq`);
CREATE INDEX        `session_message_session_time_created_id_idx`
    ON `session_message` (`session_id`,`time_created`,`id`);
```

v2 不是 v1 的替代，是 v1 的**投影**。源码里投影器写得很直白，它把事件流折成表：

```ts
export class SessionAlreadyProjected extends Error {}
...
// A newer turn supersedes stale incomplete rows; never resume an older assistant projection.
```

这条注释说的是幂等性：同一个事件被重放两次，第二次不能把已经完成的投影改回「进行中」。对应的实现是先查已有行、再决定插入还是更新，冲突时抛 `SessionAlreadyProjected` 让上层忽略。

本机库里的实测分布正好印证「迁移尚未完成」：`message` 1.16 万行、`part` 4.9 万行，而 `session_message` 只有 130 行。**新模型已上线，旧数据仍在旧表**，两条读路径并存——这是所有「在线迁移存储模型」的工程常态，也是设计成本的真实来源：不是写新表难，是要同时维护两套投影和一套回填。

---

## 5. 上下文压缩：三种持久化方式

上下文窗口是有限的，长会话必须压缩。压缩本身不新鲜，新鲜的是**压缩这件事在存储里长什么样**——它决定了「压缩之后还能不能回去看原始记录」。

**方式一：写入一条压缩记录（Pi）**。压缩不删历史，只追加一条 `compaction` 记录，带 `summary` 字段：

```json
{"type":"compaction","id":"...","parentId":"...","timestamp":"...","summary":"<摘要正文>"}
```

它在树里的位置就是新的「根」：模型上下文 = 从叶子回看到最近的 compaction 记录为止。原始记录仍在文件里，但不在上下文路径上。**压缩是无损的存储、有损的视图**。这是三种方式里唯一能反悔的。

**方式二：标记而不删除（Hermes）**。`messages` 表有两列专门干这件事：

```sql
"active"    INTEGER NOT NULL DEFAULT 1,
"compacted" INTEGER NOT NULL DEFAULT 0,
```

压缩时把旧消息 `active` 置 0（或 NULL），新消息照常插入。表里不删行，所以 FTS 索引仍然搜得到历史——**这正是双 FTS 的价值所在：压掉的上下文仍然可检索**。索引也为此专门优化：

```sql
CREATE INDEX idx_messages_session_active ON messages(session_id, active, timestamp);
CREATE INDEX idx_messages_active_null    ON messages(active) WHERE active IS NULL;
```

第二个是**部分索引**：只索引 `active IS NULL` 的行。既然大部分行 `active = 1`，为它们建索引纯属浪费，部分索引把索引体积压到只覆盖「被压缩掉的那部分」。为什么单独为 `NULL` 建一个，而不是并入上一句？因为 `WHERE active = 0` 与 `WHERE active IS NULL` 在三值逻辑下是两个不同谓词，SQLite 不会自动复用。

**方式三：引入上下文纪元（OpenCode）**。`session_context_epoch` 表：

```sql
CREATE TABLE `session_context_epoch` (
  `session_id` text PRIMARY KEY,
  `baseline` text NOT NULL,
  `snapshot` text NOT NULL,
  `baseline_seq` integer NOT NULL,
  `replacement_seq` integer,
  `revision` integer DEFAULT 0 NOT NULL,
  ...
);
```

`baseline` 是压缩前的基线，`snapshot` 是压缩后的快照，`baseline_seq` / `replacement_seq` 是两个位置标记，`revision` 做乐观并发控制。**压缩被建模成一次「区间替换」**：`[baseline_seq, replacement_seq)` 这段事件被 `snapshot` 取代。这个表示的威力在于它跟事件溯源严丝合缝——压缩本身也是一条事件，重放时按区间跳过旧事件即可。

本机这份库里 `session_context_epoch` 是 0 行，说明这条路径刚上线还没被真实使用。设计已经就位，数据还没到。

三种方式的判据很清楚：**要不要能回看原始记录**（Pi 能、其余不能）；**压掉的内容还要不要可检索**（Hermes 要、其余不要）；**压缩要不要对重放保持幂等**（OpenCode 要、其余不需要）。

---

## 6. 真正需要事务的那一小块

前面五类状态里，前四类其实都可以用追加写糊过去。真正逼着大家回到数据库的是第五类：**需要跨崩溃保持一致性的小状态**。

### 6.1 投递 outbox：WorkBuddy 的定时任务

WorkBuddy 有个「定时任务把结果推到微信/企微」的功能。这类功能有三个致命点：任务不能重复执行、消息不能漏发、失败要重试。它的 `automation_delivery_outbox` 把这三件事一次解决：

```sql
CREATE TABLE automation_delivery_outbox (
    id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'wechatmp',
    automation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    automation_name TEXT NOT NULL,
    owner_user_id TEXT,
    host_id TEXT,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    next_run_at INTEGER NOT NULL,
    lease_owner TEXT,
    lease_expire_at INTEGER,
    last_error_code TEXT,
    last_error_message TEXT,
    delivery_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    finished_at INTEGER
);

CREATE UNIQUE INDEX automation_delivery_outbox_dedupe_key_unique ON automation_delivery_outbox(dedupe_key);
CREATE INDEX idx_automation_delivery_outbox_status_next_run
    ON automation_delivery_outbox(status, next_run_at);
CREATE INDEX idx_automation_delivery_outbox_channel_status_next_run
    ON automation_delivery_outbox(channel, status, next_run_at);
CREATE INDEX idx_automation_delivery_outbox_run
    ON automation_delivery_outbox(automation_id, run_id);
CREATE INDEX idx_automation_delivery_outbox_owner_status
    ON automation_delivery_outbox(owner_user_id, status);
CREATE INDEX idx_automation_delivery_outbox_supersede_lookup
    ON automation_delivery_outbox(automation_id, channel, owner_user_id, host_id, created_at);
```

这张表几乎是教科书式的外发消息表，每一列都在解决一个具体问题：

- `dedupe_key` 上的**唯一索引**就是「恰好一次」的实现：重复入队直接撞唯一约束，不靠应用层判断；
- `lease_owner` + `lease_expire_at` 是**租约**：消费者领取时写自己的 id 和过期时间，进程崩了租约自然过期，别的进程能接手。这替代了「需要一个独立的队列服务」；
- `next_run_at` 是重试调度：失败后按退避策略往后推，消费循环的查询就是 `WHERE status='pending' AND next_run_at <= now`，正好命中 `(status, next_run_at)` 索引；
- `attempt_count` / `max_attempts` 区分「可重试」与「永久失败」；
- 最后一个复合索引 `(automation_id, channel, owner_user_id, host_id, created_at)` 名字里带 `supersede`——**用它查找「同一个任务在同一通道上是否需要被新消息取代」**，按时间倒序取最近一条。

六个索引，全部服务一个功能。这是「库不是给转录用的，是给状态机用的」最完整的例子。

同一个库里还有配套的两张表，把「崩溃检测」做成了数据库字段：

```sql
CREATE TABLE automation_runtime_state (
    automation_id TEXT PRIMARY KEY,
    last_run_at INTEGER,
    last_error TEXT,
    running INTEGER NOT NULL DEFAULT 0,
    running_started_at INTEGER,
    running_conversation_id TEXT,
    metadata_json TEXT
);
```

`running = 1` 且 `running_started_at` 很久以前 = 上次运行崩了。不需要外部健康检查，状态就在库里。

迁移历史里还有个细节：`automation_runs` 的两个索引不是在建表时创建的，而是第 6 个迁移文件单独补的：

```sql
CREATE INDEX IF NOT EXISTS idx_automation_runs_automation_id ON automation_runs(automation_id);
CREATE INDEX IF NOT EXISTS idx_automation_runs_created_at     ON automation_runs(created_at);
```

**索引是事后补的，说明查询是在有了真实数据之后才变慢的**。这条迁移本身就是一次性能问题的化石。

### 6.2 库内锁与至少一次投递：Hermes

Hermes 是常驻的多通道助手（微信、飞书、Telegram、语音），它的两个问题跟 WorkBuddy 同源，解法也同源，但更激进——**把锁和投递都做成表**：

```sql
CREATE TABLE compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX idx_compression_locks_expires ON compression_locks(expires_at);
```

压缩是贵的（要调模型），而且不能两个进程同时压同一个会话——不然两边都花了 token，还互相覆盖。用一张表 + 过期时间实现分布式锁，`expires_at` 上的索引给清理任务用。

```sql
CREATE TABLE delivery_obligations (
    obligation_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    thread_id TEXT,
    content TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    last_error TEXT
);
```

`owner_pid` + `owner_started_at` 是**进程级租约**：不仅记录哪个进程在发，还记录那个进程的启动时间。如果当前同 pid 的进程启动时间变了（pid 复用），说明原进程已经死了，投递可以回收。这个细节是长期跑在用户机器上的守护进程才会想到的——`pid` 单独不够，必须配启动时间。

### 6.3 目标状态机：Codex 的 goals 库

Codex CLI 用 55 个 JSONL 存转录，却为一个东西单开了 SQLite：

```sql
CREATE TABLE thread_goals (
    thread_id TEXT PRIMARY KEY NOT NULL,
    goal_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'active','paused','blocked','usage_limited','budget_limited','complete'
    )),
    token_budget INTEGER,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    time_used_seconds INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
CREATE TABLE thread_goal_continuation_deferrals (
    thread_id TEXT PRIMARY KEY NOT NULL
        REFERENCES thread_goals(thread_id) ON DELETE CASCADE
);
```

六个状态用 `CHECK` 约束而不是枚举表，`token_budget` 与 `tokens_used` 放在同一行——**因为「超预算就停」必须是原子的**。如果预算在文件里、状态在另一处，崩溃窗口内就会超支。这是又一个「小状态、硬约束」的例子。

值得单独说的是 `thread_goal_continuation_deferrals`：一张只有一列、只记 `thread_id` 的表。这是**关系型的集合成员判断**（「这个线程的续跑已被延期」），用一张表表达，比在主表加一个布尔列更干净——加列会让主表每次查询都多带一个字段。

---

## 7. 索引：七家怎么做，为什么

把三家的索引放在一起看，规律就出来了。

### 7.1 索引形状 = 查询形状

| 系统 | 索引 | 服务的查询 |
| --- | --- | --- |
| OpenCode | `message(session_id, time_created, id)` | 取会话全量消息，稳定排序 |
| OpenCode | `session_message(session_id, seq)` 唯一 | 按序号取消息，挡住重复写入 |
| OpenCode | `event(aggregate_id, seq)` 唯一 | 重放单个聚合的事件 |
| OpenCode | `session_input(session_id, promoted_seq, delivery, admitted_seq)` | 取「还没进上下文的用户输入」 |
| Hermes | `messages(session_id, timestamp)` | 取会话历史 |
| Hermes | `sessions(session_key, started_at DESC)` | 取某通道最新会话 |
| Hermes | `sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC)` | 多通道路由 |
| WorkBuddy | `automation_delivery_outbox(status, next_run_at)` | 扫描待投递 |
| WorkBuddy | `sessions(buddy_snapshot_id)` | 反查引用了某个配置快照的会话 |

三条规律：

**① 第一条永远是「按会话取全量，且带排序」。** 这是 agent 的核心读取路径：每一轮循环都要把整个会话重新读一遍喂给模型。所以索引的第一列必然是 `session_id`，且必然带 `time_created`/`timestamp`/`seq` 让排序免于临时文件。

**② 排序键一定要进索引，且要放在能「连续区间扫描」的位置。** 对比 `(session_id, time_created)` 与 `(time_created, session_id)`：前者能对单个会话做区间扫描，后者不能。这就是为什么 OpenCode 的 `session_message` 有 `session_message_time_created_idx` 这个「只有一列」的索引——它服务的是**跨会话按时间取**（比如「最近的会话活动」），与前面那个复合索引是两条不同的查询。

**③ 唯一索引用来表达业务不变式，不只是加速。** OpenCode 的 `session_message_session_seq_idx`、`session_input_session_admitted_seq_idx`、Hermes 的 `idx_sessions_title_unique ... WHERE title IS NOT NULL`——这些索引的真正作用是把「不许重复」「标题不许撞」交给数据库。Hermes 那个部分唯一索引尤其讲究：`WHERE title IS NOT NULL` 让未命名的会话可以有任意多个（它们的 `title` 是 NULL，不参与唯一性），只有命名之后才要求唯一。

### 7.2 索引里没有列的部分，就是代价

把三家的索引数量与表数量放在一起，比值很说明问题：OpenCode 21 张表 17 个索引，Hermes 9 张表 16 个索引，WorkBuddy 10 张表 10 个索引。**Hermes 的索引/表比最高，因为它要为多通道路由和检索买单**；WorkBuddy 次之，因为它的索引用在投递状态机上；OpenCode 最低，因为它的大部分查询都固定在「按会话取」这一种形状上。

反过来，三家里**没有一张转录表建了内容索引**（除了 Hermes 的 FTS）——因为正文在 JSON 列里，建了也用不上。这就是 §4 那个决定的必然结果。

---

## 8. JSON 列的代价，被低估的那一半

四家入库的系统，全部把内容存成 JSON 文本列：OpenCode 的 `message.data` / `part.data` / `event.data` / `session_message.data`，Hermes 的 `tool_calls` / `origin_json` / `task_json` / `event_json` / `result_json`，WorkBuddy 的 `plugin_context_json` / `skills_json` / `connector_ids_json` / `payload_json` / `metadata_json`。

这么做的好处很实在：**schema 不用为每个工具改**。工具参数千变万化，如果每个工具一套列，迁移会变成噩梦。JSON 列把「可变结构」挡在关系模型之外，这是所有 agent 存储的共同选择。

代价有四条，每条都能在真实库里看到后果：

**① 无法按内容过滤。** 「找出所有调用 `bash` 且命令里含 `rm -rf` 的消息」只能全表扫 + 解 JSON。这正是 Hermes 要双 FTS5 的原因——它是唯一一家真正回答了「用户会搜索」这个问题的。

**② 无法按内容聚合。** 「这个会话花了多少 token」在 Hermes 里是列（`input_tokens` 等），在 OpenCode 里能从 `session` 表的 `tokens_input` 列读，但在 WorkBuddy 里只能解 `credit_json`。**同样的统计需求，三家用了三种存储位置**。

**③ 体积膨胀。** JSON 的键名重复存储、缩进、转义，实际体积普遍是二进制编码的数倍。Hermes 平均每条消息约 19 KB 就是明证——其中相当一部分是重复的键名与嵌套结构。

**④ 部分更新要读改写。** 流式响应时，助手消息的 `part` 会不断增长。如果一条消息只有一个 JSON 列，每次追加都要读出整列、改、写回，写入量与消息体积成正比。OpenCode 用 `part` 表把粒度拆细（每个 part 一行、独立更新），正是为了把这个开销降到单个 part 的量级。这不是理论推演——`part` 表 4.9 万行 vs `message` 表 1.16 万行的比例（约 4.2 : 1），就是这个拆分粒度在真实数据里的体现。

`part` 拆分还有第三个好处：**工具输出可以单独存储**。一次 `cat` 大文件可能产生几百 KB 输出，如果塞进消息 JSON，整条消息的读写都变贵；拆成独立 part 后，只要读取时按需跳过。WorkBuddy 走得更远，直接把工具结果放到了文件系统的 `tool-results/` 目录——**库只留引用**。

---

## 9. 迁移：三种节奏

存储设计最容易被忽略、又最能看出工程成熟度的是迁移机制。

### 9.1 声明式 + 版本表：OpenCode

OpenCode 用 drizzle 生成迁移（`__drizzle_migrations` 表 20 行记录已应用的版本），表定义本身在源码里：

```ts
export const SessionTable = sqliteTable("session", { ... },
  (table) => [
    index("session_project_idx").on(table.project_id),
    index("session_workspace_idx").on(table.workspace_id),
    index("session_parent_idx").on(table.parent_id),
  ],
)
```

声明式的好处是「schema 即代码」，索引与列在同一个地方定义，改一处就能生成迁移。代价是**生成器的历史包袱**：本机库里既有 `__drizzle_migrations`（20 行）又有 `migration`（38 行）还有 `data_migration`（1 行）三张迁移记录表——**说明迁移系统本身被换过一次，旧表没删**。这是声明式迁移的典型遗迹：表好改，数据迁移的账难平。

### 9.2 手写 + 版本号：Hermes

Hermes 完全不依赖迁移框架，靠一个整数：

```python
SCHEMA_VERSION = 22

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
```

启动时读版本、逐级跑 `ALTER TABLE ... ADD COLUMN`、最后写回版本号。库里那些「看起来重复」的列就是这个机制的产物——`sessions` 表的 DDL 里，前 40 列是 `CREATE TABLE` 里的，后面 15 列是历次 `ALTER` 追加的（`session_key`、`chat_id`、`cwd`、`git_branch`、`rewind_count`、`archived` 等），SQLite 把它们都塞进同一段 DDL 文本里。

手写迁移的好处是**零依赖、完全可控**，代价是：22 个版本意味着 22 次人工判断，每次都要想清楚「旧库里有数据怎么办」。`SCHEMA_VERSION` 从 1 涨到 22，实际成本不在写 SQL，在每次都要兼容三种状态（新库、v21 老库、v21 里手动改过列的库）。

### 9.3 文件名 + journal：WorkBuddy

WorkBuddy 用 drizzle 的文件式迁移，迁移 SQL 直接落在用户目录里：

```text
0000_workbuddy_sqlite_baseline.sql
0001_session_plugin_context.sql
0002_add_addon_selection.sql
0002_last_user_prompt_expert_selection.sql     <- 重号
0003_add_session_settings.sql
0004_last_user_prompt_expert_selection.sql     <- 重复加列
0005_add_session_context_window.sql
0006_add_automation_runs_indexes.sql
0007_add_buddy_snapshots.sql
0008_add_session_thought_level.sql
0009_add_automation_context_window.sql
```

两个异常放在一起看很有信息量：`0002` 重号，`0004` 又加了一次 `last_user_prompt_expert_selection` 列。对同一个表重复 `ALTER TABLE ADD COLUMN` 会报错，而迁移显然跑过去了——**说明它的 runner 容忍「column already exists」这类错误**，或者按 `meta/_journal.json` 里的 hash 判断而非文件名顺序。

这种设计的好处是迁移可审计（SQL 文件就在那儿，用户能看）；代价是**编号与顺序解耦之后，「第几个」不再等于「先后」**，只能靠 journal 兜底。这是文件式迁移的固有脆弱点。

### 9.4 文件派：版本号写在数据里

不用数据库的三家，把版本写进了文件本身：

- Pi：首行 `{"type":"session","version":3,...}`，读取时按版本分支；
- DSH：文件名 `session.v4.jsonl.zstd`，版本在外层；
- Claude Code、WorkBuddy 的 JSONL：靠「未知字段忽略」的松耦合，没有显式版本。

Pi 与 DSH 的做法说明一个事实：**只追加的格式也必须做版本化**，只是粒度从「整库一个版本号」变成「每个文件一个版本号」，而且升级是惰性的——老文件不迁移，读到哪个版本就按哪个版本解释。这在文件方案里成本几乎为零，在入库方案里就是一次全库搬迁。

---

## 10. 取舍：四个象限与选型判据

把七个实现放到「是否需要跨会话查询」×「状态是否需要事务」两个轴上：

```text
                    需要事务
                        ▲
        WorkBuddy [*]   │   [*] Hermes
        （元数据+outbox）│   （全量+锁+投递）
                        │
   ─────────────────────┼─────────────────────→
   不需要跨会话查询      │        需要跨会话查询
                        │
              Pi [*]    │   [*] OpenCode
        Claude Code [*] │   （全量+事件溯源
              DSH [*]   │     +投影）
                        │
                        ▼
                    不需要事务
```

四个象限对应四种设计哲学，没有优劣，只有匹配：

**左下（纯文件）**：适合 CLI 工具。会话是「一个进程一次运行」的产物，用户不需要在几百个会话里搜索。零依赖意味着能装进任何环境，可 diff 意味着能被 git 管理，可读意味着能被别的 agent 读——**这是「工具」而不是「服务」的设计**。

**左上（元数据入库）**：适合桌面应用。需要「最近打开的会话列表」「用量统计」「定时任务」这些跨会话但不需要全文检索的功能，但没有「搜索历史对话内容」的强需求。库小到 208 KB 意味着**备份、同步、迁移都便宜**。

**右下（全量入库无事务）**：其实是过渡态。OpenCode 把转录全量入库、又不给检索建索引，说明它当前的目标是「统一读写路径」而不是「可检索」；`session_message` 只有 130 行、`session_context_epoch` 0 行，说明新架构还在路上。

**右上（全量+事务）**：适合常驻服务。Hermes 要同时服务四个聊天平台、要在后台压缩、要保证消息不漏发——这三件事各自都需要事务，于是数据库从「存储」变成了「协调中心」。199 MB 的体积是它的入场费。

三条选型判据，可以拿去套任何 agent 项目：

**判据一：转录与状态分开。** 体量大且只追加的转录，和体量小且需保持一致的状态，成本结构完全不同。七家里设计最成熟的两家（WorkBuddy、Codex）都做了这种分离：转录走文件/JSONL，事务性状态走数据库。反过来，把它们塞进同一张表，就会得到「一个 1.3 GB 的库，其中 99% 的数据不需要事务」——这不是错，但要知道自己在付什么。

**判据二：需要检索才建索引，需要索引才入库。** Hermes 的双 FTS5 是唯一真正回答「用户要搜索」的设计。如果产品里没有搜索框，全量入库就只是把文件的复杂度换成了数据库的复杂度。

**判据三：压缩策略决定存储形态。** Pi 的 compaction 记录要求「历史必须保留」；Hermes 的 `active=0` 要求「历史保留且可检索」；OpenCode 的 epoch 要求「历史可重放」。三种要求对应三种存储，反过来说，**先定压缩策略，再定存储**，顺序反了就要重写。

### 10.1 各家设计的具体优缺点

**OpenCode**：优点是模型最完整——事件溯源 + 投影 + 纪元压缩，理论上支持完整的重放与审计；索引设计克制且精准，每条都对得上查询。缺点是复杂度最高（21 张表、三张迁移记录表、v1/v2 双模型并存）、`data` 全 JSON 导致无法检索、1.3 GB 的库对一个 CLI 来说太重。

**Hermes**：优点是唯一认真做了检索（双 FTS5 + 触发器同步 + 部分索引），事务性设施（锁、投递、预算）最完备，多通道路由的索引形状精准。缺点是 22 版手写迁移的维护成本、正文字段在三套 FTS 里重复存储、`sessions` 表 55 列（大量 ALTER 追加）已接近「宽表反模式」、199 MB 单文件无归档策略。

**WorkBuddy**：优点是把「什么该入库」切得最清楚——库里只有索引、用量、任务、投递，正文全在文件；outbox 的六个索引是教科书级；快照表用三个独立版本字段（`schema_version` / `template_version` / `snapshot_format_version`）表达配置的三层兼容性，这个细节很少见。缺点是没有 FTS（历史搜索只能靠外部索引目录）、迁移文件重号、日志与 trace 的体积失控（1.1 GB 日志 + 388 MB trace 对一个 208 KB 的库）。

**Codex CLI**：优点是最小惊讶——转录就是 rollout 文件，只有「目标与预算」入库且带 `CHECK` 约束，职责边界最干净。缺点是 goals 库与转录分离后，「goal 状态」与「转录内容」的一致性只能靠应用层保证。

**Pi**：优点是所有方案里唯一「压缩可反悔」的（compaction 只追加不删）、树结构用 `parentId` 一个字段搞定分支与回退、零依赖零迁移、文件可直接进 git。缺点是无任何索引（搜索退化为扫全部文件）、并发写无保护（同 cwd 两个进程会互相覆盖）、压缩摘要是纯文本无法结构化查询。

**Claude Code**：优点是与 Pi 同构的简洁性，`file-history-snapshot` 记录把「文件修改前的内容」也纳入了转录，回滚不需要额外的快照系统。缺点是格式无版本号，靠字段松耦合演进。

**DSH**：优点是唯一做了压缩的（`zstd`），长会话的磁盘占用最低；`session.lock` 文件解决了 Pi 的并发问题；格式版本在文件名里，读取端不用解析就能判断。缺点是压缩后不可人读、不可 diff，牺牲了文件派最大的优势。

---

## 11. 小结

七个 agent、四种存储形态，收敛出三条结论。

**存储形态由「查询需求」决定，不由「数据重要性」决定。** 转录是最重要的数据，却几乎都放在文件里；投递记录最不重要，却一定在数据库里。因为前者只需「按会话顺序读回」，后者需要「跨崩溃保持一致」。

**库的体积与索引数量反向相关。** OpenCode 1.3 GB 只有 17 个索引，Hermes 199 MB 有 16 个——**索引少不是懒，是查询形状单一**。真正让库变大的是「全量入库 + 无检索需求」这个组合。

**压缩策略是存储设计的上游。** 先决定「压缩后能不能回看、能不能搜、能不能重放」，再决定存哪儿。反过来做，就要重写存储层。

下一篇从这套状态继续往上：上下文是怎么被挑选、压缩、再注入的——三种压缩方式（Pi 的摘要记录、Hermes 的 `active` 标记、OpenCode 的纪元替换）在**运行时**各自要付出什么。

---

## 参考

**源码与版本（引用即锁定，行号以锁定版本为准）**

| 项目 | 版本 / commit | 上游 | 引用位置 |
| --- | --- | --- | --- |
| OpenCode | `1.18.35`，commit `b7e4f1ef7433f83ba009eefa2997aeb81017f6ed` | `anomalyco/opencode` | `packages/core/src/session/sql.ts` L22–L176、`packages/core/src/event/sql.ts` L1–L25、`packages/core/src/session/projector.ts` L1–L40 |
| Hermes | `0.19.0` | `NousResearch/hermes-agent` | `hermes_state.py` L155（`SCHEMA_VERSION`）、L758（`schema_version` 建表）、L1621–L1626（版本读写）、L1871–L1874（版本回写）；`hermes_cli/kanban_db.py`、`hermes_cli/projects_db.py` |
| Pi | `0.84.2` | `@earendil-works/pi-coding-agent` | `dist/`（发布即编译产物），会话格式见本机 `~/.pi/agent/sessions/` |
| WorkBuddy | `5.5.6`（`com.tencent.workbuddy.mac`） | 闭源 | 迁移 SQL：`~/.workbuddy/.workbuddy-sqlite-migrations/0000_workbuddy_sqlite_baseline.sql` … `0009_add_automation_context_window.sql`、`meta/_journal.json`；库：`~/.workbuddy/workbuddy.db` |
| Codex CLI | 本机 `~/.codex` | 闭源安装 | `~/.codex/goals_1.sqlite`（`.schema`）、`~/.codex/sessions/**/*.jsonl` |
| Claude Code | 本机 `~/.claude` | 闭源安装 | `~/.claude/projects/*/*.jsonl` |
| DSH | `0.2.0-rc.2` | `@deepseek-ai/dsh` | `~/.dsh/sessions/*/<id>/session.v4.jsonl.zstd`、`session.lock` |

**OpenCode 行的行号引用**

- `packages/core/src/session/sql.ts` L22–L66（`SessionTable`）、L68–L80（`MessageTable`）、L82–L97（`PartTable`）、L100–L116（`TodoTable`）、L119–L138（`SessionMessageTable`）、L140–L166（`SessionInputTable`）、L168（`SessionContextEpochTable`）
- `packages/core/src/event/sql.ts` L4–L25（`EventSequenceTable`、`EventTable` 与两个索引）
- `packages/core/src/session/projector.ts` L18（`SessionAlreadyProjected`）、L136（投影幂等注释）

**说明**：七家里四家的版本号与库/目录体积、行数分布，均为本机安装版本的实测统计；闭源三家（WorkBuddy、Claude Code、Codex CLI）的结论来自其落盘产物（库 schema、迁移 SQL、JSONL 记录结构），不含反编译内容。
