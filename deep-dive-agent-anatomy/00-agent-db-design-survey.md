# 主流 Agent 数据库设计调研（四问）

> 素材与草稿，不发布（`_config.yml` 已 exclude）。
> 对应已发文章：`_posts/2026-10-25-agent-anatomy-where-state-lives.md`
> 本文档补齐该文未展开的 **DML 与查询计划** 部分，并给出后续篇的取证基准。

---

## 0. 口径

**四问**：① 定义了哪些数据库？② 每个库的 DDL 与 DML 是什么？③ 实际怎么用、建了哪些索引？④ 设计优缺点？

**取证方式分三类**，引用时不可混用：

| 类别 | 可引用的东西 | 不可引用 |
| --- | --- | --- |
| 开源本地仓（OpenCode） | 源码行号 + commit permalink | 从 main 引用 |
| 安装包源码（Hermes / Pi / DSH） | `site-packages` 或 `dist/` 路径 + 版本号 | 无 commit 的「最新」 |
| 闭源落盘产物（WorkBuddy / Claude Code / Codex） | schema、迁移 SQL、JSONL 记录结构 | 反编译内容 |

**硬约束**：所有行数/体积都是**时间快照**。本机 agent 常在跑，数字会漂（实测过 429→430、1987→8315 的漂移），落笔前必须重测，正文写明「稳定的是量级与比值」。

---

## 1. 第一问：定义了哪些数据库

| Agent | 版本 | 数据库 | 位置 | 附带文件存储 |
| --- | --- | --- | --- | --- |
| OpenCode | 1.18.35 | `opencode.db`（1.3 GB，WAL，21 表 17 索引） | `~/.local/share/opencode/` | `storage/` 1.3 MB、`snapshot/` 62 MB（git sha 目录）、`log/` 13 MB、`tool-output/` |
| Hermes | 0.19.0 | `state.db`（199 MB，WAL，9 表 16 索引）；`kanban.db`（112 KB）；`verification_evidence.db`（32 KB） | `~/.hermes/` | `cron/`、`cache/`、`channel_directory.json`、`auth.json`、`.hermes_history` |
| WorkBuddy | 5.5.6 | `workbuddy.db`(208 KB，10 表 10 索引)；`edge-sync-mapping-v2.db` / `-v4.db`；`security/threat-database/*.db` | `~/.workbuddy/` | `projects/*.jsonl`（67 个）、`traces/` 388 MB、`logs/` 1.1 GB、`file-history/` 16 MB |
| Codex CLI | 本机 | `goals_1.sqlite`（3 表，sqlx 迁移） | `~/.codex/` | `sessions/**/*.jsonl`（55 个）、`history.jsonl` |
| Pi | 0.84.2 | **无** | — | `sessions/<压平 cwd>/<ISO>_<uuidv7>.jsonl`（21 个） |
| Claude Code | 本机 | **无** | — | `projects/*/*.jsonl`（10 个）、`file-history/`、`history.jsonl` |
| DSH | 0.2.0-rc.2 | **无** | — | `sessions/<压平 cwd>/<id>/session.v4.jsonl.zstd`（51 个）+ `session.lock` |

**结论**：7 家里 4 家用 SQLite，但**其中 2 家只存元数据**（WorkBuddy 208 KB、Codex 3 表）。数据库不是默认选项。

---

## 2. 第二问：DDL 与 DML

### 2.1 DDL（已在正文详列，此处只登记要点）

**OpenCode**：`session` / `message` / `part` / `todo` / `session_message` / `session_input` / `session_context_epoch` / `event` / `event_sequence` / `project` / `project_directory` / `workspace` / `permission` / `account` / `account_state` / `control_account` / `credential` / `session_share` / `data_migration` / `migration` / `__drizzle_migrations`。**三张迁移记录表并存**（20 / 38 / 1 行）＝ 迁移系统换过一次。

**Hermes**：`sessions`（55 列，大量 ALTER 追加）/ `messages` / `messages_fts` + `messages_fts_trigram`（两个 FTS5 虚表 + 六个触发器）/ `session_model_usage` / `gateway_routing` / `compression_locks` / `async_delegations` / `delivery_obligations` / `state_meta` / `schema_version`。

**WorkBuddy**：`sessions` / `workspaces` / `automations` / `automation_runs` / `automation_runtime_state` / `automation_delivery_outbox` / `session_usage` / `buddy_snapshots` / `migration_meta` / `__workbuddy_drizzle_migrations`。**workbuddy.db 无任何 FK**（`PRAGMA foreign_key_list` 全空）。

**Codex**：`thread_goals`（6 状态 CHECK 约束 + token 预算）/ `thread_goal_continuation_deferrals`（单列集合成员表）/ `_sqlx_migrations`。

### 2.2 DML：写入路径的真实形状（本次新增取证）

#### Hermes：三种 upsert 与软删除双语义

**① 会话 upsert 用 COALESCE 只补 NULL，不覆盖。** 源码注释说明了动机：先 `create_session` 建行、后补真实 model，用 `INSERT OR IGNORE` 会把富信息丢掉。

```sql
INSERT INTO sessions (id, source, ..., started_at) VALUES (...)
ON CONFLICT(id) DO UPDATE SET
    model = COALESCE(sessions.model, excluded.model),
    model_config = COALESCE(sessions.model_config, excluded.model_config),
    ...
```

`COALESCE(旧值, 新值)` 保证「后面一个只带 `source="unknown"` 的裸调用不会冲掉真实值」。`hermes_state.py:1981`。

**② 计费 upsert 走复合主键。** `session_model_usage` 的 PK 是 `(session_id, model, billing_provider, billing_base_url, billing_mode, task)` 六列，写入用 `ON CONFLICT(...) DO UPDATE`（`hermes_state.py:3077`）。六列 PK 的意义：同一会话同一模型在不同 provider / 不同计价模式下是**不同的账单行**，混在一行就没法对账。

**③ 路由 upsert 走 `(scope, session_key)`**（`hermes_state.py:2102`）。

**软删除的双语义**（这是全文最有价值的一处设计细节，`hermes_state.py:4450` 注释）：

```sql
-- 压缩：active=0 + compacted=1  →「被摘要掉了」
-- 回退：active=0 + compacted=0  →「用户主动收回」
UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ? AND active = 1;
```

判据：**检索默认包含 `compacted=1`**（压缩前的转录仍可搜到），但**加载上下文只取 `active=1`**（压掉的不进上下文）。两种「消失」用两个字段区分，而不是一个布尔。

**④ 一次性的数据修复**：`UPDATE messages SET active = 1 WHERE active IS NULL`（`hermes_state.py:1603`）——把历史 NULL 归位。这就是那个 `idx_messages_active_null ... WHERE active IS NULL` 部分索引存在的第四个理由：**给迁移期的脏数据兜底**。

#### OpenCode：事务与序号分配

- 事务通过 Effect 的 `db.transaction(...)` 包装（`packages/core/src/database/migration.ts:26`、`:71`），连接层在 `database/sqlite.node.ts:117` / `sqlite.bun.ts:123` 实现 `transactionAcquirer`。
- 迁移本身在事务里跑（`yield* db.transaction(...)`），因此**迁移失败会整体回滚**。
- `event.seq` 由 `event_sequence` 表统一发放（`event/sql.ts:4`），**这是并发写的串行化点**。

#### WorkBuddy：靠唯一约束与租约，不靠应用层判断

DML 逻辑在闭源 `app.asar` 里，但表结构已经决定了写法：
- 入队靠 `dedupe_key` UNIQUE → `INSERT ... ON CONFLICT DO NOTHING`；
- 领取靠 `UPDATE ... SET lease_owner=?, lease_expire_at=? WHERE id=?`；
- 重试靠 `next_run_at` 递推 + `attempt_count` 自增。

**无法从落盘产物证实以上三条的具体 SQL**，标注为 INFERRED（由索引形状与列语义推断），写作时不得当作已验证事实。

---

## 3. 第三问：实际用法与索引（本次新增：查询计划实测）

索引是否真的被用上，`EXPLAIN QUERY PLAN` 是唯一可靠证据。以下均为实跑结果。

### 3.1 OpenCode（4 条核心查询）

| 查询 | 计划 | 结论 |
| --- | --- | --- |
| 取会话消息 `WHERE session_id=? ORDER BY time_created,id` | `SEARCH message USING INDEX message_session_time_created_id_idx (session_id=?)` | ✅ 索引全包排序，无临时表 |
| 重放事件 `WHERE aggregate_id=? AND seq>? ORDER BY seq` | `SEARCH event USING INDEX event_aggregate_seq_idx (aggregate_id=? AND seq>?)` | ✅ 区间扫描 |
| 待处理输入 `WHERE session_id=? AND promoted_seq IS NULL ORDER BY admitted_seq` | `SEARCH session_input USING INDEX session_input_session_promoted_seq_idx` + **`USE TEMP B-TREE FOR ORDER BY`** | ⚠️ 命中索引但仍需临时排序 |
| 按时间取跨会话活动 | 走 `session_message_time_created_idx`（单列） | ✅ 与复合索引分工明确 |

⚠️ 那条 `session_input` 查询暴露一个缺口：索引是 `(session_id, promoted_seq, delivery, admitted_seq)`，而查询按 `admitted_seq` 排序，**排序键不在索引前缀位置**，所以省不掉临时 B 树。修法是在索引末尾补 `admitted_seq` 或改成按 `admitted_seq` 排序。

### 3.2 Hermes（4 条核心查询，全部命中）

| 查询 | 计划 | 结论 |
| --- | --- | --- |
| 取活跃历史 `WHERE session_id=? AND active=1 ORDER BY timestamp` | `SEARCH messages USING INDEX idx_messages_session_active (session_id=? AND active=?)` | ✅ 无临时表 |
| FTS 检索 `MATCH 'pytest'` | `SCAN messages_fts VIRTUAL TABLE INDEX 0:M1` | ✅ 走 FTS 索引 |
| 多通道路由（六列） | `SEARCH sessions USING INDEX idx_sessions_gateway_peer (source=? ... thread_id=?)` | ✅ 六列全用上 |
| `WHERE active IS NULL` | **`SCAN messages USING COVERING INDEX idx_messages_active_null`** | ✅ **部分索引真的被选中，且是覆盖索引（不回表）** |

最后一条是**部分索引价值的直接证明**：SQLite 优化器认出了 `WHERE active IS NULL` 与部分索引的谓词等价，且查询只取 `id`（在主键里）所以走覆盖扫描。这条如果不实测，纯靠读 schema 是推不出来的。

### 3.3 WorkBuddy（3 条，发现一个真实缺陷）

| 查询 | 计划 | 结论 |
| --- | --- | --- |
| outbox 轮询 `WHERE status=? AND next_run_at<=? ORDER BY next_run_at` | `SEARCH ... USING INDEX idx_automation_delivery_outbox_status_next_run` | ✅ 无临时表 |
| 归属查询 `WHERE owner_user_id=? AND owner_status=? AND deleted_at IS NULL` | `SEARCH automations USING INDEX idx_automations_owner` | ✅ 三列全用上 |
| dedupe 命中 `WHERE dedupe_key=?` | `SEARCH ... USING INDEX automation_delivery_outbox_dedupe_key_unique` | ✅ 唯一索引 |
| **会话列表** `WHERE cwd=? AND deleted_at IS NULL ORDER BY updated_at DESC` | **`SCAN sessions` + `USE TEMP B-TREE FOR ORDER BY`** | ❌ **全表扫 + 临时排序** |

**这是本次调研唯一发现的明确性能缺陷**：`sessions` 表没有任何 `(cwd, updated_at)` 或 `(deleted_at, updated_at)` 索引。工作区会话列表是这个应用最常打开的界面之一，却走全扫。

现在只有 67 行所以看不出来；一旦到几千行就会卡。**这也解释了他们为什么在 `0006` 迁移里补 `automation_runs` 的索引——同一个模式：索引是事后补的。** 按同样的逻辑，`(cwd, deleted_at, updated_at)` 很可能是下一个要补的。

---

## 4. 第四问：优缺点

### 4.1 逐家

**OpenCode**（全量入库 + 事件溯源）
- 优：模型最完整，事件溯源 + 投影 + 纪元压缩三层齐备；索引克制精准，核心查询零临时表；迁移在事务里，失败可回滚。
- 缺：21 表 + 三张迁移记录表 + v1/v2 双模型并存，复杂度最高；`data` 全 JSON 导致无法检索；`session_input` 索引少了排序键；1.3 GB 对 CLI 过重。

**Hermes**（全量入库 + 双 FTS5）
- 优：唯一认真做检索的（trigram 解决中文与代码子串）；查询计划四条全命中，部分索引被优化器正确选中；软删除双语义设计精细；`checkpoint_fullfsync` 在 macOS 上补住了 fsync 屏障（注释里直接引 issue）；WAL 不兼容时回退 DELETE 而不是崩。
- 缺：55 列宽表；22 版手写迁移；正文在三套 FTS 里重复存；199 MB 单文件无归档。

**WorkBuddy**（元数据入库）
- 优：「什么该入库」切得最准——库只有索引/用量/任务/投递；outbox 六个索引是教科书级；`buddy_snapshots` 用三个独立版本字段表达三层兼容。
- 缺：**会话列表全扫**（已实测）；无 FTS；迁移文件重号（两个 `0002`、`0004` 重复加列）；日志 1.1 GB + trace 388 MB 相对 208 KB 的库严重失衡；workbuddy.db 无 FK。

**Codex CLI**（目标入库）
- 优：职责边界最干净（转录 rollout、只把带状态机的 goals 入库）；`CHECK` 约束把状态机写进 schema；单列 deferrals 表表达集合成员。
- 缺：goals 与转录分离后一致性靠应用层；库极小说明「需要事务的状态」本就很少——反过来说，专门开一个 SQLite 是否值得，是设计判断题。

**Pi / Claude Code / DSH**（纯文件）
- 优：零依赖零迁移；可 diff、可进 git、可被别的 agent 读；Pi 的 compaction 只追加不删，是**唯一压缩可反悔**的；DSH 加 zstd 把长会话磁盘占用压到最低，且用 `session.lock` 补上并发保护。
- 缺：无索引（搜索退化为扫全部文件）；无并发写保护（Pi）；压缩牺牲可读性（DSH）；格式演进靠字段松耦合（Claude Code 无版本号）。

### 4.2 跨家结论

1. **存储形态由查询需求决定，不由数据重要性决定。** 转录最重要却几乎都在文件里；投递记录最不重要却一定入库。
2. **库体积与索引数量反向相关。** OpenCode 1.3 GB/17 索引，Hermes 199 MB/16 索引——索引少不是懒，是查询形状单一。
3. **压缩策略是存储设计的上游。** 先定「压缩后能否回看/检索/重放」，再定存哪儿。
4. **索引都是事后补的。** WorkBuddy 的 `0006` 迁移、OpenCode 的 `session_message` 投影索引，都是有了真实数据才加的。反过来看，**迁移历史里的「补索引」记录就是性能问题的化石**。
5. **宽度 vs 分表。** Hermes 把 55 列堆在 `sessions`，OpenCode 拆成 `session` + `session_context_epoch` + `session_input`。前者查询一次拿全，后者每次要 join。**在 SQLite（无列存、无覆盖索引优化器）上，宽表往往更快，但改列成本高**。

---

## 5. 后续篇的取证清单

写下一篇前必须重跑的命令（数字会变，结论不会）：

```bash
# 行数 / 表数 / 索引数
sqlite3 -readonly ~/.local/share/opencode/opencode.db \
  "select 'event',count(*) from event union all select 'part',count(*) from part;"
sqlite3 -readonly ~/.local/share/opencode/opencode.db \
  "select count(*) from sqlite_master where type='table' and name not like 'sqlite_%';"

# 查询计划（写入正文的每条「走索引」论断都要有它）
sqlite3 -readonly ~/.hermes/state.db \
  "EXPLAIN QUERY PLAN SELECT id FROM messages WHERE active IS NULL;"

# 闭源三家的 schema
sqlite3 ~/.workbuddy/workbuddy.db ".schema"
sqlite3 -readonly ~/.codex/goals_1.sqlite ".schema"
```

**已核正、可直接采信的事实**（本节是踩过坑才得到的，别再犯）：

- OpenCode 远端是 **`anomalyco/opencode`**，不是 `sst/opencode`；锁 `b7e4f1ef`。
- OpenCode 索引 17 个、表 21 张（不是流传的 18/24）。
- WorkBuddy 的 JSONL 直方图**一写完就过期**（1987→8315），引用必须带时点。
- Hermes 里 `active=0` 有**两种**含义：`compacted=1`（压缩）与 `compacted=0`（回退）。只写「active=0 表示删除」是错的。
- `messages_fts` 与 `messages_fts_trigram` **不是冗余**：前者给词，后者给子串与中文。
- Codex 的 55 个 rollout 是 `sessions/<年>/...`，不是平铺。

---

## 6. 禁止事项（写下一篇时）

- 不写反编译内容；闭源三家只引落盘产物。
- 不贴任何会话正文、工具输出、路径原文（Pi 的 `compaction.summary` 含真实项目路径，只写字段结构）。
- 不写绝对路径（一律 `~/...`）。
- 不把 INFERRED 当已验证（WorkBuddy 的 DML 三条就是 INFERRED）。
- 统计数字必须带「单次快照」限定，且同段内所有数字来自同一次快照。
