# 博客访问统计（自建 Cloudflare Worker + D1）

给 `lrypcy.github.io` 加访问统计：**IP 地区分布 + 年度 / 月度 / 总访问次数**。

GitHub Pages 是纯静态托管，页面本身没有服务端，拿不到访问者 IP。所以唯一的做法是让访客浏览器在页面加载后，额外打一个请求到某个能读到 IP 的后端。这里选择自己搭后端而不是用现成服务，原因写在下面「为什么不用第三方」一节。

---

## 1. 为什么不用第三方

| 方案 | 问题 |
|---|---|
| Google Analytics 4 | 中国大陆基本加载失败，统计不到主要读者 |
| Umami Cloud 免费版 | **只保留 6 个月数据**，看不到完整年度和总览；脚本域名被 EasyPrivacy 收录，uBlock / Brave 用户全部不可见 |
| GoatCounter | 只有国家级分布，没有省 / 城市；年度数据要自己导出聚合 |
| 自建 Worker + D1 | 数据永久保留、能自由聚合、免费额度足够、不依赖任何会变卦的第三方 |

自建的核心优势是**「月度 / 年度 / 总访问」这三个数字的口径完全由自己定义**，不会被服务商的保留期或套餐改动影响。

---

## 2. 目录结构

```
tools/analytics/
├── README.md              本文档
├── worker/                部署到 Cloudflare 的部分
│   ├── wrangler.toml      部署配置（唯一需要手动改的文件）
│   ├── package.json
│   ├── schema.sql         D1 表结构
│   └── src/
│       ├── index.js       路由：/c 采集、/total 计数、/dash 看板
│       ├── lib.js         纯逻辑（时间换算、访客哈希、SQL 常量）
│       ├── dashboard.js   看板页 HTML 渲染
│       └── names.js       国家 / 省份代码 → 中文名
├── d1-shim.mjs            D1 / Cache API 的本地替身（测试与预览共用）
├── dev-server.mjs         本地预览服务器（带演示数据）
├── check_liquid.rb        _includes/analytics.html 的 Liquid 语法校验
└── test/
    ├── lib.test.mjs       纯函数单测
    ├── aggregate.test.mjs 聚合口径校验（内存 SQLite）
    └── worker.test.mjs    路由 / CORS / 去重 / 看板渲染
```

站点侧新增或改动：

| 文件 | 改动 |
|---|---|
| `_includes/analytics.html` | 新增。埋点上报 + 页脚计数，配置为空时不输出任何内容 |
| `_layouts/default.html` | 页脚加计数元素，`</body>` 前引入上面的 include |
| `_config.yml` | 新增 `analytics:` 配置段 |
| `assets/css/custom.css` | 新增 `.footer-visits` 样式 |

`tools/` 已经在 `_config.yml` 的 `exclude` 里，不会被发布到站点。

---

## 3. 部署（约 5 分钟）

需要 Node 18+（本机已有）和一次性注册的免费 Cloudflare 账号。

### 3.1 创建 D1 数据库

```bash
cd tools/analytics/worker
npx wrangler login          # 浏览器授权一次
npx wrangler d1 create pcy-analytics
```

命令会输出一段 TOML，把其中的 `database_id` 填进 `wrangler.toml` 里的 `REPLACE_WITH_D1_DATABASE_ID`。

### 3.2 建表

```bash
npx wrangler d1 execute pcy-analytics --remote --file=./schema.sql
```

### 3.3 设置两个密钥

```bash
openssl rand -hex 32 | npx wrangler secret put SALT_SECRET   # 访客哈希盐，随便一串随机字符
npx wrangler secret put DASH_TOKEN                            # 看板口令，自己定
```

`SALT_SECRET` **不要换**：换了之后同一个人会被算成新访客，当天的 UV 会虚高一次。
`DASH_TOKEN` 是看板的唯一保护，别用短口令。

### 3.4 部署

```bash
npx wrangler deploy
```

输出形如 `https://pcy-analytics.<你的子域>.workers.dev`。先验证：

```bash
curl https://pcy-analytics.<你的子域>.workers.dev/health
# {"ok":true,"service":"pcy-analytics"}
```

### 3.5 接入站点

改 `_config.yml` 两处：

```yaml
analytics:
  enabled: true                                                  # false → true
  endpoint: "https://pcy-analytics.<你的子域>.workers.dev"        # 填上面那个地址
```

提交推送后，GitHub Pages 重新构建，页脚就会出现「· 总访问 N 次」。

看板地址：`https://pcy-analytics.<你的子域>.workers.dev/dash?token=<DASH_TOKEN>`，建议存成书签。

---

## 4. 本地预览（不部署也能看）

```bash
node tools/analytics/dev-server.mjs
# 看板      http://localhost:8788/dash?token=dev
# 总计数    http://localhost:8788/total
# 采集      POST http://localhost:8788/c
```

默认灌 420 天确定性演示数据，便于确认「年度 / 月度 / 最近 30 天 / 地区分布」四张图都符合预期。

常用参数：

```bash
--db=/tmp/a.db        # 数据落盘，可累积（默认内存库，退出即清空）
--no-seed             # 空库启动
--days=180            # 演示数据天数
--port=9000
--token=mysecret      # 改看板口令
--dedupe=30           # 同一访客去重窗口（秒）
--strict-referer      # 复现线上的 Referer 拦截行为
```

用 `X-Mock-IP` / `X-Mock-Country` / `X-Mock-Region-Code` / `X-Mock-City` 头，或 `?cf=US:California:SanFrancisco`，可以在本地模拟任意来源地区。

---

## 5. 验证

```bash
# 全部单测与口径校验（40 项）
node --test --no-warnings tools/analytics/test/*.test.mjs

# Liquid 语法与多配置渲染（改了 include 之后必跑，写错会挂掉整站构建）
ruby tools/analytics/check_liquid.rb
```

`aggregate.test.mjs` 用真实的 `schema.sql` 和 Worker 里同一份 SQL 常量，在内存 SQLite 上先手工算出期望值（总 17 / 本年 17 / 本月 11 / 今日 4 / 地区排行 上海 6 · 北京 5 · 广东 4 · 美国 2），再由 SQL 反算校对，确保「同一地区同一天重复访问是累加而不是新增行」「跨月范围过滤不串月」这些容易写错的地方是对的。

---

## 6. 口径与隐私

### 访问次数（PV）
每次成功的页面上报记 1 次。以下流量不计入：

- UA 命中爬虫 / 监控 / 预览特征（`Googlebot`、`bingbot`、`Baiduspider`、`Bytespider`、`curl`、`UptimeRobot` 等），空 UA 也丢弃；
- 带 Referer 但不属于本站的请求（`STRICT_REFERER=true` 时）；
- 同一访客在 `DEDUPE_SECONDS`（默认 30 秒）内的重复上报 —— 防连续刷新刷量。

### 独立访客（UV）
标识 = `SHA-256(盐 | 月份 | IP | UA)` 截断 24 位十六进制，**盐按月轮换**。

- 「今日 / 本月独立访客」准确；
- **不展示年度独立访客**：跨月之后同一个人的哈希变了，加起来会把同一个人重复计入。这是「不长期跟踪个人」的必然代价，不是 bug。

### 地区
来自 Cloudflare 对客户端 IP 的解析（`request.cf`）。国家准确度高；中国的省级行政区基本可用；**城市级在中国大陆质量一般、经常为空**，看到部分记录只到省市不必意外。`未知地区` 表示 Cloudflare 无法定位该 IP。

### 存了什么
只存两类数据：

- `daily_stats`：`日期 × 国家 × 省 × 城市 → 访问次数`。行数由「天数 × 地区组合数」决定，十万次访问也只占几千行；
- `visitor_days`：`日期 + 访客哈希`，仅用于当日去重。

**不存 IP、不存 User-Agent、不写 cookie、不写 localStorage**，所以不需要 cookie 同意横幅。原始 IP 只在 Worker 内存里参与一次哈希运算。

---

## 7. 成本与配额

都在 Cloudflare 免费额度之内，个人博客用不到 1%：

| 资源 | 免费额度 | 本方案用量 |
|---|---|---|
| Workers 请求 | 10 万 / 天 | 每个页面浏览 2 次（采集 + 页脚计数） |
| D1 行写入 | 10 万 / 天 | ≈ 每次访问 2 行（聚合行 + 访客行） |
| D1 行读取 | 500 万 / 天 | `/total` 有 5 分钟边缘缓存，命中不读库；看板一次 7 条查询、扫表几千行 |
| D1 存储 | 5 GB | 一天几百次访问，一年几百 KB 量级 |

免费的 Workers 只有 10ms CPU / 请求，本项目单次请求只做一次 SHA-256 加两条 SQL，余量充足。

---

## 8. 已知限制

1. **`*.workers.dev` 在中国大陆的可达性不稳定。** 统计量偏低时先查这一点：浏览器开发者工具看 `/c` 请求是否成功。如果长期不可靠，可以绑一个自有域名（Cloudflare 免费支持自定义域 + 自动 SSL），代价是要有一个域名。
2. **禁用 JavaScript 的访客统计不到。** 上报依赖 `sendBeacon` / `fetch`。这是纯静态站的固有限制，用图片像素可以绕，但会牺牲去重和 Referer 校验。
3. **走代理 / VPN 的访客地区会落到代理节点。**
4. **不做文章级统计**（按需要可以后加：在 `daily_stats` 里加一列 `path`，主键扩成五列即可，埋点侧把 `location.pathname` 放进请求体）。
5. **`/total` 是公开只读端点**，任何人都能拿到总访问数字 —— 页脚要公开展示，这一点无法避免。它只返回 4 个整数，不暴露地区明细；明细只有带 `DASH_TOKEN` 才能看。

---

## 9. 日常运维

```bash
cd tools/analytics/worker

# 看总数
npx wrangler d1 execute pcy-analytics --remote \
  --command "SELECT COALESCE(SUM(pv),0) FROM daily_stats"

# 导出全部聚合数据（备份用）
npx wrangler d1 execute pcy-analytics --remote \
  --command "SELECT * FROM daily_stats" --json > backup-daily.json

# 删掉某个测试日期的脏数据
npx wrangler d1 execute pcy-analytics --remote \
  --command "DELETE FROM daily_stats WHERE day = '2026-10-01'"
```

改完 `worker/src/` 下的代码，重新 `npx wrangler deploy` 即可；改完 `schema.sql` 要再跑一次 `d1 execute`（语句都是 `IF NOT EXISTS`，可重复执行）。

看板页底部有完整的口径说明，部署后不需要回头查文档。
