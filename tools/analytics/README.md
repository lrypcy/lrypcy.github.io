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
│       ├── index.js       路由：/c 采集、/total 计数、/stats 公开看板、/dash 口令看板
│       ├── lib.js         纯逻辑（时间换算、访客哈希、热力图网格、共用色阶、SQL 常量）
│       ├── dashboard.js   看板页 HTML 渲染（含热力图、腾讯地图访问地图、内嵌模式）
│       ├── province-map.js 省级访问分布的聚合与落点（纯函数；边界交给腾讯地图底图）
│       └── names.js       国家 / 省份代码 → 中文名
├── d1-shim.mjs            D1 / Cache API 的本地替身（测试与预览共用）
├── dev-server.mjs         本地预览服务器（带演示数据）
├── check_liquid.rb        Liquid 语法校验（含站点其它源文件 + about.md 内嵌区块）
└── test/
    ├── lib.test.mjs       纯函数单测
    ├── aggregate.test.mjs 聚合口径校验（内存 SQLite）
    ├── heatmap.test.mjs   活动热力图网格构造（边界、色阶、月份标签）
    ├── province-map.test.mjs 省级落点：完备性、坐标合法性、港澳台归位、未定位流量不丢失
    └── worker.test.mjs    路由 / CORS / 去重 / 看板渲染 / 公开路由缓存
```

站点侧新增或改动：

| 文件 | 改动 |
|---|---|
| `_includes/analytics.html` | 新增。埋点上报 + 页脚计数，配置为空时不输出任何内容 |
| `_layouts/default.html` | 页脚加计数元素，`</body>` 前引入上面的 include |
| `about.md` | 新增「访问统计」区块：iframe 内嵌 `/stats?embed=1`（`referrerpolicy=strict-origin-when-cross-origin`，供地图 key 的域名校验）+ 高度回填脚本 |
| `_config.yml` | 新增 `analytics:` 配置段 |
| `assets/css/custom.css` | 新增 `.footer-visits` 与 `.about-stats-embed` 样式 |

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

### 3.3 设置密钥

```bash
openssl rand -hex 32 | npx wrangler secret put SALT_SECRET   # 访客哈希盐，随便一串随机字符
npx wrangler secret put DASH_TOKEN                            # 看板口令，自己定
npx wrangler secret put TMAP_KEY                              # 可选：腾讯地图 key，配了才显示访问地图
```

`SALT_SECRET` **不要换**：换了之后同一个人会被算成新访客，当天的 UV 会虚高一次。
`DASH_TOKEN` 是看板的唯一保护，别用短口令。

#### 访问地图的 key（可选；不配就不显示这张卡片）

地图底图由**腾讯地图 GL JS** 提供（合规白名单内的来源），需要**你自己的 key**。
WorkBuddy 的免 key 代理靠 `_TMapSecurityConfig` + `__WB_HTTP_PORT__` / `__WB_TMAP_SECRET__` 占位符，
只在它的预览环境里被替换，**部署到 Cloudflare Worker 后代理地址不存在**，所以线上必须自备 key。

不配 `TMAP_KEY`（Worker 变量为空或仍是源码里的占位符）时，「访问地图」卡片**整块不渲染** ——
页面不会留一个空白地图，也不会把开发说明泄露给访客。

1. 到 [腾讯位置服务控制台](https://lbs.qq.com/dev/console/key/manage) 新建 key，勾选 **JavaScript API GL**；
2. 在该 key 的「授权域名」里加上看板域名 —— 通常是 `<你的子域>.workers.dev`。
   站点侧是 iframe 嵌入，iframe 自己发出的 Referer 是看板的域名，**只白名单 workers.dev 就够，不用加 github.io**；
3. `npx wrangler secret put TMAP_KEY` 写入 key，再 `npx wrangler deploy`。

看板响应头带 `Referrer-Policy: strict-origin-when-cross-origin`（`about.md` 里 iframe 的 `referrerpolicy`
也是同一个值），保证 key 的域名校验能拿到 Referer。**改成 `no-referrer` 会让校验失败、底图空白。**

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

提交推送后，GitHub Pages 重新构建，页脚就会出现「· 总访问 N 次」，「关于」页也会出现内嵌看板。

两个地址：

| 地址 | 用途 |
|---|---|
| `/dash?token=<DASH_TOKEN>` | 完整看板，带口令，建议存成书签 |
| `/stats` | **公开只读看板**，不要口令，关于页内嵌用的就是它（加 `?embed=1` 走嵌入版布局） |

---

## 4. 关于页内嵌看板

`about.md` 里用 iframe 内嵌 `/stats?embed=1`，见该文件末尾的 `.about-stats-embed` 区块。

**内嵌的是完整看板**：总览卡片、活动热力图、**访问地图**、年度 / 月度 / 最近 30 天柱状图、
国家 / 地区排行、地区明细表 —— 与 `/dash` 渲染的是同一个 `renderDashboard()`，
差别只是嵌入版去掉大标题、把口径说明压成一行脚注、图表尺寸收一档。

几个刻意的决定：

- **内嵌走公开路由，不把 `DASH_TOKEN` 写进页面。** iframe 的 `src` 在页面上是明文，带口令等于公开口令 ——
  既然看板本来就是要给访客看的，不如直接把 `/stats` 做成公开视图，口令只用来保护 `/dash`。
- **高度靠 `postMessage` 回填**，不写死。看板高度会随「近 30 天柱数 / 地区明细行数」变化，写死要么留白要么出内部滚动条。
  子页在 `load` / `resize` / `ResizeObserver` 时把 `{type:'pcy-analytics:height', height}` 发给父页，父页只接受 600–6000px 的值。
- **`/stats` 有 5 分钟边缘缓存**（`caches.default`，按 `日期 + range + embed` 分开存）。
  关于页比文章页更容易被反复打开，不缓存的话每个访客都要打 8 次 D1 查询。
- **嵌入版不做横向滚动**：热力图格子比完整版小一档（10px，窄屏 9px）；访问地图在嵌入版矮一档（320px，窄屏 260px），免得在 iframe 里占掉整屏。
- 关于页整块由 `{% if site.analytics.enabled %}` 包着，关掉统计后不留空壳。

---

## 5. 本地预览（不部署也能看）

```bash
node tools/analytics/dev-server.mjs
# 看板      http://localhost:8788/dash?token=dev
# 公开看板  http://localhost:8788/stats
# 内嵌预览  http://localhost:8788/stats?embed=1
# 总计数    http://localhost:8788/total
# 采集      POST http://localhost:8788/c
```

默认灌 420 天确定性演示数据，便于确认「活动热力图 / 年度 / 月度 / 最近 30 天 / 地区分布」几张图都符合预期。

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

## 6. 验证

```bash
# 全部单测与口径校验（68 项）
node --test --no-warnings tools/analytics/test/*.test.mjs

# Liquid 语法与多配置渲染（改了 include / about.md 之后必跑，写错会挂掉整站构建）
ruby tools/analytics/check_liquid.rb
```

`aggregate.test.mjs` 用真实的 `schema.sql` 和 Worker 里同一份 SQL 常量，在内存 SQLite 上先手工算出期望值（总 17 / 本年 17 / 本月 11 / 今日 4 / 地区排行 上海 6 · 北京 5 · 广东 4 · 美国 2），再由 SQL 反算校对，确保「同一地区同一天重复访问是累加而不是新增行」「跨月范围过滤不串月」这些容易写错的地方是对的。

`heatmap.test.mjs` 只盯热力图的边界：最后一列必须含 `endDay` 且之后的格子是 `null`（「还没到」不能画成「零访问」）、窗口起点不会被补格子、色阶在「只有一个非零日」这类退化输入下不能全变最浅、月份标签必须落在真实的月初列上、跨年时用年份把同名月份区分开。

`province-map.test.mjs` 盯几条会静默出错的规则：落点表必须**恰好**覆盖 34 个省级行政区、与 `names.js` 的 `CN_PROVINCES` 同集合，且坐标都是中国范围内的合法经纬度、互不重合；**港澳台整块归位**（Cloudflare 给它们的 `country` 是 `HK`/`MO`/`TW`、不带 `province`，漏了就会凭空丢掉港澳台流量）；**定位不到的流量必须被汇总报出来**（英文省名如 `Guangdong`、空省名、`XX` 未定位都不能无声吞掉，海外流量则明确不进这张图、由排行卡片负责）；**只有有访问的省份才落点**，避免地图上飘满空心圆。

`check_liquid.rb` 现在除了渲染 `_includes/analytics.html` 的五套配置，还会**语法检查站点所有含 Liquid 的源文件**（`*.md` / `*.html` / `_layouts` / `_includes`），并单独渲染 `about.md` 的访问统计区块，确认开关生效、endpoint 被替换、没有口令泄漏。

---

## 7. 口径与隐私

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

### 活动热力图
53 列 × 7 行，一列一周、一格一天，一周从**周日**起算（与 GitHub 一致）。

- **`endDay` 之后的格子留空**（透明），不是填 0 —— 「还没到」和「到了但没人来」必须能分开看，否则月初看图会误以为流量塌了；
- 色阶按「占窗口内单日最高 PV 的比例」分 4 档（≤25% / ≤50% / ≤75% / 其余），不用分位数是因为低流量站早期只有一两个非零日，分位数会退化成全部同档；
- **只有一天有数据时直接给最高档**，避免 `max` 退化成 1 时唯一的访问日只剩最浅色；
- 月份标签打在该月 1 号所在的那一列；**年份只标在 1 月**，用来区分跨年的同名月份 ——
  不要加到首列上，首列离第二个标签只有 4 列间距，「2025年10月」会正好压住「11月」。

### 访问地图（腾讯地图 GL JS）

底图由腾讯地图 GL JS 渲染，本仓库只在上面打点：每个有访问的省级行政区一个气泡，
**气泡大小与颜色按访问量分 4 档**（与热力图共用 `lib.js` 的 `levelByRatio()`），旁边一个 `省份 次数` 标签。

**为什么是腾讯地图、为什么必须自备 key —— 这一段别删，改这块之前先读：**

1. **合规**：地图渲染只能用腾讯 / 高德 / 百度 / 天地图这类合规来源（Google / Apple / OSM / Mapbox 一律不行）。
   疆界、南海诸岛与九段线由**底图按国家标准绘制** —— 这正是「不自己画边界」的理由，自绘边界才是合规风险源。
2. **key**：这四家都要 key。WorkBuddy 的免 key 代理只在它的预览环境生效（见 3.3），线上拿不到；
   所以 key 走 Worker 变量 `TMAP_KEY`，**不配就不渲染这张卡片**。
3. **不上边界数据**：`province-map.js` 里只有 34 个省级行政区的中心点（GCJ-02，来自
   阿里云 DataV.GeoAtlas 的 `center` 字段），没有几百 KB 的边界多边形 —— 省界交给底图。

落点是**省级行政区中心**，不是几何重心精算：气泡只表达「哪个省、量级多大」，不追求毫米级位置。

- 落点依据是 Cloudflare 的 `regionCode`。它并不总是 ISO 短码：`normalizeGeo()` 取的是
  `regionCode || region`，只给了英文省名（`Guangdong`）时大写后匹配不上任何省 ——
  这类流量与空省名一起汇总成「未定位到省份」，**不会无声消失**，汇总行里会写出来。
- **海外与 XX 不进这张图**，它们由「国家 / 地区排行」负责。两个卡片的适用范围是刻意分开的。
- **只有有访问的省份才打点**，避免地图上飘满空心圆（`buildProvinceStats()` 只返回 `v > 0` 的省）。
- 港澳台与大陆同图（Cloudflare 给它们的 `country` 是 HK / MO / TW，不带 `province`，整块归一个落点）。
- 这张卡片是整个看板**唯一的外部依赖**：浏览器直接向 `map.qq.com` 请求 SDK 与底图，并带上本站域名作为 Referer。

### 存了什么
只存两类数据：

- `daily_stats`：`日期 × 国家 × 省 × 城市 → 访问次数`。行数由「天数 × 地区组合数」决定，十万次访问也只占几千行；
- `visitor_days`：`日期 + 访客哈希`，仅用于当日去重。

**不存 IP、不存 User-Agent、不写 cookie、不写 localStorage**，所以不需要 cookie 同意横幅。原始 IP 只在 Worker 内存里参与一次哈希运算。

---

## 8. 成本与配额

都在 Cloudflare 免费额度之内，个人博客用不到 1%：

| 资源 | 免费额度 | 本方案用量 |
|---|---|---|
| Workers 请求 | 10 万 / 天 | 每个页面浏览 2 次（采集 + 页脚计数） |
| D1 行写入 | 10 万 / 天 | ≈ 每次访问 2 行（聚合行 + 访客行） |
| D1 行读取 | 500 万 / 天 | `/total` 与 `/stats` 都有 5 分钟边缘缓存，命中不读库；看板一次 8 条查询、扫表几千行 |
| D1 存储 | 5 GB | 一天几百次访问，一年几百 KB 量级 |

免费的 Workers 只有 10ms CPU / 请求，本项目单次请求只做一次 SHA-256 加两条 SQL，余量充足。

---

## 9. 已知限制

1. **`*.workers.dev` 在中国大陆的可达性不稳定。** 统计量偏低时先查这一点：浏览器开发者工具看 `/c` 请求是否成功。如果长期不可靠，可以绑一个自有域名（Cloudflare 免费支持自定义域 + 自动 SSL），代价是要有一个域名。
2. **禁用 JavaScript 的访客统计不到。** 上报依赖 `sendBeacon` / `fetch`。这是纯静态站的固有限制，用图片像素可以绕，但会牺牲去重和 Referer 校验。
3. **走代理 / VPN 的访客地区会落到代理节点。**
4. **不做文章级统计**（按需要可以后加：在 `daily_stats` 里加一列 `path`，主键扩成五列即可，埋点侧把 `location.pathname` 放进请求体）。
5. **`/total` 与 `/stats` 都是公开只读端点**，任何人都能拿到总访问数字和地区分布 —— 页脚与关于页都要公开展示，这一点无法避免。`/total` 只返回 4 个整数；`/stats` 返回的是看板 HTML，不含原始 IP / UA（这些本来就没存）。两者都带 `x-robots-tag: noindex`，不会被搜索引擎单独收录。
6. **访问地图需要自备腾讯地图 key**（`TMAP_KEY`），不配就不显示这张卡片。地图底图是**外部依赖**，
   完全离线 / 内网打不开；这是全看板唯一会往外请求资源的地方。
7. **地图粒度只到省级 / 国家级。** 图上只有省级行政区的落点，市一级请看「地区明细表」；
   海外流量不进这张图（只到国家），因为它是为大陆省份 + 港澳台设计的。
8. **落点是省级行政区中心点**，不是几何重心精算；气泡只表达「哪个省、量级多大」，不追求毫米级准确。
9. **热力图色阶是「相对当日最高值」的**，不是绝对值。所以只有几十次访问的月份里，一次访问也可能显示成中等深浅；要看具体数字请悬停格子或看下面的地区明细。

---

## 10. 日常运维

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
