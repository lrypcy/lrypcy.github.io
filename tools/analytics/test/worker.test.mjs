/**
 * Worker 处理器测试：用内存 SQLite 冒充 D1、用 Map 冒充 Cache API，
 * 直接在 node 里跑 fetch，覆盖路由、CORS、去重、爬虫过滤与看板渲染。
 *
 * 运行：node --test tools/analytics/test/*.test.mjs
 */

import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import worker from '../worker/src/index.js';
import { createD1, createSqlite, installCacheMock } from '../d1-shim.mjs';

const ORIGIN = 'https://lrypcy.github.io';
const ENDPOINT = 'https://pcy-analytics.example.workers.dev';
const CHROME_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36';

function fakeRequest({ url, method = 'POST', headers = {}, cf = {} }) {
  return { url, method, headers: new Headers(headers), cf };
}

function setup() {
  installCacheMock();
  const sqlite = createSqlite();
  const pending = [];
  const env = {
    DB: createD1(sqlite),
    SALT_SECRET: 'unit-test-secret',
    DASH_TOKEN: 'unit-test-token',
    ALLOWED_ORIGIN: ORIGIN,
    SITE_TZ_OFFSET: '8',
    DEDUPE_SECONDS: '30',
    STRICT_REFERER: 'true',
    TMAP_KEY: 'unit-test-tmap-key',
  };
  const ctx = { waitUntil: (p) => pending.push(p) };
  const collect = (opts = {}) =>
    worker.fetch(
      fakeRequest({
        url: `${ENDPOINT}/c`,
        method: 'POST',
        headers: {
          'CF-Connecting-IP': '203.0.113.9',
          'User-Agent': CHROME_UA,
          Referer: `${ORIGIN}/2026/10/01/some-post/`,
          Origin: ORIGIN,
          ...(opts.headers || {}),
        },
        cf: { country: 'CN', regionCode: 'GD', city: 'Shenzhen', ...(opts.cf || {}) },
      }),
      env,
      ctx
    );
  return { sqlite, env, ctx, pending, collect };
}

let h;

beforeEach(() => {
  h = setup();
});

const pvTotal = (sqlite) => sqlite.prepare('SELECT COALESCE(SUM(pv), 0) AS t FROM daily_stats').get().t;

/* ---------- 采集端点 ---------- */

test('OPTIONS 预检返回 204 且带 CORS 头', async () => {
  const res = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/c`, method: 'OPTIONS', headers: { Origin: ORIGIN } }),
    h.env,
    h.ctx
  );
  assert.equal(res.status, 204);
  assert.equal(res.headers.get('access-control-allow-origin'), ORIGIN);
  assert.match(res.headers.get('access-control-allow-methods'), /POST/);
});

test('GET /c 返回 405，不写库', async () => {
  const res = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/c`, method: 'GET' }), h.env, h.ctx);
  assert.equal(res.status, 405);
  assert.equal(pvTotal(h.sqlite), 0);
});

test('正常访问记 1 次，并落到正确的地理格子里', async () => {
  const res = await h.collect();
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { ok: true });

  const row = h.sqlite.prepare('SELECT * FROM daily_stats').get();
  assert.equal(row.country, 'CN');
  assert.equal(row.province, 'GD');
  assert.equal(row.city, 'Shenzhen');
  assert.equal(row.pv, 1);
  assert.equal(h.sqlite.prepare('SELECT COUNT(*) AS c FROM visitor_days').get().c, 1);
});

test('同一访客 30 秒内重复上报被去重，PV 仍为 1', async () => {
  await h.collect();
  const second = await h.collect();
  assert.deepEqual(await second.json(), { ok: true, skipped: 'dedupe' });
  assert.equal(pvTotal(h.sqlite), 1);
});

test('不同 IP 视为不同访客，各自计数', async () => {
  await h.collect();
  await h.collect({ headers: { 'CF-Connecting-IP': '198.51.100.4' } });
  assert.equal(pvTotal(h.sqlite), 2);
  assert.equal(h.sqlite.prepare('SELECT COUNT(*) AS c FROM visitor_days').get().c, 2);
});

test('爬虫与空 UA 直接丢弃', async () => {
  const bot = await h.collect({ headers: { 'User-Agent': 'Googlebot/2.1 (+http://www.google.com/bot.html)' } });
  assert.deepEqual(await bot.json(), { ok: true, skipped: 'bot' });
  const none = await h.collect({ headers: { 'User-Agent': '' } });
  assert.deepEqual(await none.json(), { ok: true, skipped: 'bot' });
  assert.equal(pvTotal(h.sqlite), 0);
});

test('带外站 Referer 的请求被拒，空 Referer 仍统计', async () => {
  const bad = await h.collect({ headers: { Referer: 'https://spam.example.net/x' } });
  assert.deepEqual(await bad.json(), { ok: true, skipped: 'referer' });
  assert.equal(pvTotal(h.sqlite), 0);

  const empty = await h.collect({ headers: { Referer: '' } });
  assert.deepEqual(await empty.json(), { ok: true });
  assert.equal(pvTotal(h.sqlite), 1);
});

test('无法定位的 IP 落到 XX，不会因为缺字段报错', async () => {
  const res = await h.collect({ cf: { country: undefined, regionCode: undefined, city: undefined } });
  assert.equal(res.status, 200);
  assert.equal(h.sqlite.prepare('SELECT country FROM daily_stats').get().country, 'XX');
});

test('CORS：Origin 不在白名单时不回 allow-origin 头', async () => {
  const res = await h.collect({ headers: { Origin: 'https://evil.test' } });
  assert.equal(res.headers.get('access-control-allow-origin'), null);
  // 仍然照常记账：CORS 只保护浏览器读取响应，不是防刷手段
  assert.equal(pvTotal(h.sqlite), 1);
});

test('存储失败时返回 500 而不是把异常抛给访客', async () => {
  const brokenEnv = { ...h.env, DB: { prepare: () => ({ bind: () => ({ run: () => {}, first: () => {} }) }), batch: async () => { throw new Error('D1 unavailable'); } } };
  const res = await worker.fetch(
    fakeRequest({
      url: `${ENDPOINT}/c`,
      method: 'POST',
      headers: { 'CF-Connecting-IP': '203.0.113.9', 'User-Agent': CHROME_UA, Referer: `${ORIGIN}/` },
      cf: { country: 'CN' },
    }),
    brokenEnv,
    h.ctx
  );
  assert.equal(res.status, 500);
  assert.equal((await res.json()).ok, false);
});

/* ---------- 公开计数端点 ---------- */

test('GET /total 返回总数，并用边缘缓存兜住重复读', async () => {
  await h.collect();
  await h.collect({ headers: { 'CF-Connecting-IP': '198.51.100.7' } });

  const first = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/total`, method: 'GET' }), h.env, h.ctx);
  assert.equal(first.status, 200);
  assert.equal(first.headers.get('x-pcy-cache'), 'miss');
  const body = await first.json();
  assert.equal(body.total, 2);
  assert.equal(body.today, 2);
  assert.equal(body.month, 2);
  assert.equal(body.year, 2);

  await Promise.all(h.pending); // 让 waitUntil 里的 cache.put 落地

  const second = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/total`, method: 'GET', headers: { Origin: ORIGIN } }),
    h.env,
    h.ctx
  );
  assert.equal(second.headers.get('x-pcy-cache'), 'hit');
  assert.equal((await second.json()).total, 2);
  assert.equal(second.headers.get('access-control-allow-origin'), ORIGIN, '缓存命中也要带当前调用方的 CORS 头');
});

test('GET /total 空库返回 0 而不是 null', async () => {
  const res = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/total`, method: 'GET' }), h.env, h.ctx);
  assert.deepEqual(await res.json(), { total: 0, today: 0, month: 0, year: 0 });
});

/* ---------- 看板 ---------- */

test('看板：缺口令 401，错口令 401，对口令 200', async () => {
  const noToken = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/dash`, method: 'GET' }), h.env, h.ctx);
  assert.equal(noToken.status, 401);

  const wrong = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/dash?token=nope`, method: 'GET' }),
    h.env,
    h.ctx
  );
  assert.equal(wrong.status, 401);

  const ok = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/dash?token=unit-test-token`, method: 'GET' }),
    h.env,
    h.ctx
  );
  assert.equal(ok.status, 200);
  assert.match(ok.headers.get('content-type'), /text\/html/);
});

test('看板渲染出总览、年月曲线与中文地区名', async () => {
  await h.collect(); // 中国 · 广东 · Shenzhen
  await h.collect({
    headers: { 'CF-Connecting-IP': '198.51.100.7' },
    // 真实场景下 Cloudflare 对同一请求只会给 regionCode 或 region 之一，这里显式清掉 regionCode
    cf: { country: 'US', regionCode: '', region: 'California', city: 'San Francisco' },
  });
  await h.collect({ headers: { 'CF-Connecting-IP': '198.51.100.8' }, cf: { country: 'CN', regionCode: 'BJ', city: 'Beijing' } });

  const res = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/dash?token=unit-test-token&range=all`, method: 'GET' }),
    h.env,
    h.ctx
  );
  const html = await res.text();

  assert.match(html, /访问统计/);
  assert.match(html, /总访问次数/);
  assert.match(html, /年度访问次数/);
  assert.match(html, /月度访问次数/);
  assert.match(html, /国家 \/ 地区排行/);
  assert.ok(html.includes('中国 · 广东 · Shenzhen'), '地区明细里应出现中文地区名');
  assert.ok(html.includes('美国 · California · San Francisco'));
  assert.ok(html.includes('中国 · 北京'));
  assert.match(html, /noindex/, '看板不应被搜索引擎收录');

  // 访问地图（腾讯地图 GL JS）：广东 / 北京落点，美国那一行不该进图
  assert.match(html, /id="pcy-map-canvas"/);
  assert.match(html, /map\.qq\.com\/api\/gljs\?v=1\.exp&key=/);
  assert.ok(html.includes('"unit-test-tmap-key"'), 'TMAP_KEY 应注入到地图脚本');
  assert.match(html, /"n":"广东","v":\d+/);
  assert.match(html, /"n":"北京","v":\d+/);
  assert.ok(!/"n":"美国"/.test(html), '海外国家不进省级地图');
  assert.match(html, /GCJ-02/, '口径要讲明落点坐标系');
  assert.match(html, /class="mp-legend"/, '地图要带图例，否则颜色深浅没有参照');
  assert.ok(
    ['#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8'].every((c) => html.includes(c)),
    '图例色块必须与气泡的 4 个档位同色'
  );
  assert.equal(
    res.headers.get('referrer-policy'),
    'strict-origin-when-cross-origin',
    '地图 key 的域名校验需要 Referer，看板必须放行来源'
  );
});

test('未配置 TMAP_KEY 时，访问地图卡片整块不渲染，其余卡片不受影响', async () => {
  await h.collect();
  const env = { ...h.env, TMAP_KEY: '' };
  const res = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/dash?token=unit-test-token`, method: 'GET' }),
    env,
    h.ctx
  );
  const html = await res.text();
  assert.ok(!html.includes('pcy-map-canvas'), '没配 key 不应渲染地图容器');
  assert.ok(!html.includes('class="mp-legend"'), '没配 key 也不该留下光秃秃的图例');
  assert.ok(!html.includes('map.qq.com/api/gljs'), '没配 key 不应请求腾讯地图 SDK');
  assert.match(html, /国家 \/ 地区排行/, '其它卡片不受影响');
});

test('看板 range 参数非法时回落到全部时间，不报错', async () => {
  await h.collect();
  const res = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/dash?token=unit-test-token&range=quarter`, method: 'GET' }),
    h.env,
    h.ctx
  );
  assert.equal(res.status, 200);
  assert.match(await res.text(), /全部时间/);
});

test('range=month 的地理合计与月度总数一致', async () => {
  await h.collect();
  await h.collect({ headers: { 'CF-Connecting-IP': '198.51.100.7' } });
  const html = await (
    await worker.fetch(
      fakeRequest({ url: `${ENDPOINT}/dash?token=unit-test-token&range=month`, method: 'GET' }),
      h.env,
      h.ctx
    )
  ).text();
  assert.match(html, /范围「本月」，共 2 次/);
});

/* ---------- 活动热力图 ---------- */

test('看板渲染出活动热力图，且「还没到的日子」与「零访问」是不同 class', async () => {
  await h.collect(); // 今天 · 中国 · 广东 · Shenzhen

  const html = await (
    await worker.fetch(
      fakeRequest({ url: `${ENDPOINT}/dash?token=unit-test-token`, method: 'GET' }),
      h.env,
      h.ctx
    )
  ).text();

  assert.match(html, /活动热力图/);
  assert.match(html, /class="hm-grid"/);
  assert.match(html, /class="hm-month"/);

  // 53 周 × 7 天 = 371 个格子
  const cells = html.match(/class="hm-cell[^"]*"/g) || [];
  assert.equal(cells.length, 371 + 5, '热力图格子 371 个 + 图例 5 个');
  assert.ok(html.includes('hm-void'), '末尾未到的格子必须是空格子而不是 l0');
  assert.match(html, /class="hm-cell l4" title="\d{4}-\d{2}-\d{2} · 1 次访问"/, '唯一一次访问应是最高档');
  assert.match(html, /近 53 周有 1 天有访问，单日最高 1 次/);
});

/* ---------- 公开看板（关于页内嵌） ---------- */

test('GET /stats 不需要口令就是 200，且不会把口令带进页面里的链接', async () => {
  await h.collect();
  const res = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats`, method: 'GET' }), h.env, h.ctx);

  assert.equal(res.status, 200, '公开路由不能要口令');
  assert.equal(res.headers.get('x-pcy-cache'), 'miss');
  assert.equal(res.headers.get('x-robots-tag'), 'noindex');

  const html = await res.text();
  assert.match(html, /活动热力图/);
  assert.ok(!html.includes('token='), '公开页面的链接不能带 token，否则等于把口令公开');
});

test('GET /stats?embed=1 走嵌入版：有高度上报脚本、没有完整口径说明', async () => {
  await h.collect();
  const html = await (
    await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats?embed=1`, method: 'GET' }), h.env, h.ctx)
  ).text();

  assert.match(html, /<body class="embed">/);
  assert.match(html, /pcy-analytics:height/, '需要把高度 postMessage 给父页，否则 iframe 高度写死');
  assert.ok(!html.includes('<h2>口径说明</h2>'), '嵌入版不该塞整段口径说明');
  assert.match(html, /class="footnote"/);
});

test('GET /stats 只缓存取数结果：命中时不再打 D1，HTML 仍每次现渲染', async () => {
  await h.collect();
  const first = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats`, method: 'GET' }), h.env, h.ctx);
  assert.equal(first.headers.get('x-pcy-cache'), 'miss');
  const firstHtml = await first.text();

  await Promise.all(h.pending); // 让 waitUntil 里的 cache.put 落地

  // 换一个 D1 替身：数据缓存真命中就不会再碰数据库，内容仍应与首次一致。
  const envWithoutDb = { ...h.env, DB: { prepare: () => ({ bind: () => ({ all: async () => { throw new Error('缓存命中却打了 D1'); } }) }), batch: async () => { throw new Error('缓存命中却打了 D1'); } } };
  const second = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats`, method: 'GET' }), envWithoutDb, h.ctx);

  assert.equal(second.headers.get('x-pcy-cache'), 'hit');
  assert.equal(await second.text(), firstHtml, '同一份数据应当渲染出同一份 HTML');
});

/*
 * 回归用例：这里曾经缓存「渲染好的 HTML」，缓存键又不含版本标识，
 * 于是把自绘方块图换成腾讯地图之后，线上一直吐旧方块图。改成缓存数据后，
 * HTML 必须 no-store —— 谁都不许把它留住。
 */
test('GET /stats 的 HTML 必须 no-store，且带 Referrer 策略', async () => {
  await h.collect();
  const res = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats?embed=1`, method: 'GET' }), h.env, h.ctx);
  assert.equal(res.headers.get('cache-control'), 'no-store');
  assert.equal(res.headers.get('referrer-policy'), 'strict-origin-when-cross-origin');
});

test('GET /stats 的数据缓存只按 day / range 分：embed 与完整版共用一份', async () => {
  await h.collect();
  const embed = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats?embed=1`, method: 'GET' }), h.env, h.ctx);
  assert.equal(embed.headers.get('x-pcy-cache'), 'miss');
  await Promise.all(h.pending);

  // 同一 day + range：完整版直接吃上刚才 embed 那次留下的数据缓存，但走的模板不同。
  const full = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/stats`, method: 'GET' }), h.env, h.ctx);
  assert.equal(full.headers.get('x-pcy-cache'), 'hit', 'embed 与完整版共用同一份数据缓存');
  const fullHtml = await full.text();
  assert.match(fullHtml, /<h2>口径说明<\/h2>/, '共用数据也必须渲染成完整版');
  assert.ok(!fullHtml.includes('<body class="embed">'), '完整版不能渲染成嵌入版');

  const month = await worker.fetch(
    fakeRequest({ url: `${ENDPOINT}/stats?range=month&embed=1`, method: 'GET' }),
    h.env,
    h.ctx
  );
  assert.equal(month.headers.get('x-pcy-cache'), 'miss', '换 range 要另取一份数据');
});

/* ---------- 其他路由 ---------- */

test('健康检查与 404', async () => {
  const health = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/health`, method: 'GET' }), h.env, h.ctx);
  assert.equal(health.status, 200);
  assert.equal((await health.json()).ok, true);

  const missing = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/nope`, method: 'GET' }), h.env, h.ctx);
  assert.equal(missing.status, 404);
});

test('路径里的重复斜杠被折叠：endpoint 配成带尾斜杠也不会 404', async () => {
  await h.collect();
  const res = await worker.fetch(fakeRequest({ url: `${ENDPOINT}//stats`, method: 'GET' }), h.env, h.ctx);
  assert.equal(res.status, 200, '//stats 应等同于 /stats');
  assert.match(await res.text(), /活动热力图/);

  const total = await worker.fetch(fakeRequest({ url: `${ENDPOINT}//total`, method: 'GET' }), h.env, h.ctx);
  assert.equal(total.status, 200);
});
