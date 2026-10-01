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
  assert.match(html, /访问来源地区/);
  assert.ok(html.includes('中国 · 广东 · Shenzhen'), '地区明细里应出现中文地区名');
  assert.ok(html.includes('美国 · California · San Francisco'));
  assert.ok(html.includes('中国 · 北京'));
  assert.match(html, /noindex/, '看板不应被搜索引擎收录');
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

/* ---------- 其他路由 ---------- */

test('健康检查与 404', async () => {
  const health = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/health`, method: 'GET' }), h.env, h.ctx);
  assert.equal(health.status, 200);
  assert.equal((await health.json()).ok, true);

  const missing = await worker.fetch(fakeRequest({ url: `${ENDPOINT}/nope`, method: 'GET' }), h.env, h.ctx);
  assert.equal(missing.status, 404);
});
