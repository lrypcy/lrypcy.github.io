/**
 * 本地预览 / 自测服务器：把 Worker 逻辑跑在 node:http 上。
 *
 * 用途：
 *   1. 不部署就能看到真实看板长什么样（含演示数据）；
 *   2. 用真实 HTTP 请求验证 CORS、Referer 校验、去重与记账口径。
 *
 * 用法：
 *   node tools/analytics/dev-server.mjs                  # 内存库 + 演示数据
 *   node tools/analytics/dev-server.mjs --db=/tmp/a.db   # 落盘，可累积
 *   node tools/analytics/dev-server.mjs --no-seed        # 空库
 *
 * 模拟 Cloudflare 注入的信息（真实环境里由 CF 提供）：
 *   X-Mock-IP / X-Mock-Country / X-Mock-Region-Code / X-Mock-Region / X-Mock-City
 *   也支持 ?cf=US:California:SanFrancisco
 */

import { createServer } from 'node:http';

import worker from './worker/src/index.js';
import { createD1, createSqlite, installCacheMock } from './d1-shim.mjs';

const TZ_OFFSET = 8;

function parseArgs(argv) {
  const args = {
    port: 8788,
    db: ':memory:',
    seed: true,
    days: 420,
    token: 'dev',
    dedupe: 10,
    strictReferer: false,
    // 访问地图的腾讯地图 key：命令行 --tmap-key=xxx 或环境变量 TMAP_KEY，留空则不渲染地图卡片
    tmapKey: process.env.TMAP_KEY || '',
  };
  for (const raw of argv) {
    const [key, value] = raw.replace(/^--/, '').split('=');
    if (key === 'port') args.port = Number(value);
    else if (key === 'db') args.db = value;
    else if (key === 'no-seed') args.seed = false;
    else if (key === 'days') args.days = Number(value);
    else if (key === 'dedupe') args.dedupe = Number(value);
    else if (key === 'strict-referer') args.strictReferer = true;
    else if (key === 'token') args.token = value || 'dev';
    else if (key === 'tmap-key') args.tmapKey = value || '';
  }
  return args;
}

const localDay = (tsMs) => new Date(tsMs + TZ_OFFSET * 3600 * 1000).toISOString().slice(0, 10);
const shiftDate = (day, delta) =>
  new Date(Date.parse(`${day}T00:00:00Z`) + delta * 86400000).toISOString().slice(0, 10);

/** [国家, 省级短码/省名, 城市, 相对权重] */
const REGIONS = [
  ['CN', 'BJ', 'Beijing', 96],
  ['CN', 'SH', 'Shanghai', 74],
  ['CN', 'GD', 'Shenzhen', 68],
  ['CN', 'ZJ', 'Hangzhou', 52],
  ['CN', 'JS', 'Nanjing', 40],
  ['CN', 'SC', 'Chengdu', 32],
  ['CN', 'HB', 'Wuhan', 28],
  ['CN', 'SD', 'Qingdao', 22],
  ['CN', 'SN', "Xi'an", 16],
  ['CN', 'HI', 'Haikou', 8],
  ['HK', '', 'Hong Kong', 26],
  ['TW', '', 'Taipei', 14],
  ['US', 'California', 'San Francisco', 34],
  ['US', 'Washington', 'Seattle', 16],
  ['JP', '13', 'Tokyo', 20],
  ['SG', '', 'Singapore', 18],
  ['DE', 'BE', 'Berlin', 10],
  ['GB', 'ENG', 'London', 12],
  ['CA', 'Ontario', 'Toronto', 8],
  ['AU', 'NSW', 'Sydney', 7],
  ['XX', '', '', 6],
];

/**
 * 灌入确定性的演示数据（固定种子，同样的天数跑出来完全一致）。
 * 曲线形状：缓慢增长 + 周末回落 + 噪声，便于验证「月度 / 年度 / 总计」的聚合口径。
 */
function seed(sqlite, days) {
  let state = 20261001;
  const rand = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };

  const weightSum = REGIONS.reduce((sum, r) => sum + r[3], 0);
  const today = localDay(Date.now());
  const upsert = sqlite.prepare(
    `INSERT INTO daily_stats (day, country, province, city, pv) VALUES (?, ?, ?, ?, ?)
     ON CONFLICT (day, country, province, city) DO UPDATE SET pv = pv + excluded.pv`
  );
  const visitor = sqlite.prepare('INSERT OR IGNORE INTO visitor_days (day, vid) VALUES (?, ?)');

  let totalPv = 0;
  sqlite.exec('BEGIN');
  for (let i = days - 1; i >= 0; i -= 1) {
    const day = shiftDate(today, -i);
    const dow = new Date(`${day}T00:00:00Z`).getUTCDay();
    const growth = 18 + (days - i) * 0.55;
    const weekend = dow === 0 || dow === 6 ? 0.62 : 1;
    const noise = 0.75 + rand() * 0.5;
    const dayTotal = Math.max(1, Math.round(growth * weekend * noise));

    for (const [country, province, city, weight] of REGIONS) {
      const share = (weight / weightSum) * (0.65 + rand() * 0.7);
      const pv = Math.round(dayTotal * share);
      if (pv <= 0) continue;
      upsert.run(day, country, province, city, pv);
      totalPv += pv;
    }

    const uv = Math.max(1, Math.round(dayTotal * (0.45 + rand() * 0.3)));
    for (let k = 0; k < uv; k += 1) visitor.run(day, `seed-${day}-${k}`);
  }
  sqlite.exec('COMMIT');
  return { days, totalPv };
}

/* ---------------- HTTP -> Worker 适配 ---------------- */

function parseCfQuery(value) {
  const [country = '', region = '', city = ''] = String(value).split(':');
  return { country, regionCode: '', region, city };
}

function toRequestLike(req, url) {
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (Array.isArray(value)) value.forEach((v) => headers.append(key, v));
    else if (value != null) headers.set(key, value);
  }

  // 真实环境由 Cloudflare 注入；本地用 X-Mock-* 头或 ?cf= 模拟
  headers.set('CF-Connecting-IP', headers.get('x-mock-ip') || req.socket.remoteAddress || '127.0.0.1');

  const fromQuery = url.searchParams.get('cf');
  const cf = fromQuery
    ? parseCfQuery(fromQuery)
    : {
        country: headers.get('x-mock-country') || 'CN',
        regionCode: headers.get('x-mock-region-code') || 'GD',
        region: headers.get('x-mock-region') || '',
        city: headers.get('x-mock-city') || 'Shenzhen',
      };

  return { url: url.toString(), method: req.method, headers, cf };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  installCacheMock();

  const sqlite = createSqlite(args.db);
  const before = sqlite.prepare('SELECT COUNT(*) AS c FROM daily_stats').get().c;
  let seeded = null;
  if (args.seed && before === 0) seeded = seed(sqlite, args.days);

  const env = {
    DB: createD1(sqlite),
    SALT_SECRET: 'local-dev-secret',
    DASH_TOKEN: args.token,
    ALLOWED_ORIGIN: `http://localhost:${args.port},https://lrypcy.github.io`,
    SITE_TZ_OFFSET: String(TZ_OFFSET),
    DEDUPE_SECONDS: String(args.dedupe),
    // 本地用 curl 自测时通常没有 Referer，默认放宽；加 --strict-referer 可复现线上拦截行为
    STRICT_REFERER: args.strictReferer ? 'true' : 'false',
    TMAP_KEY: args.tmapKey,
  };

  const server = createServer(async (req, res) => {
    const url = new URL(req.url, `http://localhost:${args.port}`);
    for await (const _chunk of req) { /* 读掉请求体，避免连接挂住 */ }

    const pending = [];
    const ctx = { waitUntil: (p) => pending.push(p) };

    try {
      const response = await worker.fetch(toRequestLike(req, url), env, ctx);
      await Promise.allSettled(pending);
      const headers = {};
      response.headers.forEach((value, key) => {
        headers[key] = value;
      });
      res.writeHead(response.status, headers);
      res.end(Buffer.from(await response.arrayBuffer()));
    } catch (error) {
      res.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' });
      res.end(`本地服务出错：${error && error.message}`);
    }
  });

  server.listen(args.port, '127.0.0.1', () => {
    const base = `http://localhost:${args.port}`;
    console.log(`pcy-analytics 本地预览已启动（${args.db === ':memory:' ? '内存库' : args.db}）`);
    if (seeded) console.log(`已灌入演示数据：${seeded.days} 天，合计 ${seeded.totalPv} 次访问`);
    console.log('');
    console.log(`  看板      ${base}/dash?token=${args.token}`);
    console.log(`  公开看板  ${base}/stats        （关于页内嵌用的那条路由，不要口令）`);
    console.log(`  内嵌预览  ${base}/stats?embed=1`);
    console.log(`  总计数    ${base}/total`);
    console.log(`  采集      POST ${base}/c   （可加 ?cf=CN:GD:Shenzhen 模拟地区）`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
