/**
 * pcy-analytics：lrypcy.github.io 的访问统计后端。
 *
 * 路由：
 *   POST /c            采集端点，前端用 sendBeacon 打这个地址
 *   GET  /total        公开只读，返回总/年/月/今日访问次数，给页脚计数用
 *   GET  /stats        公开只读看板，供「关于」页 iframe 内嵌；不校验口令，带 5 分钟边缘缓存
 *   GET  /dash?token=  看板，需要 DASH_TOKEN（口令只用于「自己单独打开」这条路径）
 *   GET  /             健康检查
 *
 * 隐私设计：不存 IP、UA、Cookie；访客标识是「盐 + 月份 + IP + UA」的 SHA-256 截断值，
 * 盐按月轮换，因此跨月无法关联个人。
 */

import {
  DAILY_SQL,
  DEFAULT_TZ_OFFSET_HOURS,
  HEATMAP_WEEKS,
  INSERT_VISITOR_SQL,
  MONTHLY_SQL,
  RANGE_UV_SQL,
  TODAY_UV_SQL,
  TOTALS_SQL,
  UPSERT_DAILY_SQL,
  YEARLY_SQL,
  addDays,
  buildCalendar,
  geoRankSql,
  isBot,
  localDay,
  makeVid,
  monthKeyOf,
  normalizeGeo,
  normalizeRange,
  parseOrigins,
  pickIp,
  rangeOf,
  refererAllowed,
  safeEqual,
  yearKeyOf,
} from './lib.js';
import { renderDashboard, renderUnauthorized } from './dashboard.js';

const JSON_CT = 'application/json; charset=utf-8';
const HTML_CT = 'text/html; charset=utf-8';
const STATS_CACHE_SECONDS = 300;

function corsHeaders(env, request) {
  const origin = request.headers.get('Origin') || '';
  const allowed = parseOrigins(env.ALLOWED_ORIGIN);
  const headers = {
    vary: 'Origin',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
    'access-control-allow-headers': 'content-type',
    'access-control-max-age': '86400',
  };
  if (origin && allowed.includes(origin)) headers['access-control-allow-origin'] = origin;
  return headers;
}

function jsonResponse(payload, status, extraHeaders) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': JSON_CT, 'cache-control': 'no-store', ...extraHeaders },
  });
}

function tzOffset(env) {
  const n = Number(env.SITE_TZ_OFFSET);
  return Number.isFinite(n) ? n : DEFAULT_TZ_OFFSET_HOURS;
}

/**
 * 看板 HTML 统一带上 `Referrer-Policy: strict-origin-when-cross-origin`：
 * 访问地图的底图由浏览器直接向腾讯地图请求，key 的「授权域名」校验看的是 Referer；
 * 不带 Referer 会被判为未授权（地图空白）。这个策略只发送站点来源、不发送完整路径。
 */
function htmlResponse(body, status = 200, extraHeaders = {}) {
  return new Response(body, {
    status,
    headers: {
      'content-type': HTML_CT,
      'cache-control': 'no-store',
      'referrer-policy': 'strict-origin-when-cross-origin',
      ...extraHeaders,
    },
  });
}

/** POST /c —— 记录一次页面访问。 */
async function handleCollect(request, env, ctx) {
  const cors = corsHeaders(env, request);
  if (request.method !== 'POST') {
    return jsonResponse({ ok: false, error: 'method_not_allowed' }, 405, cors);
  }

  const userAgent = request.headers.get('User-Agent') || '';
  if (isBot(userAgent)) {
    return jsonResponse({ ok: true, skipped: 'bot' }, 200, cors);
  }

  if (
    String(env.STRICT_REFERER || '').toLowerCase() === 'true' &&
    !refererAllowed(request.headers.get('Referer'), env.ALLOWED_ORIGIN)
  ) {
    return jsonResponse({ ok: true, skipped: 'referer' }, 200, cors);
  }

  const day = localDay(Date.now(), tzOffset(env));
  const geo = normalizeGeo(request.cf);
  const vid = await makeVid({
    ip: pickIp(request.headers),
    userAgent,
    secret: env.SALT_SECRET || 'pcy-analytics-insecure-dev-salt',
    monthKey: monthKeyOf(day),
  });

  // 短窗口去重：同一访客连续刷新 / 多标签同时打开，只算一次。
  const dedupeSeconds = Number(env.DEDUPE_SECONDS || 0);
  if (dedupeSeconds > 0) {
    const dedupeKey = new Request(`https://dedupe.pcy-analytics.internal/${day}/${vid}`);
    const hit = await caches.default.match(dedupeKey);
    if (hit) return jsonResponse({ ok: true, skipped: 'dedupe' }, 200, cors);
    ctx.waitUntil(
      caches.default.put(
        dedupeKey,
        new Response('1', { headers: { 'cache-control': `max-age=${Math.floor(dedupeSeconds)}` } })
      )
    );
  }

  try {
    await env.DB.batch([
      env.DB.prepare(UPSERT_DAILY_SQL).bind(day, geo.country, geo.province, geo.city),
      env.DB.prepare(INSERT_VISITOR_SQL).bind(day, vid),
    ]);
  } catch (error) {
    // 采集失败不应影响访客，也不要把 SQL 细节回给浏览器。
    return jsonResponse({ ok: false, error: 'store_failed', detail: String(error && error.message) }, 500, cors);
  }

  return jsonResponse({ ok: true }, 200, cors);
}

/** GET /total —— 公开计数，给页脚用。边缘缓存 5 分钟，避免页脚把 D1 读额度吃光。 */
async function handleTotal(request, env, ctx) {
  const cors = corsHeaders(env, request);
  const day = localDay(Date.now(), tzOffset(env));
  const cacheKey = new Request(`https://total.pcy-analytics.internal/${day}`);
  const headers = { 'content-type': JSON_CT, 'cache-control': 'public, max-age=300' };

  // 缓存里只放纯数据，CORS 头每次按调用方 Origin 重新拼 ——
  // 否则第一个访客的 Origin 会被缓存下来，发给后面所有来源。
  const cached = await caches.default.match(cacheKey);
  if (cached) {
    return new Response(await cached.text(), { headers: { ...headers, ...cors, 'x-pcy-cache': 'hit' } });
  }

  const row = await env.DB.prepare(TOTALS_SQL)
    .bind(day, `${monthKeyOf(day)}-01`, `${yearKeyOf(day)}-01-01`)
    .first();

  const payload = {
    total: Number((row && row.total) || 0),
    today: Number((row && row.today) || 0),
    month: Number((row && row.month) || 0),
    year: Number((row && row.year) || 0),
  };
  const body = JSON.stringify(payload);

  ctx.waitUntil(caches.default.put(cacheKey, new Response(body, { headers })));
  return new Response(body, { headers: { ...headers, ...cors, 'x-pcy-cache': 'miss' } });
}

/**
 * 把看板需要的数据一次批量取齐。
 * `/dash` 与 `/stats` 共用这一份，避免两条路由的取数口径漂移。
 */
async function loadDashboardData(env, day, range) {
  const monthStart = `${monthKeyOf(day)}-01`;
  const yearStart = `${yearKeyOf(day)}-01-01`;
  const recentStart = addDays(day, -29);
  // 多取一周，保证热力图最左一列要么整列有数据、要么整列是空档，不会半列缺数。
  const heatStart = addDays(day, -(HEATMAP_WEEKS * 7 + 7));
  const geoRange = rangeOf(range, day);

  const [statsRes, todayUvRes, monthUvRes, yearlyRes, monthlyRes, recentRes, heatRes, geoRes] = await env.DB.batch([
    env.DB.prepare(TOTALS_SQL).bind(day, monthStart, yearStart),
    env.DB.prepare(TODAY_UV_SQL).bind(day),
    env.DB.prepare(RANGE_UV_SQL).bind(monthStart),
    env.DB.prepare(YEARLY_SQL),
    env.DB.prepare(MONTHLY_SQL),
    env.DB.prepare(DAILY_SQL).bind(recentStart),
    env.DB.prepare(DAILY_SQL).bind(heatStart),
    env.DB.prepare(geoRankSql(geoRange.clause)).bind(...geoRange.params),
  ]);

  const statsRow = statsRes.results[0] || {};
  const yearly = (yearlyRes.results || []).map((r) => ({ k: String(r.k), v: Number(r.v) || 0 }));
  const monthly = (monthlyRes.results || []).map((r) => ({ k: String(r.k), v: Number(r.v) || 0 }));
  const recent = (recentRes.results || []).map((r) => ({ k: String(r.k), v: Number(r.v) || 0 }));
  const heatRows = (heatRes.results || []).map((r) => ({ k: String(r.k), v: Number(r.v) || 0 }));
  const geos = (geoRes.results || []).map((r) => ({
    country: String(r.country || 'XX'),
    province: String(r.province || ''),
    city: String(r.city || ''),
    v: Number(r.v) || 0,
  }));

  return {
    day,
    range,
    stats: {
      total: Number(statsRow.total) || 0,
      today: Number(statsRow.today) || 0,
      month: Number(statsRow.month) || 0,
      year: Number(statsRow.year) || 0,
      todayUv: Number((todayUvRes.results[0] || {}).uv) || 0,
      monthUv: Number((monthUvRes.results[0] || {}).uv) || 0,
    },
    yearly,
    monthly,
    recent,
    geos,
    calendar: buildCalendar(heatRows, { endDay: day, weeks: HEATMAP_WEEKS }),
  };
}

/** GET /dash —— 需口令的看板，用于自己单独打开。 */
async function handleDash(request, env, url) {
  const token = url.searchParams.get('token') || request.headers.get('x-dash-token') || '';
  if (!env.DASH_TOKEN || !safeEqual(token, env.DASH_TOKEN)) {
    return htmlResponse(renderUnauthorized(), 401);
  }

  const day = localDay(Date.now(), tzOffset(env));
  const range = normalizeRange(url.searchParams.get('range'));
  const data = await loadDashboardData(env, day, range);
  const embed = url.searchParams.get('embed') === '1';
  return htmlResponse(renderDashboard({ ...data, token, embed, tmapKey: env.TMAP_KEY || '' }));
}

/**
 * GET /stats —— 公开只读看板，供「关于」页 iframe 内嵌。
 *
 * 这条路故意不要口令：页面本身就是公开给访客看的，把口令写进可被查看的 iframe src
 * 等于把口令也公开了，不如直接做一个公开视图。边缘缓存 5 分钟，
 * 否则每个打开关于页的人都会打 8 次 D1 查询，白吃免费额度。
 */
async function handleStats(request, env, ctx, url) {
  const day = localDay(Date.now(), tzOffset(env));
  const range = normalizeRange(url.searchParams.get('range'));
  const embed = url.searchParams.get('embed') === '1';

  const cacheKey = new Request(`https://stats.pcy-analytics.internal/${day}/${range}/${embed ? 'embed' : 'full'}`);
  const headers = {
    'content-type': HTML_CT,
    'cache-control': `public, max-age=${STATS_CACHE_SECONDS}`,
    'x-robots-tag': 'noindex',
    // 地图 key 的域名校验看 Referer，必须放行 —— 只发来源、不发路径。
    'referrer-policy': 'strict-origin-when-cross-origin',
  };

  const cached = await caches.default.match(cacheKey);
  if (cached) {
    return new Response(await cached.text(), { headers: { ...headers, 'x-pcy-cache': 'hit' } });
  }

  const data = await loadDashboardData(env, day, range);
  const html = renderDashboard({ ...data, token: '', embed, tmapKey: env.TMAP_KEY || '' });

  ctx.waitUntil(
    caches.default.put(cacheKey, new Response(html, { headers: { 'content-type': HTML_CT } }))
  );
  return new Response(html, { headers: { ...headers, 'x-pcy-cache': 'miss' } });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const cors = corsHeaders(env, request);
    // endpoint 配成带尾斜杠时，站点侧会拼出 `//stats`。折叠重复斜杠，
    // 免得一个配置笔误在页面上表现成「看板 404」这种难查的现象。
    const pathname = url.pathname.replace(/\/{2,}/g, '/');

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }

    switch (pathname) {
      case '/c':
        return handleCollect(request, env, ctx);
      case '/total':
        return handleTotal(request, env, ctx);
      case '/stats':
        return handleStats(request, env, ctx, url);
      case '/dash':
        return handleDash(request, env, url);
      case '/':
      case '/health':
        return jsonResponse({ ok: true, service: 'pcy-analytics' }, 200, cors);
      default:
        return jsonResponse({ ok: false, error: 'not_found' }, 404, cors);
    }
  },
};
