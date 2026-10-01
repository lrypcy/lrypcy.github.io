/**
 * pcy-analytics 的纯逻辑层。
 *
 * 这里刻意不引用任何 Workers / D1 运行时对象（request.cf、env.DB、caches…），
 * 只处理「时间换算 / 访客标识 / 地理归一 / SQL 片段拼装 / 请求取舍」，
 * 这样 node 可以直接 import 本文件跑单测，不必登录 Cloudflare。
 */

export const DEFAULT_TZ_OFFSET_HOURS = 8;

/** 站点时区下的 YYYY-MM-DD。中国全境 UTC+8 且无夏令时，固定偏移即可。 */
export function localDay(tsMs, offsetHours = DEFAULT_TZ_OFFSET_HOURS) {
  return new Date(tsMs + offsetHours * 3600 * 1000).toISOString().slice(0, 10);
}

export function monthKeyOf(day) {
  return day.slice(0, 7);
}

export function yearKeyOf(day) {
  return day.slice(0, 4);
}

/**
 * 把 day / month / year / all 翻译成 SQL 的 day 过滤片段。
 * day 是零填充日期串，字典序等于时间序，所以可以用字符串比较并走索引。
 */
export function rangeOf(kind, day) {
  switch (kind) {
    case 'day':
      return { clause: 'day = ?', params: [day] };
    case 'month': {
      const m = monthKeyOf(day);
      return { clause: 'day >= ? AND day <= ?', params: [`${m}-01`, `${m}-31`] };
    }
    case 'year': {
      const y = yearKeyOf(day);
      return { clause: 'day >= ? AND day <= ?', params: [`${y}-01-01`, `${y}-12-31`] };
    }
    case 'all':
      return { clause: '1 = 1', params: [] };
    default:
      throw new Error(`unknown range: ${kind}`);
  }
}

/**
 * 爬虫 / 监控 / 预览请求的 UA 特征。
 * 注意 `bot\b` 不会误伤 "robot"（`t` 后面不是单词边界），这是刻意的。
 */
const BOT_RE =
  /(bot\b|crawler|spider|crawl-|slurp|bingpreview|headless|phantomjs|python-requests|python-urllib|aiohttp|httpx|curl\/|wget|libwww|lwp::|java\/|okhttp|go-http-client|httpclient|node-fetch|axios\/|got\/|undici|facebookexternalhit|embedly|quora link preview|outbrain|pinterest|vkshare|w3c_validator|semrush|ahrefs|mj12bot|dotbot|bytespider|petalbot|yandex|baiduspider|sogou|360spider|exabot|ia_archiver|scrapy|monitor|uptime|pingdom|statuscake|newrelic|datadog|site24x7|preview|feedfetcher|feedburner|rss)/i;

/** 空 UA 一律当作非人类流量丢弃（正常浏览器不会不发 UA）。 */
export function isBot(userAgent) {
  const ua = String(userAgent || '').trim();
  if (!ua) return true;
  return BOT_RE.test(ua);
}

/**
 * 从请求头里取客户端 IP。Worker 场景下 CF-Connecting-IP 总是存在。
 * 这个值只在内存里用于算哈希，绝不落库。
 */
export function pickIp(headers) {
  return (
    headers.get('CF-Connecting-IP') ||
    headers.get('X-Real-IP') ||
    (headers.get('X-Forwarded-For') || '').split(',')[0].trim() ||
    ''
  );
}

/**
 * 归一化 Cloudflare 的地理信息。
 * cf.country 为空时用 'XX' 占位，让看板能看出「未定位流量」占比。
 * cf.regionCode（ISO 3166-2 短码，如中国广东 = GD）比 cf.region（英文全名）更适合做统计维度。
 */
export function normalizeGeo(cf) {
  const src = cf || {};
  const country = String(src.country || '').trim().toUpperCase();
  if (!country || country === 'T1') {
    return { country: 'XX', province: '', city: '' };
  }
  const rawProvince = String(src.regionCode || src.region || '').trim();
  const province = country === 'CN' ? rawProvince.toUpperCase() : rawProvince;
  return {
    country: country.slice(0, 8),
    province: province.slice(0, 32),
    city: String(src.city || '').trim().slice(0, 48),
  };
}

export function toHex(bytes) {
  let out = '';
  for (const b of bytes) out += b.toString(16).padStart(2, '0');
  return out;
}

/**
 * 访客标识：SHA-256(盐 | 月份 | IP | UA) 截断 24 位十六进制。
 * 同一个月内同设备稳定（所以「本月独立访客」准确），跨月变化（所以无法长期跟踪个人）。
 * 原始 IP / UA 不落库，也不往浏览器写 cookie 或 localStorage。
 */
export async function makeVid({ ip, userAgent, secret, monthKey }) {
  const payload = `${secret}|${monthKey}|${ip}|${userAgent}`;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(payload));
  return toHex(new Uint8Array(digest)).slice(0, 24);
}

/** 解析 ALLOWED_ORIGIN（逗号分隔）为数组。 */
export function parseOrigins(value) {
  return String(value || '')
    .split(',')
    .map((s) => s.trim().replace(/\/+$/, ''))
    .filter(Boolean);
}

/**
 * Referer 是否可信。
 * Referer 为空的请求照常统计 —— 浏览器隐私设置、从地址栏直接打开、https 跳转都可能不带 Referer，
 * 一刀切会把真实流量丢掉；带 Referer 但不属于本站的才判定为可疑。
 */
export function refererAllowed(referer, origins) {
  const list = parseOrigins(origins);
  if (!list.length) return true;
  const ref = String(referer || '').trim();
  if (!ref) return true;
  return list.some((origin) => ref === origin || ref.startsWith(`${origin}/`));
}

/** 常量时间字符串比较，避免看板口令按字符逐位泄露。 */
export function safeEqual(a, b) {
  const x = String(a == null ? '' : a);
  const y = String(b == null ? '' : b);
  if (x.length !== y.length || x.length === 0) return false;
  let diff = 0;
  for (let i = 0; i < x.length; i += 1) diff |= x.charCodeAt(i) ^ y.charCodeAt(i);
  return diff === 0;
}

/** 把 querystring 里的 range 收敛到允许值，默认 all。 */
export function normalizeRange(value) {
  const v = String(value || '').toLowerCase();
  return ['day', 'month', 'year', 'all'].includes(v) ? v : 'all';
}

/** 百分比（0~100），分母为 0 时返回 0。用于看板条形宽度。 */
export function percentOf(value, total) {
  if (!total || total <= 0) return 0;
  return Math.round((Number(value) / Number(total)) * 1000) / 10;
}

/* ------------------------------------------------------------------ *
 * SQL 常量：实现与测试共用同一份，避免两边漂移。
 * ------------------------------------------------------------------ */

/** 一次访问 = 聚合表对应格子 +1；同一地区同一天重复访问只让 pv 增长，不新增行。 */
export const UPSERT_DAILY_SQL = `INSERT INTO daily_stats (day, country, province, city, pv)
VALUES (?1, ?2, ?3, ?4, 1)
ON CONFLICT (day, country, province, city) DO UPDATE SET pv = pv + 1`;

/** 当天访客哈希；同一天同一人重复访问不会新增行（用于日 UV）。 */
export const INSERT_VISITOR_SQL = `INSERT OR IGNORE INTO visitor_days (day, vid) VALUES (?1, ?2)`;

/** 总 / 今日 / 本月 / 本年，一次全表扫描拿到。 */
export const TOTALS_SQL = `SELECT
  COALESCE(SUM(pv), 0) AS total,
  COALESCE(SUM(CASE WHEN day = ?1 THEN pv END), 0) AS today,
  COALESCE(SUM(CASE WHEN day >= ?2 THEN pv END), 0) AS month,
  COALESCE(SUM(CASE WHEN day >= ?3 THEN pv END), 0) AS year
FROM daily_stats`;

/** 年月日分组曲线。 */
export const YEARLY_SQL = `SELECT substr(day, 1, 4) AS k, SUM(pv) AS v FROM daily_stats GROUP BY k ORDER BY k`;
export const MONTHLY_SQL = `SELECT substr(day, 1, 7) AS k, SUM(pv) AS v FROM daily_stats GROUP BY k ORDER BY k`;
export const DAILY_SQL = `SELECT day AS k, SUM(pv) AS v FROM daily_stats WHERE day >= ?1 GROUP BY day ORDER BY day`;

/** 地理排行（按范围过滤后，省 / 城市级明细）。 */
export function geoRankSql(rangeClause) {
  return `SELECT country, province, city, SUM(pv) AS v
    FROM daily_stats
   WHERE ${rangeClause}
   GROUP BY country, province, city
   ORDER BY v DESC
   LIMIT 2000`;
}

export const TODAY_UV_SQL = `SELECT COUNT(*) AS uv FROM visitor_days WHERE day = ?1`;
export const RANGE_UV_SQL = `SELECT COUNT(DISTINCT vid) AS uv FROM visitor_days WHERE day >= ?1`;
