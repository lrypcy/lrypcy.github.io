/**
 * 纯逻辑单测：不依赖 Cloudflare、不依赖网络。
 * 运行：node --test tools/analytics/test/
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  isBot,
  localDay,
  makeVid,
  monthKeyOf,
  normalizeGeo,
  normalizeRange,
  parseOrigins,
  percentOf,
  pickIp,
  rangeOf,
  refererAllowed,
  safeEqual,
  yearKeyOf,
} from '../worker/src/lib.js';
import { CN_PROVINCES, countryLabel, geoLabel, provinceLabel } from '../worker/src/names.js';

test('localDay：按 UTC+8 归属统计日，跨零点正确滚动', () => {
  // UTC 15:24:35 → 北京时间 23:24:35，仍是 10-01
  assert.equal(localDay(Date.UTC(2026, 9, 1, 15, 24, 35)), '2026-10-01');
  // UTC 16:00 → 北京时间次日 00:00，应滚动到 10-02
  assert.equal(localDay(Date.UTC(2026, 9, 1, 16, 0, 0)), '2026-10-02');
  // 月末跨月
  assert.equal(localDay(Date.UTC(2026, 8, 30, 17, 0, 0)), '2026-10-01');
  // 年末跨年
  assert.equal(localDay(Date.UTC(2026, 11, 31, 16, 30, 0)), '2027-01-01');
  // UTC 白天不应该被推移
  assert.equal(localDay(Date.UTC(2026, 9, 1, 0, 0, 0)), '2026-10-01');
});

test('monthKeyOf / yearKeyOf', () => {
  assert.equal(monthKeyOf('2026-10-01'), '2026-10');
  assert.equal(yearKeyOf('2026-10-01'), '2026');
});

test('rangeOf：四种范围的时间窗边界', () => {
  assert.deepEqual(rangeOf('day', '2026-10-01'), { clause: 'day = ?', params: ['2026-10-01'] });
  assert.deepEqual(rangeOf('month', '2026-10-01'), {
    clause: 'day >= ? AND day <= ?',
    params: ['2026-10-01', '2026-10-31'],
  });
  assert.deepEqual(rangeOf('year', '2026-10-01'), {
    clause: 'day >= ? AND day <= ?',
    params: ['2026-01-01', '2026-12-31'],
  });
  assert.deepEqual(rangeOf('all', '2026-10-01'), { clause: '1 = 1', params: [] });
  assert.throws(() => rangeOf('week', '2026-10-01'), /unknown range/);
});

test('rangeOf：月度窗口用 -31 收口，二月/闰年也不会漏掉月末', () => {
  // 不存在 2024-02-31 这样的数据行，所以上界写 31 是安全的（字典序收口）
  assert.deepEqual(rangeOf('month', '2024-02-15').params, ['2024-02-01', '2024-02-31']);
  assert.ok('2024-02-29' >= '2024-02-01' && '2024-02-29' <= '2024-02-31');
});

test('isBot：真实浏览器放行，爬虫/监控/空 UA 拦截', () => {
  assert.equal(isBot('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140.0 Safari/537.36'), false);
  assert.equal(isBot('Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1'), false);
  // "robot" 里虽然含 "bot"，但前面是字母，不构成单词边界，不应误伤
  assert.equal(isBot('Mozilla/5.0 RoboTest/1.0 Chrome/140.0'), false);

  assert.equal(isBot(''), true);
  assert.equal(isBot('   '), true);
  assert.equal(isBot('Googlebot/2.1 (+http://www.google.com/bot.html)'), true);
  assert.equal(isBot('Mozilla/5.0 (compatible; bingbot/2.0)'), true);
  assert.equal(isBot('Mozilla/5.0 (compatible; Baiduspider/2.0)'), true);
  assert.equal(isBot('Mozilla/5.0 (compatible; Bytespider)'), true);
  assert.equal(isBot('curl/8.7.1'), true);
  assert.equal(isBot('python-requests/2.32.3'), true);
  assert.equal(isBot('UptimeRobot/2.0'), true);
  assert.equal(isBot('HeadlessChrome/140.0.0.0'), true);
});

test('pickIp：优先级 CF-Connecting-IP > X-Real-IP > X-Forwarded-For 首项', () => {
  assert.equal(pickIp(new Headers({ 'CF-Connecting-IP': '1.2.3.4' })), '1.2.3.4');
  assert.equal(pickIp(new Headers({ 'X-Real-IP': '5.6.7.8' })), '5.6.7.8');
  assert.equal(pickIp(new Headers({ 'X-Forwarded-For': '9.9.9.9, 10.0.0.1' })), '9.9.9.9');
  assert.equal(pickIp(new Headers()), '');
});

test('normalizeGeo：正常、缺失、Tor 与 CN 短码大写', () => {
  assert.deepEqual(normalizeGeo({ country: 'cn', regionCode: 'gd', city: 'Shenzhen' }), {
    country: 'CN',
    province: 'GD',
    city: 'Shenzhen',
  });
  // 非中国不强制大写省份，保留 Cloudflare 的英文名
  assert.deepEqual(normalizeGeo({ country: 'us', region: 'California', city: 'San Francisco' }), {
    country: 'US',
    province: 'California',
    city: 'San Francisco',
  });
  // 无法定位
  assert.deepEqual(normalizeGeo({}), { country: 'XX', province: '', city: '' });
  assert.deepEqual(normalizeGeo(null), { country: 'XX', province: '', city: '' });
  // Tor 出口节点归入未知
  assert.deepEqual(normalizeGeo({ country: 'T1' }), { country: 'XX', province: '', city: '' });
  // regionCode 优先于 region
  assert.deepEqual(normalizeGeo({ country: 'CN', region: 'Guangdong', regionCode: 'GD' }).province, 'GD');
  // 超长字段截断，防止脏数据撑大主键
  assert.equal(normalizeGeo({ country: 'CN', regionCode: 'G'.repeat(50) }).province.length, 32);
  assert.equal(normalizeGeo({ country: 'CN', city: 'C'.repeat(90) }).city.length, 48);
});

test('makeVid：同月稳定、跨月变化、不同设备不同', async () => {
  const base = { ip: '203.0.113.7', userAgent: 'Mozilla/5.0 Chrome/140.0', secret: 's3cret' };
  const oct1 = await makeVid({ ...base, monthKey: '2026-10' });
  const oct2 = await makeVid({ ...base, monthKey: '2026-10' });
  const sep = await makeVid({ ...base, monthKey: '2026-09' });
  const other = await makeVid({ ...base, ip: '203.0.113.8', monthKey: '2026-10' });

  assert.equal(oct1, oct2, '同月同设备必须得到同一个 vid，否则日 UV 会虚高');
  assert.notEqual(oct1, sep, '跨月必须换盐，否则可以长期跟踪同一个访客');
  assert.notEqual(oct1, other, '不同 IP 必须区分');
  assert.match(oct1, /^[0-9a-f]{24}$/);
  assert.equal(oct1.length, 24);
});

test('safeEqual：仅长度与内容都相同才为真，空口令一律拒绝', () => {
  assert.equal(safeEqual('abc123', 'abc123'), true);
  assert.equal(safeEqual('abc123', 'abc124'), false);
  assert.equal(safeEqual('abc', 'abc123'), false);
  assert.equal(safeEqual('', ''), false);
  assert.equal(safeEqual(null, 'abc'), false);
  assert.equal(safeEqual('abc', undefined), false);
});

test('refererAllowed：本站放行、他站拒绝、空 Referer 放行', () => {
  const origins = 'https://lrypcy.github.io, https://www.example.com/';
  assert.equal(refererAllowed('https://lrypcy.github.io/2026/10/01/post/', origins), true);
  assert.equal(refererAllowed('https://lrypcy.github.io', origins), true);
  assert.equal(refererAllowed('https://www.example.com/page', origins), true);
  assert.equal(refererAllowed('https://evil.test/page', origins), false);
  assert.equal(refererAllowed('https://lrypcy.github.io.evil.test/', origins), false);
  assert.equal(refererAllowed('', origins), true, '浏览器隐私设置可能不带 Referer，不应丢弃真实流量');
  assert.equal(refererAllowed('', ''), true);
});

test('parseOrigins / normalizeRange / percentOf', () => {
  assert.deepEqual(parseOrigins('https://a.com/, https://b.com'), ['https://a.com', 'https://b.com']);
  assert.deepEqual(parseOrigins(''), []);
  assert.equal(normalizeRange('MONTH'), 'month');
  assert.equal(normalizeRange('week'), 'all');
  assert.equal(normalizeRange(null), 'all');
  assert.equal(percentOf(1, 4), 25);
  assert.equal(percentOf(1, 0), 0);
  assert.equal(percentOf(0, 0), 0);
  assert.equal(percentOf(1, 3), 33.3);
});

test('地理展示名：代码映射到中文，未收录回退原码', () => {
  assert.equal(countryLabel('CN'), '中国');
  assert.equal(countryLabel('hk'), '中国香港');
  assert.equal(countryLabel('TW'), '中国台湾');
  assert.equal(countryLabel('ZZ'), 'ZZ');
  assert.equal(countryLabel(''), '未知地区');
  assert.equal(provinceLabel('CN', 'gd'), '广东');
  assert.equal(provinceLabel('CN', 'BJ'), '北京');
  assert.equal(provinceLabel('US', 'California'), 'California');
  assert.equal(provinceLabel('CN', ''), '');
  assert.equal(Object.keys(CN_PROVINCES).length, 34);

  assert.equal(geoLabel({ country: 'CN', province: 'GD', city: 'Shenzhen' }), '中国 · 广东 · Shenzhen');
  assert.equal(geoLabel({ country: 'CN', province: 'GD', city: '' }), '中国 · 广东');
  assert.equal(geoLabel({ country: 'US', province: 'California', city: 'SF' }), '美国 · California · SF');
  assert.equal(geoLabel({ country: 'XX', province: '', city: '' }), '未知地区');
});
