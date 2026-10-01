/**
 * 省级访问分布的聚合与落点单测。
 *
 * 重点不在「画得好看」，而在几条会静默出错的规则：
 *   1. 落点表必须恰好覆盖 34 个省级行政区，且与 names.js 的表同集合（含港澳台）；
 *   2. 港澳台整块归位（Cloudflare 给它们的 country 是 HK/MO/TW，不带 province）；
 *   3. 定位不到的流量必须被汇总出来，不能被无声吞掉；
 *   4. 没访问的省份不落点，避免地图上飘满空心圆。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  PROVINCE_CENTER,
  PROVINCE_SHORT,
  provinceCodeOf,
  inMapScope,
  buildProvinceStats,
} from '../worker/src/province-map.js';
import { CN_PROVINCES } from '../worker/src/names.js';

test('落点表恰好覆盖 34 个省级行政区，与 names.js 的表同集合', () => {
  const codes = Object.keys(PROVINCE_CENTER).sort();
  assert.equal(codes.length, 34);
  assert.deepEqual(codes, Object.keys(CN_PROVINCES).sort(), 'PROVINCE_CENTER 与 CN_PROVINCES 必须同集合（含港澳台）');
});

test('每个省级行政区都有短名，短名不重复且不超过 3 字', () => {
  const shorts = Object.keys(PROVINCE_CENTER).map((code) => PROVINCE_SHORT[code]);
  assert.equal(shorts.filter(Boolean).length, 34, '不许有省份缺短名');
  assert.equal(new Set(shorts).size, 34, '短名不能重复');
  for (const [code, name] of Object.entries(PROVINCE_SHORT)) {
    assert.ok(name.length <= 3, `${code} 的短名「${name}」超过 3 个字`);
  }
});

test('落点是合法经纬度、大致落在中国范围内、且无重复', () => {
  const seen = new Map();
  for (const [code, [lat, lng]] of Object.entries(PROVINCE_CENTER)) {
    assert.ok(Number.isFinite(lat) && Number.isFinite(lng), `${code} 落点不是数字`);
    assert.ok(lat >= 3 && lat <= 54, `${code} 纬度 ${lat} 不在中国范围内`);
    assert.ok(lng >= 73 && lng <= 136, `${code} 经度 ${lng} 不在中国范围内`);
    const key = `${lat},${lng}`;
    assert.equal(seen.get(key), undefined, `${code} 与 ${seen.get(key)} 落点重复`);
    seen.set(key, code);
  }
});

test('港澳台整块归位：国家码即省份码，不依赖 province', () => {
  assert.equal(provinceCodeOf('HK', ''), 'HK');
  assert.equal(provinceCodeOf('hk', ''), 'HK');
  assert.equal(provinceCodeOf('MO', ''), 'MO');
  assert.equal(provinceCodeOf('TW', ''), 'TW');
  assert.equal(inMapScope('HK'), true);
  assert.equal(inMapScope('TW'), true);
});

test('大陆按省码落点；英文省名与空省码落不了点', () => {
  assert.equal(provinceCodeOf('CN', 'GD'), 'GD');
  assert.equal(provinceCodeOf('CN', 'gd'), 'GD');
  assert.equal(provinceCodeOf('CN', 'Guangdong'), null, '英文省名不是 ISO 短码');
  assert.equal(provinceCodeOf('CN', ''), null);
  assert.equal(provinceCodeOf('US', 'California'), null, '海外不进这张图');
});

test('海外国家既不进图也不计入未定位', () => {
  const stats = buildProvinceStats([
    { country: 'US', province: 'California', v: 100 },
    { country: 'JP', province: '13', v: 50 },
  ]);
  assert.equal(stats.mapped, 0);
  assert.equal(stats.unmapped, 0, '海外流量由「国家 / 地区排行」负责，不算未定位');
  assert.equal(stats.activeCount, 0);
  assert.equal(stats.items.length, 0);
});

test('聚合：同省多城相加；只含有访问的省份，按量降序', () => {
  const stats = buildProvinceStats([
    { country: 'CN', province: 'GD', city: 'Shenzhen', v: 30 },
    { country: 'CN', province: 'GD', city: 'Guangzhou', v: 12 },
    { country: 'CN', province: 'BJ', city: 'Beijing', v: 20 },
    { country: 'HK', province: '', city: 'Hong Kong', v: 8 },
    { country: 'CN', province: '', city: '', v: 5 }, // 只定位到国家
    { country: 'XX', province: '', city: '', v: 3 }, // 未定位
    { country: 'US', province: 'California', city: 'SF', v: 40 },
  ]);
  const byCode = Object.fromEntries(stats.items.map((p) => [p.code, p.v]));

  assert.equal(byCode.GD, 42, '同省多城应相加');
  assert.equal(byCode.BJ, 20);
  assert.equal(byCode.HK, 8);
  assert.equal(stats.items.length, 3, '只有有访问的省份才落点');
  assert.deepEqual(stats.items.map((p) => p.code), ['GD', 'BJ', 'HK'], '按访问量降序');
  assert.equal(stats.mapped, 70);
  assert.equal(stats.unmapped, 5, '只定位到国家的 CN 流量要单独报出来');
  assert.equal(stats.activeCount, 3);
  assert.equal(stats.max, 42);
});

test('落点带经纬度与短名，量最大的省份给最高档', () => {
  const stats = buildProvinceStats([
    { country: 'CN', province: 'GD', v: 100 },
    { country: 'CN', province: 'BJ', v: 1 },
  ]);
  const gd = stats.items.find((p) => p.code === 'GD');
  const bj = stats.items.find((p) => p.code === 'BJ');
  assert.deepEqual([gd.lat, gd.lng], PROVINCE_CENTER.GD);
  assert.equal(gd.name, '广东');
  assert.equal(gd.level, 4, '量最大的省份给最高档');
  assert.ok(bj.level >= 1, '有访问就至少 1 档');
});

test('空输入：不落点、计数为 0，但省级行政区总数仍是 34', () => {
  const stats = buildProvinceStats([]);
  assert.equal(stats.items.length, 0);
  assert.equal(stats.mapped, 0);
  assert.equal(stats.unmapped, 0);
  assert.equal(stats.max, 0);
  assert.equal(stats.activeCount, 0);
  assert.equal(stats.totalCount, 34);
});

test('负值与非数字不会污染统计', () => {
  const stats = buildProvinceStats([
    { country: 'CN', province: 'GD', v: -5 },
    { country: 'CN', province: 'BJ', v: 'abc' },
    { country: 'CN', province: 'SH', v: null },
  ]);
  assert.equal(stats.mapped, 0);
  assert.equal(stats.activeCount, 0);
});

test('geos 为 undefined / null 时不报错', () => {
  assert.equal(buildProvinceStats(undefined).items.length, 0);
  assert.equal(buildProvinceStats(null).mapped, 0);
});
