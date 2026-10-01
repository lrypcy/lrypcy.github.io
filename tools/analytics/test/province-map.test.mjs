/**
 * 省级方块地图的单测。
 *
 * 重点不在「画得好看」，而在三条会静默出错的规则：
 *   1. 方块布局必须恰好覆盖 34 个省级行政区、且没有两格落在同一个位置；
 *   2. 港澳台整块归位（Cloudflare 给它们的 country 是 HK/MO/TW，不带 province）；
 *   3. 定位不到的流量必须被汇总出来，不能被无声吞掉。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  TILE_LAYOUT,
  TILE_SHORT,
  TILE_COLS,
  TILE_ROWS,
  tileKeyOf,
  inMapScope,
  buildProvinceTiles,
} from '../worker/src/province-map.js';
import { CN_PROVINCES } from '../worker/src/names.js';
import { levelByRatio } from '../worker/src/lib.js';

test('方块布局恰好覆盖 34 个省级行政区，与 names.js 的表一致', () => {
  const layout = Object.keys(TILE_LAYOUT).sort();
  const table = Object.keys(CN_PROVINCES).sort();
  assert.equal(layout.length, 34);
  assert.deepEqual(layout, table, 'TILE_LAYOUT 与 CN_PROVINCES 必须同集合（含港澳台）');
});

test('每个方块都有短名，且短名不重复', () => {
  const shorts = Object.keys(TILE_LAYOUT).map((code) => TILE_SHORT[code]);
  assert.equal(shorts.filter(Boolean).length, 34, '不许有方块缺短名');
  assert.equal(new Set(shorts).size, 34, '短名不能重复');
  // 方块只有几十像素宽，短名太长会溢出
  for (const [code, name] of Object.entries(TILE_SHORT)) {
    assert.ok(name.length <= 3, `${code} 的短名「${name}」超过 3 个字`);
  }
});

test('没有两个方块落在同一个网格位置', () => {
  const seen = new Map();
  for (const [code, [col, row]] of Object.entries(TILE_LAYOUT)) {
    assert.ok(Number.isInteger(col) && Number.isInteger(row), `${code} 的坐标必须是整数`);
    assert.ok(col >= 0 && col < TILE_COLS, `${code} 的列 ${col} 越界`);
    assert.ok(row >= 0 && row < TILE_ROWS, `${code} 的行 ${row} 越界`);
    const key = `${col},${row}`;
    assert.equal(seen.get(key), undefined, `${code} 与 ${seen.get(key)} 都落在 (${key})`);
    seen.set(key, code);
  }
});

test('港澳台整块归位：国家码即方块码，不依赖 province', () => {
  assert.equal(tileKeyOf('HK', ''), 'HK');
  assert.equal(tileKeyOf('hk', ''), 'HK');
  assert.equal(tileKeyOf('MO', ''), 'MO');
  assert.equal(tileKeyOf('TW', ''), 'TW');
  assert.equal(inMapScope('HK'), true);
  assert.equal(inMapScope('TW'), true);
});

test('大陆按省码落格；英文省名与空省码落不了格', () => {
  assert.equal(tileKeyOf('CN', 'GD'), 'GD');
  assert.equal(tileKeyOf('CN', 'gd'), 'GD');
  assert.equal(tileKeyOf('CN', 'Guangdong'), null, '英文省名不属于 ISO 短码');
  assert.equal(tileKeyOf('CN', ''), null);
  assert.equal(tileKeyOf('US', 'California'), null, '海外不进这张图');
});

test('海外国家既不进图也不计入未定位', () => {
  const grid = buildProvinceTiles([
    { country: 'US', province: 'California', v: 100 },
    { country: 'JP', province: '13', v: 50 },
  ]);
  assert.equal(grid.mapped, 0);
  assert.equal(grid.unmapped, 0, '海外流量由「国家 / 地区排行」负责，不该算成未定位');
  assert.equal(grid.activeTiles, 0);
});

test('聚合：同省多城相加，大陆未定位与海外分开统计', () => {
  const grid = buildProvinceTiles([
    { country: 'CN', province: 'GD', city: 'Shenzhen', v: 30 },
    { country: 'CN', province: 'GD', city: 'Guangzhou', v: 12 },
    { country: 'CN', province: 'BJ', city: 'Beijing', v: 20 },
    { country: 'HK', province: '', city: 'Hong Kong', v: 8 },
    { country: 'CN', province: '', city: '', v: 5 }, // 只定位到国家
    { country: 'XX', province: '', city: '', v: 3 }, // 未定位
    { country: 'US', province: 'California', city: 'SF', v: 40 },
  ]);
  const byCode = Object.fromEntries(grid.tiles.map((t) => [t.code, t.v]));

  assert.equal(byCode.GD, 42);
  assert.equal(byCode.BJ, 20);
  assert.equal(byCode.HK, 8);
  assert.equal(byCode.TW, 0);
  assert.equal(grid.mapped, 70);
  assert.equal(grid.unmapped, 5, '只定位到国家的 CN 流量要单独报出来');
  assert.equal(grid.activeTiles, 3);
  assert.equal(grid.max, 42);
});

test('网格恒定输出 34 格，值全为 0 时也不缺格', () => {
  const grid = buildProvinceTiles([]);
  assert.equal(grid.tiles.length, 34);
  assert.equal(grid.mapped, 0);
  assert.equal(grid.max, 0);
  assert.ok(grid.tiles.every((t) => t.level === 0), '没有数据时全部是 0 档');
  assert.equal(grid.cols, TILE_COLS);
  assert.equal(grid.rows, TILE_ROWS);
});

test('色阶与热力图同源：最大值为 1 时给最高档，避免唯一有色格只剩最浅色', () => {
  assert.equal(levelByRatio(0, 0), 0);
  assert.equal(levelByRatio(1, 1), 4);
  assert.equal(levelByRatio(1, 100), 1);
  assert.equal(levelByRatio(25, 100), 1);
  assert.equal(levelByRatio(50, 100), 2);
  assert.equal(levelByRatio(75, 100), 3);
  assert.equal(levelByRatio(100, 100), 4);
  assert.equal(levelByRatio(90, 100, 3), 2, '档数可调');

  const grid = buildProvinceTiles([{ country: 'CN', province: 'GD', v: 1 }]);
  const gd = grid.tiles.find((t) => t.code === 'GD');
  assert.equal(gd.level, 4);
});

test('负值与非数字不会污染统计', () => {
  const grid = buildProvinceTiles([
    { country: 'CN', province: 'GD', v: -5 },
    { country: 'CN', province: 'BJ', v: 'abc' },
    { country: 'CN', province: 'SH', v: null },
  ]);
  assert.equal(grid.mapped, 0);
  assert.equal(grid.activeTiles, 0);
});
