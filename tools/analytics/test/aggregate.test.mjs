/**
 * 端到端聚合校验：用真实的 schema.sql + Worker 里同一份 SQL 常量，在内存 SQLite 上
 * 复现「若干次访问 → 日 / 月 / 年 / 总 反算」。
 *
 * 这样在登录 Cloudflare 之前就能确认记账口径正确 —— 尤其是
 *   - 同一地区同一天重复访问是累加而不是新增行；
 *   - 跨月范围过滤不会串月；
 *   - 访客去重（UV）在「天」这一层生效。
 *
 * 运行：node --test tools/analytics/test/*.test.mjs
 */

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { SCHEMA_PATH, createSqlite } from '../d1-shim.mjs';
import {
  DAILY_SQL,
  INSERT_VISITOR_SQL,
  MONTHLY_SQL,
  RANGE_UV_SQL,
  TODAY_UV_SQL,
  TOTALS_SQL,
  UPSERT_DAILY_SQL,
  YEARLY_SQL,
  geoRankSql,
  rangeOf,
} from '../worker/src/lib.js';

const GEO = {
  BJ: { country: 'CN', province: 'BJ', city: 'Beijing' },
  SH: { country: 'CN', province: 'SH', city: 'Shanghai' },
  GD: { country: 'CN', province: 'GD', city: 'Shenzhen' },
  US: { country: 'US', province: 'California', city: 'San Francisco' },
  XX: { country: 'XX', province: '', city: '' },
};

/**
 * 构造真值表。[天, 地区, 访客哈希]
 * 期望值先手算出来写死在断言里，再由 SQL 反算校对（而不是反过来）。
 *
 *   2026-09-30: BJ×3 + SH×1 + US×2 = 6
 *   2026-10-01: BJ×2 + SH×5         = 7
 *   2026-10-02: GD×4                = 4
 *   合计 17
 */
const VISITS = [
  ['2026-09-30', GEO.BJ, 'A'],
  ['2026-09-30', GEO.BJ, 'B'],
  ['2026-09-30', GEO.BJ, 'C'],
  ['2026-09-30', GEO.SH, 'A'],
  ['2026-09-30', GEO.US, 'X'],
  ['2026-09-30', GEO.US, 'X'],

  ['2026-10-01', GEO.BJ, 'X'],
  ['2026-10-01', GEO.BJ, 'B'],
  ['2026-10-01', GEO.SH, 'B'],
  ['2026-10-01', GEO.SH, 'M'],
  ['2026-10-01', GEO.SH, 'M'],
  ['2026-10-01', GEO.SH, 'M'],
  ['2026-10-01', GEO.SH, 'M'],

  ['2026-10-02', GEO.GD, 'X'],
  ['2026-10-02', GEO.GD, 'X'],
  ['2026-10-02', GEO.GD, 'M'],
  ['2026-10-02', GEO.GD, 'M'],
];

const TODAY = '2026-10-02';
let db;

before(() => {
  db = createSqlite();

  const upsert = db.prepare(UPSERT_DAILY_SQL);
  const visitor = db.prepare(INSERT_VISITOR_SQL);
  for (const [day, geo, vid] of VISITS) {
    upsert.run(day, geo.country, geo.province, geo.city);
    visitor.run(day, vid);
  }
});

test('schema.sql 可以在空库上重复执行（IF NOT EXISTS 幂等）', () => {
  assert.doesNotThrow(() => db.exec(readFileSync(SCHEMA_PATH, 'utf8')));
});

test('同一「地区 × 天」重复访问只累加 pv，不新增行', () => {
  const rows = db.prepare('SELECT COUNT(*) AS c FROM daily_stats').get();
  // BJ 两天 = 2 行；SH 两天 = 2 行；US 一天 = 1 行；GD 一天 = 1 行 → 共 6 行，
  // 而不是 17 行。这正是这套表结构能把行数压到「日期 × 地区数」量级的原因。
  assert.equal(rows.c, 6);

  const bj = db.prepare(`SELECT pv FROM daily_stats WHERE day = '2026-09-30' AND province = 'BJ'`).get();
  assert.equal(bj.pv, 3);
  const sh = db.prepare(`SELECT pv FROM daily_stats WHERE day = '2026-10-01' AND province = 'SH'`).get();
  assert.equal(sh.pv, 5);
  const us = db.prepare(`SELECT pv FROM daily_stats WHERE day = '2026-09-30' AND country = 'US'`).get();
  assert.equal(us.pv, 2);
});

test('TOTALS_SQL：总 / 今日 / 本月 / 本年', () => {
  const row = db.prepare(TOTALS_SQL).get(TODAY, `${TODAY.slice(0, 7)}-01`, `${TODAY.slice(0, 4)}-01-01`);
  assert.equal(row.total, 17);
  assert.equal(row.today, 4);
  assert.equal(row.month, 11, '10 月 = 7 + 4');
  assert.equal(row.year, 17, '全部数据都在 2026 年');
});

test('TOTALS_SQL：没有任何数据时返回 0 而不是 null', () => {
  const empty = createSqlite();
  empty.exec(readFileSync(SCHEMA_PATH, 'utf8'));
  const row = empty.prepare(TOTALS_SQL).get('2026-10-02', '2026-10-01', '2026-01-01');
  assert.equal(row.total, 0);
  assert.equal(row.today, 0);
  assert.equal(row.month, 0);
  assert.equal(row.year, 0);
});

test('年月分组曲线', () => {
  assert.deepEqual(
    db.prepare(YEARLY_SQL).all().map((r) => [r.k, r.v]),
    [['2026', 17]]
  );
  assert.deepEqual(
    db.prepare(MONTHLY_SQL).all().map((r) => [r.k, r.v]),
    [
      ['2026-09', 6],
      ['2026-10', 11],
    ]
  );
});

test('DAILY_SQL：近 30 天按天曲线，起始日含在内', () => {
  assert.deepEqual(
    db.prepare(DAILY_SQL).all('2026-09-30').map((r) => [r.k, r.v]),
    [
      ['2026-09-30', 6],
      ['2026-10-01', 7],
      ['2026-10-02', 4],
    ]
  );
  // 起始日比最早数据晚一天 → 只剩两天
  assert.deepEqual(
    db.prepare(DAILY_SQL).all('2026-10-01').map((r) => [r.k, r.v]),
    [
      ['2026-10-01', 7],
      ['2026-10-02', 4],
    ]
  );
});

test('地理排行：全部时间按省/市聚合后降序', () => {
  const all = rangeOf('all', TODAY);
  const rows = db
    .prepare(geoRankSql(all.clause))
    .all()
    .map((r) => [`${r.country}:${r.province}`, r.v]);
  assert.deepEqual(rows, [
    ['CN:SH', 6],
    ['CN:BJ', 5],
    ['CN:GD', 4],
    ['US:California', 2],
  ]);
});

test('地理排行：按范围过滤时不串月', () => {
  const month = rangeOf('month', TODAY); // 2026-10
  const rows = db
    .prepare(geoRankSql(month.clause))
    .all(...month.params)
    .map((r) => [`${r.country}:${r.province}`, r.v]);
  assert.deepEqual(rows, [
    ['CN:SH', 5],
    ['CN:GD', 4],
    ['CN:BJ', 2],
  ]);
  assert.equal(
    rows.reduce((sum, [, v]) => sum + v, 0),
    11,
    '范围内地理合计必须等于同范围的 TOTALS month'
  );

  const year = rangeOf('year', TODAY);
  assert.equal(
    db
      .prepare(geoRankSql(year.clause))
      .all(...year.params)
      .reduce((sum, r) => sum + r.v, 0),
    17
  );

  const day = rangeOf('day', '2026-09-30');
  assert.deepEqual(
    db
      .prepare(geoRankSql(day.clause))
      .all(...day.params)
      .map((r) => [`${r.country}:${r.province}`, r.v]),
    [
      ['CN:BJ', 3],
      ['US:California', 2],
      ['CN:SH', 1],
    ]
  );
});

test('UV：同日同人只算一次，跨月需要换 vid', () => {
  // 09-30 共 4 个不同 vid：A B C X
  assert.equal(db.prepare(TODAY_UV_SQL).get('2026-09-30').uv, 4);
  // 10-01：X B M → 3
  assert.equal(db.prepare(TODAY_UV_SQL).get('2026-10-01').uv, 3);
  // 10-02：X M → 2
  assert.equal(db.prepare(TODAY_UV_SQL).get(TODAY).uv, 2);
  // 10 月：X B M → 3
  assert.equal(db.prepare(RANGE_UV_SQL).get('2026-10-01').uv, 3);
  // 全年：A B C X M → 5（X 在 9 月与 10 月是两个不同 vid，这正是「跨月重复计入」的口径）
  assert.equal(db.prepare(RANGE_UV_SQL).get('2026-01-01').uv, 5);
});

test('visitor_days 主键去重：同一人同一天插入多次只留一行', () => {
  const rows = db.prepare(`SELECT COUNT(*) AS c FROM visitor_days WHERE day = '2026-10-01' AND vid = 'M'`).get();
  assert.equal(rows.c, 1);
  const total = db.prepare('SELECT COUNT(*) AS c FROM visitor_days').get();
  // 17 次访问去重到 11 行（09-30 ×4、10-01 ×3、10-02 ×2 … 按上面逐日相加）
  assert.equal(total.c, 4 + 3 + 2);
});

test('索引命中：日过滤查询计划走 idx，不退化成全表扫描（本轮数据量小，只确认索引存在）', () => {
  const idx = db.prepare(`SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'visitor_days'`).all();
  assert.ok(idx.some((row) => row.name === 'idx_visitor_day'));
});
