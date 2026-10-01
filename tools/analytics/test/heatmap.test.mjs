/**
 * 活动热力图的网格构造校验。
 *
 * 这块最容易错的地方是「边界语义」，所以断言都围绕边界写：
 *   - 最后一列必须含 endDay，且 endDay 之后的格子是 null（还没到）而不是 0（没人来）；
 *   - 首列之前的格子也必须是 null，不能把窗口外的空白算成零访问日；
 *   - 色阶在「只有一个非零日」这类退化输入下不能全变最浅。
 *
 * 运行：node --test tools/analytics/test/*.test.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { HEATMAP_WEEKS, addDays, buildCalendar, weekdayOf } from '../worker/src/lib.js';

const END = '2026-10-01';

/** 从网格里取某一天的格子，取不到返回 undefined。 */
function cellOf(cal, day) {
  for (const week of cal.weeks) for (const cell of week) if (cell && cell.day === day) return cell;
  return undefined;
}

function flatCells(cal) {
  return cal.weeks.flat();
}

test('addDays / weekdayOf：跨月跨年不跑偏', () => {
  assert.equal(addDays('2026-10-01', -1), '2026-09-30');
  assert.equal(addDays('2026-03-01', -1), '2026-02-28');
  assert.equal(addDays('2026-01-01', -1), '2025-12-31');
  assert.equal(addDays('2025-12-31', 1), '2026-01-01');
  // 2026-10-01 是周四
  assert.equal(weekdayOf('2026-10-01'), 4);
  assert.equal(weekdayOf('2026-09-27'), 0, '2026-09-27 是周日');
});

test('网格形状：固定 weeks 列、每列 7 行', () => {
  const cal = buildCalendar([], { endDay: END });
  assert.equal(cal.weeks.length, HEATMAP_WEEKS);
  assert.equal(HEATMAP_WEEKS, cal.weeks.length, '默认周数与常量一致');
  for (const week of cal.weeks) assert.equal(week.length, 7);

  const custom = buildCalendar([], { endDay: END, weeks: 4 });
  assert.equal(custom.weeks.length, 4);
  assert.equal(custom.startDay, addDays(END, -weekdayOf(END) - 3 * 7));
});

test('最后一列含 endDay，且其后是 null（「还没到」≠「零访问」）', () => {
  const cal = buildCalendar([], { endDay: END });
  const lastWeek = cal.weeks[cal.weeks.length - 1];
  const filled = lastWeek.filter(Boolean);

  assert.equal(filled[filled.length - 1].day, END, '最后一列的最后一个实格必须是 endDay');
  assert.equal(filled.length, weekdayOf(END) + 1, '周四 → 该列只有 5 天是已过去的');
  assert.deepEqual(lastWeek.slice(filled.length), [null, null]);

  // 全窗口只有「最后那一列的后半段」是空档，其余格子都该是实格
  const voids = flatCells(cal).filter((c) => c === null);
  assert.equal(voids.length, 6 - weekdayOf(END));
});

test('窗口起点之前的格子不会被补进来：cell 数量 = 周数 × 7 − 尾部空档', () => {
  const cal = buildCalendar([], { endDay: END });
  const cells = flatCells(cal);
  assert.equal(cells.filter(Boolean).length, HEATMAP_WEEKS * 7 - (6 - weekdayOf(END)));
  assert.equal(cells[0].day, cal.startDay);
});

test('色阶：按占单日最高值的比例分 4 档', () => {
  const rows = [
    { k: '2026-09-30', v: 8 }, // 100%  → 4
    { k: '2026-09-29', v: 6 }, // 75%   → 3
    { k: '2026-09-28', v: 4 }, // 50%   → 2
    { k: '2026-09-27', v: 2 }, // 25%   → 1
    { k: '2026-09-26', v: 1 }, // 12.5% → 1
  ];
  const cal = buildCalendar(rows, { endDay: END });

  assert.equal(cellOf(cal, '2026-09-30').level, 4);
  assert.equal(cellOf(cal, '2026-09-29').level, 3);
  assert.equal(cellOf(cal, '2026-09-28').level, 2);
  assert.equal(cellOf(cal, '2026-09-27').level, 1);
  assert.equal(cellOf(cal, '2026-09-26').level, 1);
  assert.equal(cellOf(cal, '2026-09-25').level, 0, '没数据的日子是 0 档');
  assert.equal(cellOf(cal, '2026-09-25').pv, 0);
});

test('退化输入：只有一个非零日时，那一天必须是最高档', () => {
  const cal = buildCalendar([{ k: END, v: 3 }], { endDay: END });
  assert.equal(cellOf(cal, END).level, 4, 'max 退化成 3 时不能只剩最浅色');
  assert.equal(cal.activeDays, 1);
  assert.equal(cal.max, 3);

  const single = buildCalendar([{ k: END, v: 1 }], { endDay: END });
  assert.equal(cellOf(single, END).level, 4, 'max = 1 时同样给最高档');
});

test('汇总口径自洽：合计 / 活跃天数 / 各档计数', () => {
  const rows = [
    { k: '2026-09-30', v: 5 },
    { k: '2026-09-28', v: 3 },
    { k: '2026-09-20', v: 1 },
  ];
  const cal = buildCalendar(rows, { endDay: END });

  assert.equal(cal.total, 9);
  assert.equal(cal.activeDays, 3);
  assert.equal(cal.max, 5);

  const distinct = flatCells(cal).filter(Boolean);
  assert.equal(cal.levelCounts.reduce((a, b) => a + b, 0), distinct.length, '各档计数之和 = 实格数');
  assert.equal(
    flatCells(cal).reduce((sum, c) => sum + (c ? c.pv : 0), 0),
    cal.total,
    '网格里的 pv 之和 = total'
  );
  assert.equal(cal.levelCounts[0], distinct.length - 3, '非活跃日全落在 0 档');
});

/** 把 "3月" / "2026年1月" 解析成 { year, month }；没有年份时沿用上一个标签的年份。 */
function parseLabel(label, fallbackYear) {
  const m = /^(?:(\d{4})年)?(\d{1,2})月$/.exec(label);
  assert.ok(m, `标签格式不认识：${label}`);
  return { year: m[1] ? Number(m[1]) : fallbackYear, month: Number(m[2]) };
}

test('月份标签：每个标签落在「该月第一天」所在的那一列', () => {
  const cal = buildCalendar([], { endDay: END });
  assert.ok(cal.monthLabels.length >= 12, '跨 53 周应覆盖 12 个月以上');
  assert.equal(cal.monthLabels[0].col, 0, '首列必须带标签，否则最左边缺一个月');

  for (const { col, label } of cal.monthLabels) {
    const month = String(parseLabel(label, Number(cal.startDay.slice(0, 4))).month).padStart(2, '0');
    const days = cal.weeks[col].filter(Boolean).map((c) => c.day);
    const first = days.find((d) => d.slice(8, 10) === '01');
    if (first) {
      assert.equal(first.slice(5, 7), month, `第 ${col} 列的标签「${label}」应对应列内的真实月初`);
    } else {
      // 只有首列允许没有月初：窗口从月中切进来时，标签退化成「这列第一天的月份」。
      assert.equal(col, 0, `第 ${col} 列既不含月初却仍被打了标签`);
      assert.equal(days[0].slice(5, 7), month);
    }
  }

  // 列号严格递增；相邻标签必须正好差一个月，中间不能跳月。
  const cols = cal.monthLabels.map((m) => m.col);
  assert.deepEqual(cols, [...cols].sort((a, b) => a - b));

  let prev = parseLabel(cal.monthLabels[0].label, Number(cal.startDay.slice(0, 4)));
  for (const { label } of cal.monthLabels.slice(1)) {
    const cur = parseLabel(label, prev.year);
    const gap = (cur.year - prev.year) * 12 + (cur.month - prev.month);
    assert.equal(gap, 1, `${prev.year}-${prev.month} 之后直接跳到 ${cur.year}-${cur.month}`);
    prev = cur;
  }
});

test('同名月份用年份区分：年份只标在 1 月，且不能压在下一个标签上', () => {
  const cal = buildCalendar([], { endDay: END });
  const labels = cal.monthLabels.map((m) => m.label);

  assert.equal(labels[0], '10月', '首列不带年份 —— 它离第二个标签只有 4 列，加上「2025年」会压字');
  assert.ok(labels.includes('2026年1月'), `年份分界线要标在 1 月：${labels}`);
  assert.equal(labels.filter((l) => l === '10月').length, 2, '窗口两端都是 10 月，靠中间的年份分界线区分');
  assert.equal(new Set(labels).size, labels.length - 1, '只有 10 月这一对重复');

  // 带年份的标签最长，必须放得进它到下一个标签之间的列距（最少 4 列）。
  const yearCol = cal.monthLabels.findIndex((m) => /年/.test(m.label));
  const nextCol = cal.monthLabels[yearCol + 1].col;
  assert.ok(nextCol - cal.monthLabels[yearCol].col >= 4, '4 列间距是「2026年1月」能放下的下限');
});

test('窗口跨年时标签连续覆盖每个月', () => {
  const cal = buildCalendar([], { endDay: '2026-10-15', weeks: 53 });
  assert.ok(cal.startDay < '2025-11-01', '53 周应回到上一年');

  const parsed = [];
  let year = Number(cal.startDay.slice(0, 4));
  for (const { label } of cal.monthLabels) {
    const p = parseLabel(label, year);
    year = p.year;
    parsed.push(p.month);
  }
  assert.ok(parsed.includes(12) && parsed.includes(1), `跨年窗口应同时覆盖 12 月和 1 月：${parsed}`);
});
