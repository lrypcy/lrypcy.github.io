/**
 * 省级方块地图（tile grid map）：布局 + 聚合。纯函数，node 可直接单测。
 *
 * 【为什么不是一张真地图】
 * 1. 合规：地图渲染只能用腾讯 / 高德 / 百度 / 天地图这类合规来源；而本看板的流量是
 *    全球的，渲染海外地理区域属于「需要自备 key」的场景。看板又是 Cloudflare Worker
 *    直接吐 HTML、以 iframe 嵌在 GitHub Pages 上，WorkBuddy 的 key 代理在这里不生效，
 *    等于要博客作者自己申请 key 并承担 Referer 白名单维护。
 * 2. 依赖：地图 SDK 要在国内可访问，必须从对方 CDN 加载几百 KB，还会把看板的
 *    「零外部资源」这条底线破掉（见 dashboard.js 头部注释）。
 * 3. 数据：落库的只有 country / province / city 三个字段（见 lib.js 的 UPSERT_DAILY_SQL），
 *    没有经纬度，真地图上落点需要额外采集或维护一份坐标表。
 *
 * 所以这里画**方块地图**：每个省级行政区一个等大方块，按近似相对位置摆放。
 * 方块地图是数据可视化的通行做法（美国人口普查、各家媒体的美国/中国方块图都这么画），
 * 它只表达「相对位置 + 量级」，**不描绘任何边界线**，因此这一页不涉及任何边界画法。
 */

import { levelByRatio } from './lib.js';

/** 34 个省级行政区在 7 列 × 8 行网格中的位置（[列, 行]，从 0 起；北在上、西在左）。 */
export const TILE_LAYOUT = {
  HL: [6, 0],

  XJ: [0, 1], NM: [3, 1], JL: [6, 1],

  GS: [2, 2], NX: [3, 2], BJ: [4, 2], LN: [5, 2],

  XZ: [0, 3], QH: [1, 3], SN: [2, 3], SX: [3, 3], HE: [4, 3], TJ: [5, 3],

  SC: [1, 4], CQ: [2, 4], HA: [3, 4], SD: [4, 4], JS: [5, 4],

  YN: [1, 5], GZ: [2, 5], HB: [3, 5], AH: [4, 5], ZJ: [5, 5], SH: [6, 5],

  GX: [2, 6], HN: [3, 6], JX: [4, 6], FJ: [5, 6], TW: [6, 6],

  GD: [2, 7], HI: [3, 7], HK: [4, 7], MO: [5, 7],
};

/** 方块里的短名：方块宽度只有几十像素，`CN_PROVINCES` 的正式名（如「中国香港」）放不下。 */
export const TILE_SHORT = {
  BJ: '北京', TJ: '天津', HE: '河北', SX: '山西', NM: '内蒙', LN: '辽宁', JL: '吉林',
  HL: '黑龙江', SH: '上海', JS: '江苏', ZJ: '浙江', AH: '安徽', FJ: '福建', JX: '江西',
  SD: '山东', HA: '河南', HB: '湖北', HN: '湖南', GD: '广东', GX: '广西', HI: '海南',
  CQ: '重庆', SC: '四川', GZ: '贵州', YN: '云南', XZ: '西藏', SN: '陕西', GS: '甘肃',
  QH: '青海', NX: '宁夏', XJ: '新疆', TW: '台湾', HK: '香港', MO: '澳门',
};

export const TILE_COLS = 7;
export const TILE_ROWS = 8;

/**
 * 港澳台在 Cloudflare 的 country 字段里各自是一个代码（HK / MO / TW），不带 province，
 * 但它们与大陆省份同属一张图，所以整块归到一个方块上。
 */
const TERRITORY_TILES = new Set(['HK', 'MO', 'TW']);

/**
 * 一行地理记录落到哪个方块。返回 null 表示这张图放不下它。
 *
 * 注意 `province` 并不总是 ISO 短码：`normalizeGeo()` 取的是 `regionCode || region`，
 * 当 Cloudflare 只给了英文省名（如 'Guangdong'）时大写后是 'GUANGDONG'，匹配不上任何方块。
 * 这类流量不能悄悄丢掉，由调用方汇总成「未定位到省份」再展示。
 */
export function tileKeyOf(country, province) {
  const cc = String(country || '').toUpperCase();
  if (TERRITORY_TILES.has(cc)) return cc;
  if (cc !== 'CN') return null;
  const code = String(province || '').trim().toUpperCase();
  return Object.prototype.hasOwnProperty.call(TILE_LAYOUT, code) ? code : null;
}

/** 这张图负责统计哪些流量：大陆（按省）+ 港澳台（整块）。其它国家交给「国家 / 地区排行」。 */
export function inMapScope(country) {
  const cc = String(country || '').toUpperCase();
  return cc === 'CN' || TERRITORY_TILES.has(cc);
}

/**
 * 把 `geos`（[{country, province, city, v}]）聚合成方块网格。
 *
 * 返回的 `tiles` 永远包含全部 34 个方块（值为 0 的也在），这样网格位置恒定、
 * 不会因为某天没有某个省的数据就整体位移。
 */
export function buildProvinceTiles(rows) {
  const values = new Map();
  let mapped = 0;
  let unmapped = 0;

  for (const row of rows || []) {
    const v = Number(row && row.v) || 0;
    if (v <= 0) continue;
    const key = tileKeyOf(row && row.country, row && row.province);
    if (key) {
      values.set(key, (values.get(key) || 0) + v);
      mapped += v;
    } else if (inMapScope(row && row.country)) {
      unmapped += v; // 中国大陆但没定位到省 —— 只汇总，不落到具体方块
    }
    // 其它国家不计入这张图，也不计入 unmapped（它们由排行卡片负责）
  }

  const max = values.size ? Math.max(...values.values()) : 0;

  const tiles = Object.entries(TILE_LAYOUT).map(([code, [col, row]]) => {
    const v = values.get(code) || 0;
    // 色阶复用热力图那一套，保证同一页里「深浅」含义一致
    return { code, col, row, v, level: levelByRatio(v, max) };
  });

  return {
    tiles,
    cols: TILE_COLS,
    rows: TILE_ROWS,
    max,
    mapped,
    unmapped,
    activeTiles: tiles.filter((t) => t.v > 0).length,
    totalTiles: tiles.length,
  };
}
