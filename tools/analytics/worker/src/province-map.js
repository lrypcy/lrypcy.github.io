/**
 * 省级访问分布：聚合 + 落点。纯函数，node 可直接单测。
 *
 * 【为什么这里只有中心点，没有边界多边形】
 * 地图本身交给腾讯地图 GL JS（合规白名单内的来源）来画，本模块只负责两件事：
 *   1. 把 `geos`（[{country, province, city, v}]）按省级行政区聚合；
 *   2. 给每个省级行政区一个落点，供地图打气泡。
 * 中心点取自阿里云 DataV.GeoAtlas 行政区数据的 `center` 字段（与腾讯地图同为 GCJ-02），
 * 所以页面里不需要再挂一份几百 KB 的边界数据。
 *
 * 边界与领土由腾讯地图底图负责：底图按国家标准绘制，含南海诸岛与九段线，
 * 港澳台与大陆同图 —— 这正是「不自己画边界」的理由，自绘边界才是合规风险源。
 */

import { levelByRatio } from './lib.js';

/** 省级行政区短名（地图气泡标签用；正式名如「内蒙古自治区」太长，会盖住底图）。 */
export const PROVINCE_SHORT = {
  BJ: '北京', TJ: '天津', HE: '河北', SX: '山西', NM: '内蒙', LN: '辽宁', JL: '吉林',
  HL: '黑龙江', SH: '上海', JS: '江苏', ZJ: '浙江', AH: '安徽', FJ: '福建', JX: '江西',
  SD: '山东', HA: '河南', HB: '湖北', HN: '湖南', GD: '广东', GX: '广西', HI: '海南',
  CQ: '重庆', SC: '四川', GZ: '贵州', YN: '云南', XZ: '西藏', SN: '陕西', GS: '甘肃',
  QH: '青海', NX: '宁夏', XJ: '新疆', TW: '台湾', HK: '香港', MO: '澳门',
};

/**
 * 省级行政区中心点 `[纬度, 经度]`，坐标系 GCJ-02（与腾讯地图一致）。
 * 来源：阿里云 DataV.GeoAtlas `areas_v3/bound/100000_full.json` 的 `properties.center`。
 */
export const PROVINCE_CENTER = {
  BJ: [39.904989, 116.405285],
  TJ: [39.125596, 117.190182],
  HE: [38.045474, 114.502461],
  SX: [37.857014, 112.549248],
  NM: [40.818311, 111.670801],
  LN: [41.796767, 123.429096],
  JL: [43.886841, 125.3245],
  HL: [45.756967, 126.642464],
  SH: [31.231706, 121.472644],
  JS: [32.041544, 118.767413],
  ZJ: [30.287459, 120.153576],
  AH: [31.86119, 117.283042],
  FJ: [26.075302, 119.306239],
  JX: [28.676493, 115.892151],
  SD: [36.675807, 117.000923],
  HA: [34.757975, 113.665412],
  HB: [30.584355, 114.298572],
  HN: [28.19409, 112.982279],
  GD: [23.125178, 113.280637],
  GX: [22.82402, 108.320004],
  HI: [20.031971, 110.33119],
  CQ: [29.533155, 106.504962],
  SC: [30.659462, 104.065735],
  GZ: [26.578343, 106.713478],
  YN: [25.040609, 102.712251],
  XZ: [29.660361, 91.132212],
  SN: [34.263161, 108.948024],
  GS: [36.058039, 103.823557],
  QH: [36.623178, 101.778916],
  NX: [38.46637, 106.278179],
  XJ: [43.792818, 87.617733],
  TW: [25.044332, 121.509062],
  HK: [22.320048, 114.173355],
  MO: [22.198951, 113.54909],
};

/**
 * 港澳台在 Cloudflare 的 `country` 字段里各自是一个代码（HK / MO / TW），不带 `province`，
 * 但它们与大陆省份同属一张图，所以整块归到一个落点上。
 */
const TERRITORY_CODES = new Set(['HK', 'MO', 'TW']);

/**
 * 一行地理记录落到哪个省级行政区。返回 null 表示这张图放不下它。
 *
 * 注意 `province` 并不总是 ISO 短码：`normalizeGeo()` 取的是 `regionCode || region`，
 * 当 Cloudflare 只给了英文省名（如 'Guangdong'）时大写后是 'GUANGDONG'，匹配不上任何省。
 * 这类流量不能悄悄丢掉，由调用方汇总成「未定位到省份」再展示。
 */
export function provinceCodeOf(country, province) {
  const cc = String(country || '').toUpperCase();
  if (TERRITORY_CODES.has(cc)) return cc;
  if (cc !== 'CN') return null;
  const code = String(province || '').trim().toUpperCase();
  return Object.prototype.hasOwnProperty.call(PROVINCE_CENTER, code) ? code : null;
}

/** 这张图负责统计哪些流量：大陆（按省）+ 港澳台（整块）。其它国家交给「国家 / 地区排行」。 */
export function inMapScope(country) {
  const cc = String(country || '').toUpperCase();
  return cc === 'CN' || TERRITORY_CODES.has(cc);
}

/**
 * 把 `geos` 聚合成「有访问的省份 + 落点」，供地图打点。
 *
 * `items` 只含有访问的省份（值为 0 的不打气泡，否则地图上会飘满空心圆），
 * 按访问量降序；`level` 是 1..4 的档位，与热力图共用同一套 `levelByRatio` 分档。
 */
export function buildProvinceStats(rows) {
  const values = new Map();
  let mapped = 0;
  let unmapped = 0;

  for (const row of rows || []) {
    const v = Number(row && row.v) || 0;
    if (v <= 0) continue;
    const code = provinceCodeOf(row && row.country, row && row.province);
    if (code) {
      values.set(code, (values.get(code) || 0) + v);
      mapped += v;
    } else if (inMapScope(row && row.country)) {
      unmapped += v; // 中国大陆但没定位到省 —— 只汇总，不落点
    }
    // 其它国家不计入这张图，也不计入 unmapped（它们由排行卡片负责）
  }

  const max = values.size ? Math.max(...values.values()) : 0;
  const items = [...values.entries()]
    .map(([code, v]) => {
      const [lat, lng] = PROVINCE_CENTER[code];
      return { code, name: PROVINCE_SHORT[code], v, lat, lng, level: levelByRatio(v, max) };
    })
    .sort((a, b) => b.v - a.v);

  return {
    items,
    max,
    mapped,
    unmapped,
    activeCount: items.length,
    totalCount: Object.keys(PROVINCE_CENTER).length,
  };
}
