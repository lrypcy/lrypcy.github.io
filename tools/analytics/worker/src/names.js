/**
 * 地理代码 → 中文展示名。只影响看板渲染，不影响落库内容。
 * 未收录的国家直接显示 ISO 代码，避免维护一张 249 行的表。
 */

export const COUNTRY_NAMES = {
  CN: '中国',
  HK: '中国香港',
  MO: '中国澳门',
  TW: '中国台湾',
  US: '美国',
  JP: '日本',
  KR: '韩国',
  SG: '新加坡',
  MY: '马来西亚',
  TH: '泰国',
  VN: '越南',
  ID: '印度尼西亚',
  PH: '菲律宾',
  IN: '印度',
  PK: '巴基斯坦',
  BD: '孟加拉国',
  LK: '斯里兰卡',
  NP: '尼泊尔',
  MM: '缅甸',
  KH: '柬埔寨',
  LA: '老挝',
  MN: '蒙古',
  KZ: '哈萨克斯坦',
  UZ: '乌兹别克斯坦',
  AU: '澳大利亚',
  NZ: '新西兰',
  GB: '英国',
  IE: '爱尔兰',
  DE: '德国',
  FR: '法国',
  NL: '荷兰',
  BE: '比利时',
  LU: '卢森堡',
  CH: '瑞士',
  AT: '奥地利',
  SE: '瑞典',
  NO: '挪威',
  DK: '丹麦',
  FI: '芬兰',
  IS: '冰岛',
  IT: '意大利',
  ES: '西班牙',
  PT: '葡萄牙',
  PL: '波兰',
  CZ: '捷克',
  SK: '斯洛伐克',
  HU: '匈牙利',
  RO: '罗马尼亚',
  BG: '保加利亚',
  HR: '克罗地亚',
  SI: '斯洛文尼亚',
  RS: '塞尔维亚',
  EE: '爱沙尼亚',
  LV: '拉脱维亚',
  LT: '立陶宛',
  GR: '希腊',
  CY: '塞浦路斯',
  MT: '马耳他',
  TR: '土耳其',
  RU: '俄罗斯',
  UA: '乌克兰',
  BY: '白俄罗斯',
  CA: '加拿大',
  MX: '墨西哥',
  BR: '巴西',
  AR: '阿根廷',
  CL: '智利',
  CO: '哥伦比亚',
  PE: '秘鲁',
  ZA: '南非',
  EG: '埃及',
  NG: '尼日利亚',
  KE: '肯尼亚',
  MA: '摩洛哥',
  AE: '阿联酋',
  SA: '沙特阿拉伯',
  QA: '卡塔尔',
  KW: '科威特',
  IL: '以色列',
  IR: '伊朗',
  IQ: '伊拉克',
  XX: '未知地区',
};

/** 中国省级行政区（ISO 3166-2 短码）。Cloudflare 对 CN 返回的 regionCode 就是这套。 */
export const CN_PROVINCES = {
  BJ: '北京',
  TJ: '天津',
  HE: '河北',
  SX: '山西',
  NM: '内蒙古',
  LN: '辽宁',
  JL: '吉林',
  HL: '黑龙江',
  SH: '上海',
  JS: '江苏',
  ZJ: '浙江',
  AH: '安徽',
  FJ: '福建',
  JX: '江西',
  SD: '山东',
  HA: '河南',
  HB: '湖北',
  HN: '湖南',
  GD: '广东',
  GX: '广西',
  HI: '海南',
  CQ: '重庆',
  SC: '四川',
  GZ: '贵州',
  YN: '云南',
  XZ: '西藏',
  SN: '陕西',
  GS: '甘肃',
  QH: '青海',
  NX: '宁夏',
  XJ: '新疆',
  TW: '中国台湾',
  HK: '中国香港',
  MO: '中国澳门',
};

export function countryLabel(code) {
  const c = String(code || '').toUpperCase();
  if (!c) return '未知地区';
  return COUNTRY_NAMES[c] || c;
}

export function provinceLabel(country, code) {
  const cc = String(country || '').toUpperCase();
  const p = String(code || '').trim();
  if (!p) return '';
  if (cc === 'CN') return CN_PROVINCES[p.toUpperCase()] || p.toUpperCase();
  return p;
}

/** 一行地理记录的可读标题，如「中国 · 广东 · Shenzhen」。 */
export function geoLabel(row) {
  const country = countryLabel(row.country);
  const province = provinceLabel(row.country, row.province);
  const parts = [country];
  if (province && province !== country) parts.push(province);
  if (row.city) parts.push(row.city);
  return parts.join(' · ');
}
