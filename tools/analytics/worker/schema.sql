-- pcy-analytics 表结构（D1 / SQLite）
-- 执行：wrangler d1 execute pcy-analytics --remote --file=./schema.sql
--
-- 设计取舍：
--   1. 不做文章级统计，所以一次访问只在聚合表里 +1，行数由「日期 × 地区组合数」决定，
--      而不是由访问量决定 —— 十万次访问也只占用几千行。
--   2. 不存 IP、不存 UA、不写 cookie。访客标识 vid 是「盐 + 月份 + IP + UA」的 SHA-256 截断值，
--      盐按月轮换，所以跨月无法把同一个人关联起来。

-- 按「地区 × 天」聚合的访问次数（PV）
CREATE TABLE IF NOT EXISTS daily_stats (
  day      TEXT    NOT NULL,          -- 站点时区（UTC+8）下的 YYYY-MM-DD
  country  TEXT    NOT NULL DEFAULT '', -- ISO 3166-1 alpha-2，'XX' 表示无法定位
  province TEXT    NOT NULL DEFAULT '', -- 中国为 ISO 3166-2 短码（GD/SH…），其他为 Cloudflare region
  city     TEXT    NOT NULL DEFAULT '', -- Cloudflare 给出的城市（中国大陆质量一般，可能为空）
  pv       INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, country, province, city)
) WITHOUT ROWID;

-- 按天去重的访客（UV）。vid 在同一个自然月内稳定，跨月变化。
CREATE TABLE IF NOT EXISTS visitor_days (
  day TEXT NOT NULL,
  vid TEXT NOT NULL,
  PRIMARY KEY (day, vid)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_visitor_day ON visitor_days (day);
