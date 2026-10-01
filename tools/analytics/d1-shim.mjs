/**
 * D1 / Cache API 的本地替身。
 * 单元测试与本地预览服务共用同一份实现，避免两边行为漂移。
 */

import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
export const SCHEMA_PATH = join(here, 'worker', 'schema.sql');

const isSelect = (sql) => /^\s*(select|with)\b/i.test(sql);

/** 把 node:sqlite 包装成 D1 的 prepare/bind/batch 接口。 */
export function createD1(sqlite) {
  const wrap = (sql, params) => ({
    async first() {
      if (!isSelect(sql)) {
        sqlite.prepare(sql).run(...params);
        return null;
      }
      return sqlite.prepare(sql).get(...params) ?? null;
    },
    async run() {
      return { success: true, meta: sqlite.prepare(sql).run(...params) };
    },
    async all() {
      if (!isSelect(sql)) {
        sqlite.prepare(sql).run(...params);
        return { results: [] };
      }
      return { results: sqlite.prepare(sql).all(...params) };
    },
  });

  return {
    prepare(sql) {
      return { bind: (...params) => wrap(sql, params), ...wrap(sql, []) };
    },
    async batch(statements) {
      const out = [];
      for (const stmt of statements) out.push(await stmt.all());
      return out;
    },
  };
}

/** 内存版 SQLite，建表后返回。dbPath 传文件路径即可持久化到本地磁盘。 */
export function createSqlite(dbPath = ':memory:') {
  const sqlite = new DatabaseSync(dbPath);
  sqlite.exec(readFileSync(SCHEMA_PATH, 'utf8'));
  return sqlite;
}

/** 用 Map 冒充 Cache API（Cloudflare 的 caches.default）。 */
export function installCacheMock() {
  const store = new Map();
  globalThis.caches = {
    default: {
      async match(req) {
        const hit = store.get(req.url);
        return hit ? hit.clone() : null;
      },
      async put(req, res) {
        store.set(req.url, res);
      },
    },
  };
  return store;
}
