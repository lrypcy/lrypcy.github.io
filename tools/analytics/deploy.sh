#!/usr/bin/env bash
#
# 一键部署 pcy-analytics 到 Cloudflare Workers + D1。
#
#   ./deploy.sh                 首次部署（创建 D1、建表、写入密钥、部署）
#   ./deploy.sh --rotate-token  重置看板口令
#   ./deploy.sh --rotate-salt   重置访客哈希盐（会重置当日去重口径，谨慎）
#
# 前置：先跑 `npx --yes wrangler@4.145.0 login` 完成一次浏览器授权。
# 幂等：重复执行不会重复创建数据库、不会覆盖已有密钥，只做部署。

set -euo pipefail

cd "$(dirname "$0")"

DB_NAME="pcy-analytics"
WRANGLER=(npx --yes wrangler@4.145.0)
TOKEN_FILE=".secrets.local"

ROTATE_TOKEN=false
ROTATE_SALT=false
for arg in "$@"; do
  case "$arg" in
    --rotate-token) ROTATE_TOKEN=true ;;
    --rotate-salt) ROTATE_SALT=true ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

# 支持用 API Token 免浏览器授权：把 CLOUDFLARE_API_TOKEN 写进 .env.local（.gitignore 已忽略）
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.local
  set +a
  echo "已从 .env.local 加载凭据"
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\033[31m失败：%s\033[0m\n' "$1" >&2; exit 1; }

# ---------- 0. 登录检查 ----------
step "检查 Cloudflare 登录状态"
if ! "${WRANGLER[@]}" whoami >/dev/null 2>&1; then
  fail "尚未登录。请先执行：
  cd $(pwd)
  npx --yes wrangler@4.145.0 login"
fi
"${WRANGLER[@]}" whoami 2>/dev/null | grep -iE "associated|account|email" || true

# ---------- 1. 复用或创建 D1 ----------
# 先查再建：对已存在的库调用 `d1 create` 会长时间等待（上一版脚本就卡在这里）。
step "准备 D1 数据库 $DB_NAME"
lookup_db_id() {
  "${WRANGLER[@]}" d1 list --json 2>/dev/null | python3 -c "import json,sys
try:
    data = json.load(sys.stdin)
except Exception:
    data = []
rows = data if isinstance(data, list) else data.get('result', [])
print(next((r.get('uuid') or r.get('id') or '' for r in rows if r.get('name') == '$DB_NAME'), ''))"
}

DB_ID=$(lookup_db_id)
if [ -n "$DB_ID" ]; then
  echo "数据库已存在，复用 $DB_ID"
else
  CREATE_OUT=$("${WRANGLER[@]}" d1 create "$DB_NAME" 2>&1 || true)
  DB_ID=$(printf '%s' "$CREATE_OUT" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
  [ -n "$DB_ID" ] || DB_ID=$(lookup_db_id)
fi

[ -n "$DB_ID" ] || fail "拿不到 database_id，请手动跑一次 'wrangler d1 create $DB_NAME' 看报错"
echo "database_id = $DB_ID"

# ---------- 2. 写回 wrangler.toml ----------
step "把 database_id 写入 worker/wrangler.toml"
python3 - "$DB_ID" <<'PY'
import pathlib, re, sys

db_id = sys.argv[1]
path = pathlib.Path('worker/wrangler.toml')
text = path.read_text(encoding='utf-8')
new, count = re.subn(r'database_id\s*=\s*"[^"]*"', f'database_id = "{db_id}"', text)
if count != 1:
    sys.exit(f'wrangler.toml 里 database_id 出现 {count} 次，预期 1 次')
path.write_text(new, encoding='utf-8')
print('已更新')
PY

# ---------- 3. 建表 ----------
step "建表（IF NOT EXISTS，可重复执行）"
(cd worker && "${WRANGLER[@]}" d1 execute "$DB_NAME" --remote --file=./schema.sql)

# ---------- 4. 部署 Worker ----------
# 必须先于 secret：secret 挂在 Worker 上，Worker 不存在时 `secret put` 会报 not found。
step "部署 Worker"
(cd worker && "${WRANGLER[@]}" deploy) | tee /tmp/pcy-deploy.out

if grep -q "register a workers.dev subdomain" /tmp/pcy-deploy.out; then
  fail "账号还没有 workers.dev 子域名，部署无法完成。两种做法：
  1) 打开 https://dash.cloudflare.com/?to=/:account/workers/workers-and-pages
     首次进入 Workers & Pages 会自动引导注册一个子域名；
  2) 或用 API 直接指定名字：
     curl -X PUT -H \"Authorization: Bearer <token>\" -H 'Content-Type: application/json' \\
          --data '{\"subdomain\":\"你的名字\"}' \\
          https://api.cloudflare.com/client/v4/accounts/<account_id>/workers/subdomain
  完成后重跑本脚本。"
fi

ENDPOINT=$(grep -oE 'https://[A-Za-z0-9._-]+\.workers\.dev' /tmp/pcy-deploy.out | head -1 || true)
[ -n "$ENDPOINT" ] || fail "部署输出里没找到 workers.dev 地址，请检查上面的日志"
echo "线上地址 $ENDPOINT"

# ---------- 5. 密钥 ----------
# secret 相关命令必须带着 wrangler.toml 所在目录一起跑，否则 wrangler 找不到 Worker name，
# 会报 "Required Worker name missing"。
step "配置密钥"
EXISTING_SECRETS=$( (cd worker && "${WRANGLER[@]}" secret list 2>/dev/null) || echo '[]')

has_secret() { printf '%s' "$EXISTING_SECRETS" | grep -q "\"$1\""; }

SALT=""
if has_secret SALT_SECRET && [ "$ROTATE_SALT" = false ]; then
  echo "SALT_SECRET 已存在，保持不动（换盐会让当日独立访客口径重置）"
else
  SALT=$(openssl rand -hex 32)
  (cd worker && printf '%s' "$SALT" | "${WRANGLER[@]}" secret put SALT_SECRET)
  echo "SALT_SECRET 已写入"
fi

TOKEN=""
if has_secret DASH_TOKEN && [ "$ROTATE_TOKEN" = false ]; then
  echo "DASH_TOKEN 已存在，保持不动（忘了就加 --rotate-token 重置）"
else
  TOKEN=$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-24)
  (cd worker && printf '%s' "$TOKEN" | "${WRANGLER[@]}" secret put DASH_TOKEN)
  echo "DASH_TOKEN 已写入"
fi

# 访问地图的腾讯地图 key（可选）。不配就不渲染「访问地图」卡片，其余看板照常。
# 首次配置：TMAP_KEY=你的key ./deploy.sh；已存在时不会覆盖。
if [ -n "${TMAP_KEY:-}" ]; then
  (cd worker && printf '%s' "$TMAP_KEY" | "${WRANGLER[@]}" secret put TMAP_KEY)
  echo "TMAP_KEY 已写入（访问地图会显示）"
elif has_secret TMAP_KEY; then
  echo "TMAP_KEY 已存在，保持不动"
else
  echo "未配置 TMAP_KEY —— 「访问地图」卡片不会渲染（怎么申请见 README 3.3）"
fi

# ---------- 6. 保存口令到本地（已 gitignore） ----------
if [ -n "$TOKEN" ] || [ -n "$SALT" ]; then
  {
    echo "# pcy-analytics 本地备忘，不要提交（.gitignore 已忽略）"
    if [ -n "$TOKEN" ]; then echo "DASH_TOKEN=$TOKEN"; fi
    if [ -n "$SALT" ]; then echo "SALT_SECRET=$SALT"; fi
  } >> "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE" 2>/dev/null || true
fi

step "完成"
echo "采集端点  $ENDPOINT/c"
echo "计数端点  $ENDPOINT/total"
echo "看板      $ENDPOINT/dash?token=<你的 DASH_TOKEN>"
if [ -n "$TOKEN" ]; then
  echo
  echo "本次生成的口令（也存进了 $TOKEN_FILE，请自己保存好）："
  echo "  $TOKEN"
fi
echo
echo "下一步：把 _config.yml 的 analytics 改成"
echo "  enabled: true"
echo "  endpoint: \"$ENDPOINT\""
