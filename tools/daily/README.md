# 每日资讯流水线

每天 08:00 从多个信源采集具身智能 / 世界动作模型 / 强化学习方向的进展，
在 `_daily/` 写一篇资讯条目并推送。**完整手册见 `.agents/DAILY-PIPELINE.md`。**

## 三步跑完

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
python3 tools/daily/fetch_all.py --days 7            # 五源采集
python3 tools/daily/pick_topics.py --top 12 --by-source   # 筛选，看候选
# 精读后写 _daily/YYYY-MM-DD-<slug>.md，校验，提交推送
```

## 信源

| 脚本 | 信源 | 代理 |
| --- | --- | --- |
| `fetch_all.py` | 青稞 / HuggingFace Daily Papers / arXiv / GitHub / NVIDIA Blog | HF·GitHub·NVIDIA 需代理 |
| `fetch_qingke.py` | 只抓青稞（调试用，`fetch_all` 已含） | 不需要 |

**代理 `127.0.0.1:7890`（ClashX）是硬依赖**，脚本启动会先探测，挂了直接报错。
不带代理跑：`DAILY_PROXY= python3 tools/daily/fetch_all.py`。

**知乎抓不了**：匿名访问全部 403（网页、API、问答接口都需要登录态）。
当前靠青稞正文里附的作者知乎原文链接间接覆盖。

## 常用参数

- `--days N` 采集窗口（默认 7 天）
- `--only qingke,hf` 只跑指定信源
- `--date YYYY-MM-DD` 补跑历史某天
- `--source hf,arxiv` 只看某些信源的候选
- `--dump-body` 打印摘要，方便直接阅读
- `--no-body` 跳过青稞正文抓取（快，但拿不到细节）

## 注意

- `out/` 在 `.gitignore` 里，是抓取产物，不入库
- **`pick_topics.py` 不能改名成 `select.py`** —— 会遮蔽标准库 `select`，所有网络脚本都会挂
- 筛选器只做初筛，直播预告 / 招聘 / 广告要人工丢掉
- 同一工作常被多源重复命中，合并成一条写，链接给全
- 当天没有符合主题的新内容就跳过，不凑数
