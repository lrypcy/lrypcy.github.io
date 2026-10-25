# 每日资讯流水线

每天 08:00 从青稞社区采集具身智能 / 世界动作模型 / 强化学习方向的进展，
在 `_daily/` 写一篇资讯条目并推送。**完整手册见 `.agents/DAILY-PIPELINE.md`。**

## 三步跑完

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
python3 tools/daily/fetch_qingke.py --size 40 --days 7   # 采集
python3 tools/daily/select.py --top 8                    # 筛选，看候选清单
# 精读后写 _daily/YYYY-MM-DD-<slug>.md，校验，提交推送
```

## 两个脚本

| 脚本 | 干什么 |
| --- | --- |
| `fetch_qingke.py` | 抓青稞列表 API（默认序）+ 每篇正文（`/blog/<slug>` 的 `<article>`），RSS 兜底。输出到 `out/raw-<date>.json` |
| `select.py` | 按关键词权重给标题+正文前 3000 字打分，输出候选到 `out/brief-<date>.json`。**只做初筛，终判靠人** |

常用参数：
- `--days N` 采集窗口（默认 7 天）
- `--date YYYY-MM-DD` 补跑历史某天
- `--dump-body` 打印正文摘要，方便直接阅读

## 注意

- `out/` 在 `.gitignore` 里，是抓取产物（含正文全文），不入库
- 青稞的 Halo 列表 API **带 `sort` / `fieldSelector` 参数会返回 500**，脚本已规避
- 筛选器会误判，直播预告 / 招聘 / 广告要人工丢掉；分数低但确实是重磅的手动捞回
- 当天没有符合主题的新内容就跳过，不凑数
