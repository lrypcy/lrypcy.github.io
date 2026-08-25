# PROGRESS 进度追踪

| # | 文件 | 状态 | 备注 |
|---|------|------|------|
| 0 | RESEARCH_PLAN.md | ✅ Done | 2026-08-25 |
| 1 | _posts/2026-08-25-embodied-ai-00-overview.md | ✅ Done | 总览篇，238 行 |
| 2 | _posts/2026-08-25-embodied-ai-01-control-foundations.md | ✅ Done | 控制地基（LQR/MPC/WBC 推导），254 行 |
| 3 | _posts/2026-08-26-embodied-ai-02-rl-il-sim2real.md | ✅ Done | RL/IL/Sim2Real |
| 4 | _posts/2026-08-27-embodied-ai-03-perception.md | ✅ Done | 感知 |
| 5 | _posts/2026-08-28-embodied-ai-04-manipulation.md | ✅ Done | 操作（Diffusion Policy/ACT 推导） |
| 6 | _posts/2026-08-29-embodied-ai-05-vla-foundation-models.md | ✅ Done | VLA（流匹配推导） |
| 7 | _posts/2026-08-30-embodied-ai-06-navigation-planning.md | ✅ Done | 导航与规划 |
| 8 | _posts/2026-08-31-embodied-ai-07-data-sim-benchmark.md | ✅ Done | 数据仿真评测 |
| 9 | _posts/2026-09-01-embodied-ai-08-industry-roadmap.md | ✅ Done | 产业与路线图 |
| 10 | README.md（阅读指南） | ✅ Done | Phase 4 |

## Phase 3 自审记录

- [x] 数学一致性：变量映射表 Shape 与代码张量维度逐一核对
- [x] Mermaid 渲染安全：无 `\n` 换行、节点文本均含 `<br>` 且无双引号冲突（grep 验证通过）
- [x] 链接有效性：60+ arXiv ID 经 export.arxiv.org API 核实；官方站点 curl 探测；不可达链接全部标注"未本地验证"
  - 修正记录：GR-1(2310.16825→2312.13139)、BC-Z(2206.00198→2202.02005)、BridgeV2(2303.01566→2308.12952)、Rudin(2109.11962→2109.11978)、Tobin DR(1703.06903→1703.06907)、3DGS(2308.14737→2308.04079)、ManiSkill3(2410.00414→2410.00425)、Meta-World(1910.11625→1910.10897)
- [x] 中文语境补充：知乎/B站/深蓝学院等资源以检索入口形式给出（正文需登录故不引具体帖）
- [x] Lab Exercises：每篇 2–3 个可执行实验
- [x] 已知限制：本地 Jekyll 因系统 Ruby 缺 bundler 无法构建（环境既有问题）；IEEE Xplore 链接返回 202 反爬响应，视为规范地址但内容未逐一核对

## 未尽事项与后续建议

- 公司动态/价格等时效信息发布前建议二次核实
- 若需配图，可在 assets/images/ 下补系列插图并在各篇引用
- 后续可扩展篇目：灵巧手专题、世界模型专题、具身智能安全认证
