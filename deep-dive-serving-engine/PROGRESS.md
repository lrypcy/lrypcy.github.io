# 进度：SGLang vs vLLM deep dive

## 基准（已锁定，勿动）

| | tag | commit | 日期 |
|:---|:---|:---|:---|
| vLLM | `v0.31.0` | `db9527a46873454610df6dbedf79a36d6bf1a7f6` | 2026-10-05 |
| SGLang | `v0.5.21` | `e00930c5489053f26d86b179cee0d087f846acbb` | 2026-10-02 |

复现：`git -C <repo> rev-list -n1 <tag>`。
读文件用 `git show "<sha>:<path>"` —— **引号必须加**（zsh 下 `$C:path` 展开会静默污染输出，
这个坑踩过一次，把 1641 行的 `cache_aware.rs` 误读成 45 行）。

## 已完成

- [x] 锁定两个 release tag 的 commit
- [x] 源码级取证：调度器 / KV 缓存 / 进程模型
- [x] 源码级取证：功能矩阵（模型/量化/后端/LoRA/PD/结构化输出/API/硬件/指标）
- [x] 公开benchmark 交叉验证与可信度分级
- [x] `README.md`
- [x] `01-architecture-divergence.md`
- [x] `02-feature-matrix.md`
- [x] `08-benchmark-crosscheck.md`
- [x] `10-selection-playbook.md`
- [x] `_config.yml` 的 `exclude` 加入 `deep-dive-serving-engine`
- [x] `python3 tools/check_liquid.py` 全量通过（211 文件）

## 待写（计划中有，当前为空文件占位）

- [ ] `03-model-hardware-support.md` —— 模型与硬件覆盖细节
      （内容已分散在 02 §1/§7/§11，若要单独成篇需补充：vLLM `_PREVIOUSLY_SUPPORTED_MODELS`
      的 50 条移除记录、各国产 NPU 的实际落地状态、sglang-jax 内部实现）
- [ ] `04-scheduling-and-cache.md` —— 调度器与 KV 缓存机制
      （与 01 §1/§2 重复，建议**不单独成篇**，需要时把 01 拆细或加实验）
- [ ] `05-speculative-decoding.md` —— 投机解码
      （与 02 §2 重复，同上）
- [ ] `06-pd-disaggregation.md` —— PD 分离
      （与 02 §6 重复，同上）
- [ ] `07-quantization-and-kernels.md` —— 量化与 kernel
      （与 02 §7/§3 重复，同上）
- [ ] `09-production-and-ecosystem.md` —— 生产运维与社区
      （**未做**，需要：GitHub API 客观统计 star/contributor、CVE 记录核实、
      K8s/Helm 成熟度、vLLM 商业化动向）

> **判断**：03/04/05/06/07 与已完成的 02 高度重叠，强行拆开会造成大量重复。
> 建议**只补 09**（这是唯一没有内容覆盖的），其余按需扩写 02 的对应小节。
> 待用户确认。

## 可做的实验（本机无 GPU，但这两件可跑）

1. **前缀命中率分析器**（最有价值）
   输入真实请求日志 → tokenize → 统计两两公共前缀 token 数 → 估算命中率 →
   直接映射到「选谁」。成本比性能 benchmark 低两个数量级。
2. **迁移兼容性检查器**
   给一组 OpenAI API 请求 → 标出哪些参数在两边语义不同
   （已知：`enable_prefix_caching` / `disable_radix_cache` 一正一反；
   `async_scheduling` / `disable_overlap_schedule` 一正一反；
   structured output 的 `disable_any_whitespace` 支持后端集合不同）

按项目约定，实验脚本应放 `ipynbs` 仓库的 `experiments/serving-engine/`，不写进本仓库 `tools/`。

## 本文推翻的六个流传说法（已核源码/原文）

1. 「+29%」是 2024 年数字（v0.2.3 vs 0.11.0），且是同一基准三次转载
2. IoT DT 的 5450 属 Triton；原文 vLLM 5100 / SGLang 5300 / Triton 5450
3. SGLang 默认调度是 `fcfs` 不是 `lpm`
4. TokenSpeed MLA 两边都有
5. vLLM 有 `return_sampling_mask`（18 文件完整链路）
6. SGLang 的 Rust router 在 v0.5.21 叫 `sgl-model-gateway/`，不是 `sgl-router`

## 下次动笔前要记得

- 写性能数字前必读 `08-benchmark-crosscheck.md` 的可信度分级表
- 「vLLM 支持某模型」必须带版本号（两边 breaking change 都很频繁）
- 改 `_config.yml` 的 `exclude` 后要重跑 `check_liquid.py`