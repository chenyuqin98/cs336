# CS336 (Spring 2026) — 自学起步

全部为当届 student 版（非 `spring2024-*` 归档）。`uv` 已装到 `~/.local/bin`。

```bash
export PATH="$HOME/.local/bin:$PATH"      # 建议写进 ~/.zshrc
cd assignment1-basics && uv sync && uv run pytest -q
```

A1 已验证：46 failed / 2 skipped，全是 `NotImplementedError` —— 这就是正确起点。

## 各作业本机（M 芯片 Mac）可做性

| 作业 | 主题 | Mac 上能做什么 |
|---|---|---|
| 1-basics | BPE + Transformer + AdamW 全手写 | **全部**。18 个 stub 在 `tests/adapters.py`，测试全 CPU。官方 low-resource 路线：MPS 上 ~5 分钟训出能说人话的 TinyStories LM |
| 2-systems | profiling / FlashAttention-2 Triton / DDP / FSDP | **一半**。DDP、optimizer sharding、FSDP 测试走 gloo CPU 能过；Triton 是 Linux only，两个 kernel 测试直接 skip |
| 3-scaling | scaling law 拟合 + 训练 API 询价 | **只有 `chinchilla_isoflops`**（数据 `data/isoflops_curves.json` 已在仓库）。其余 50/55 分要连 Stanford 内网 API + 8 位学号，做不了 |
| 4-data | Common Crawl 清洗全流水线 | **测试全部可过**（12 个 adapter 纯 CPU）。要先下 ~3GB 离线资产，且 `cs336_data/modal_utils.py` 的 `SUNET_ID = "TODO"` 会在 import 时抛错，随便填个字符串 |
| 5-alignment | GRPO / RLVR 复现 | **测试全部可过**（27MB tiny-gpt2 fixture，强制 CPU）。GSM8K 数据已在仓库。真跑实验要 OLMo-2-1B + N 卡 |
| lectures | 17 讲 | `var/traces/*.json` 是预编译产物，**不跑代码也能在 viewer 里逐行读**，含 Mac 跑不了的第 6/7 讲 |

要真训模型时，租 GPU 比配环境划算（作业里所有预算都按 B200-hour 计）。

## 坑

- 各仓库的 `CLAUDE.md` / `AGENTS.md` 是课程学术诚信约定：**要求 AI 只做助教、不写解法代码**。在这些目录里用 Claude Code 它会照办并拒绝帮你实现。
- A1 `test_train_bpe_speed` 卡 1.5 秒 —— 朴素 O(n²) 实现「正确但过不了」，得按 `<|endoftext|>` 切块多进程。
- A1 两个内存测试在 macOS 上 skip（RLIMIT_AS 不可靠），`encode_iterable` 的流式特性得自己验。
- A4 `fasttext` 无预编译 wheel，Apple Silicon 上要 Xcode CLT 现编，是最可能装不上的依赖。
- A5 快照测试对 dict key 严格相等，返回多余的 debug key 也算 fail。
- lectures README 里的 `python execute.py -m lecture_01` 是坏的，仓库没这个文件；用 `python lecture_01.py`。

## 建议顺序

A1 全做（这是整门课的地基）→ lectures 边读边补 → A2 的 CPU 部分（DDP/FSDP 手写一遍价值很高）→ A5 测试 → A4 测试 → A3 只做 isoflops 拟合。
