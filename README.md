# CS336 自学记录

记录 Stanford CS336「Language Modeling from Scratch」的学习过程，包括作业实现、测试和实验。课程脚手架与个人实现保存在同一个 Git 仓库中，各目录独立管理 Python 环境。

## 目录

| 目录 | 内容 |
| --- | --- |
| [`assignment1-basics/`](assignment1-basics/) | BPE 分词器、Transformer 语言模型、优化器与训练基础 |
| `assignment2-systems/` | 性能分析、GPU kernel 与分布式训练 |
| `assignment3-scaling/` | Scaling laws 与计算预算分析 |
| `assignment4-data/` | 训练数据清洗、过滤与去重 |
| `assignment5-alignment/` | 语言模型对齐与强化学习 |
| `lectures/` | 课程讲义与示例代码 |

Assignment 1–5 统一由根目录的 Git 仓库管理。`lectures/` 目前仅保存在本地；当前没有 Assignment 6 目录。

## 开始学习

使用 `uv` 管理依赖。在对应作业目录中安装环境并运行测试：

```bash
cd assignment1-basics
uv sync
uv run pytest -q
```

只检查 Transformer 相关测试：

```bash
uv run pytest tests/test_model.py -q
```

每份作业的具体要求、环境配置和数据准备步骤见该目录的 README 与讲义 PDF。各目录使用的课程版本以对应讲义为准。

## Assignment 1 进度

已有基础模块实现位于 [`cs336_basics/my_torch/`](assignment1-basics/cs336_basics/my_torch/)，包括线性层、Embedding、RMSNorm、SiLU / SwiGLU、Softmax、RoPE、注意力与 Transformer。

[`tests/adapters.py`](assignment1-basics/tests/adapters.py) 负责将实现接入课程测试。BPE 训练与流式分词、数据采样、交叉熵、梯度裁剪、AdamW、学习率调度和 checkpoint 接口均已实现。训练、续训、生成与消融实验入口见 [Assignment 1 使用说明](assignment1-basics/README.md#本地实现与训练入口)。测试和短跑实验结果见 [实验记录](assignment1-basics/EXPERIMENTS.md)；讲义规模的完整训练实验尚未完成。

## Assignment 1：词表生成与训练

先按 [作业说明](assignment1-basics/README.md#download-data) 将 TinyStories 的训练集和验证集下载到 `assignment1-basics/data/`。以下命令从仓库根目录开始，进入作业目录后依次执行：

```bash
cd assignment1-basics
uv sync
```

### 1. 生成 10K BPE 词表

```bash
uv run python -m cs336_basics.cli train-tokenizer \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories-tokenizer.json \
  --vocab-size 10000
```

输出 JSON 包含词表、BPE 合并规则和默认特殊 token `<|endoftext|>`。

### 2. 将文本编码为训练 tokens

训练集和验证集使用同一个分词器，输出为 `uint16` token ID 数组：

```bash
uv run python -m cs336_basics.cli tokenize \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories-train.bin \
  --tokenizer data/tinystories-tokenizer.json

uv run python -m cs336_basics.cli tokenize \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories-valid.bin \
  --tokenizer data/tinystories-tokenizer.json
```

### 3. 按讲义低资源配置训练

512 维、4 层、16 头；batch 32 × 5,000 步 × context 256，共处理 **40,960,000 tokens**。这与之前用于验证流程的 1K 词表、128 维、300 步短跑不同。

```bash
uv run python -m cs336_basics.cli train \
  --train-data data/tinystories-train.bin \
  --valid-data data/tinystories-valid.bin \
  --output runs/tinystories \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 --d-ff 1344 \
  --num-layers 4 --num-heads 16 \
  --batch-size 32 --steps 5000 \
  --warmup-iters 100 \
  --lr 0.0003 --min-lr 0.00003 \
  --eval-every 100 --eval-batches 10 \
  --checkpoint-every 500 \
  --device auto
```

`auto` 自动选择 CUDA、MPS 或 CPU。学习率是可调起点，尚未验证该设置能达到讲义的验证 loss ≤2.00 目标。输出目录保存 `config.json`、训练和验证指标 `metrics.jsonl`、续训文件 `checkpoint.pt`。

中断后，在相同训练命令末尾加 `--resume runs/tinystories/checkpoint.pt` 即可恢复；保持其他参数一致，`--steps 5000` 表示训练总步数。

### 4. 生成文本

```bash
uv run python -m cs336_basics.cli generate \
  --checkpoint runs/tinystories/checkpoint.pt \
  --tokenizer data/tinystories-tokenizer.json \
  --prompt 'Once upon a time' \
  --max-new-tokens 256 --temperature 0.8 --top-p 0.95 \
  --device auto
```

## 数据与实验产物

数据集、虚拟环境、工具缓存和训练产物不提交到 Git。相关忽略规则见 [`.gitignore`](.gitignore)。训练所需的数据请按各作业说明单独下载。

## 来源与许可

作业脚手架、测试和讲义来自 Stanford CS336，各目录保留其原始来源说明与许可证。Assignment 1 来源为 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，许可证见 [`assignment1-basics/LICENSE`](assignment1-basics/LICENSE)。

本仓库中的个人实现用于自学，与 Stanford 无隶属关系。
