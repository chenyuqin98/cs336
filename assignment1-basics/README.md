# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


## 本地实现与训练入口

以下命令均在 `assignment1-basics/` 下运行。基础模块、BPE、AdamW、损失、采样和 checkpoint 已接通测试；训练入口支持 CPU、MPS 和 CUDA。

### 1. 训练分词器并准备 token 数据

```bash
uv run python -m cs336_basics.cli train-tokenizer \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories-tokenizer.json --vocab-size 10000

uv run python -m cs336_basics.cli tokenize \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories-train.bin --tokenizer data/tinystories-tokenizer.json

uv run python -m cs336_basics.cli tokenize \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories-valid.bin --tokenizer data/tinystories-tokenizer.json
```

分词器使用可无损保存 bytes 的 JSON 格式，通过 `Tokenizer.save/load` 读写；`Tokenizer.from_files` 支持课程提供的 GPT-2 vocab/merges 格式。编码按文本块处理，输出默认为 `uint16` 原始数组；词表超过 65,536 个 ID 时使用 `--dtype uint32`，训练时也要指定相同 dtype。

### 2. 训练与续训

```bash
uv run python -m cs336_basics.cli train \
  --train-data data/tinystories-train.bin \
  --valid-data data/tinystories-valid.bin \
  --output runs/tinystories --device auto
```

默认模型采用讲义的 512 维、4 层、16 头、context 256 配置；默认 batch 32、5,000 步共处理 40,960,000 tokens。数据通过 `np.memmap` 读取。输出包括模型/训练配置、带迭代数与耗时的 `metrics.jsonl` 和 `checkpoint.pt`。

```bash
uv run python -m cs336_basics.cli train \
  --train-data data/tinystories-train.bin \
  --valid-data data/tinystories-valid.bin \
  --output runs/tinystories --resume runs/tinystories/checkpoint.pt --device auto
```

续训恢复模型、优化器、完成步数和随机状态。继续同一实验时，应保持原有 batch、总步数、数据和调度参数；`--steps` 表示总步数，不是额外步数。checkpoint 内的模型配置会覆盖命令行模型配置。调整总步数会改变后续学习率调度。

### 3. 生成文本

```bash
uv run python -m cs336_basics.cli generate \
  --checkpoint runs/tinystories/checkpoint.pt \
  --tokenizer data/tinystories-tokenizer.json \
  --prompt 'Once upon a time' --max-new-tokens 256 \
  --temperature 0.8 --top-p 0.95
```

支持 EOS 停止、温度采样和 top-p；`--temperature 0` 使用贪心解码。超过上下文长度后，使用最近的 context tokens 继续生成。

### 4. 消融与对比实验

单次训练可指定：

- `--norm-mode none`：移除 RMSNorm。
- `--norm-mode post`：切换为 post-norm。
- `--no-rope`：去掉位置编码。
- `--ffn-type silu --d-ff 2048`：使用无门控 SiLU，近似匹配默认 SwiGLU 参数量。

批量执行学习率、batch size 和架构对比：

```bash
uv run python -m cs336_basics.experiments \
  --train-data data/tinystories-train.bin \
  --valid-data data/tinystories-valid.bin \
  --output runs/comparisons --vocab-size 10000 --steps 5000 --device auto
```

加 `--small-model --steps 300` 可运行小模型检查。每个实验单独保存配置、学习曲线数据和 checkpoint，汇总保存在 `summary.json`。较大的 batch 需要足够显存；失败会记录退出码。

实测结果与尚未执行的完整实验见 [`EXPERIMENTS.md`](EXPERIMENTS.md)，概念题和资源核算见 [`WRITEUP.md`](WRITEUP.md)。
