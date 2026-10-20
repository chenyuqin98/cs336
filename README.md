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

目前 Git 跟踪以 Assignment 1 为主，其余作业和 lectures 目录仍在根目录 `.gitignore` 中，仅保存在本地。

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

[`tests/adapters.py`](assignment1-basics/tests/adapters.py) 负责将实现接入课程测试。BPE 训练与分词、数据采样、交叉熵、梯度裁剪、AdamW、学习率调度和 checkpoint 接口仍待补齐。已有实现是否满足要求，以实际测试结果为准。

## 数据与实验产物

数据集、虚拟环境、工具缓存和训练产物不提交到 Git。相关忽略规则见 [`.gitignore`](.gitignore)。训练所需的数据请按各作业说明单独下载。

## 来源与许可

作业脚手架、测试和讲义来自 Stanford CS336，各目录保留其原始来源说明与许可证。Assignment 1 来源为 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，许可证见 [`assignment1-basics/LICENSE`](assignment1-basics/LICENSE)。

本仓库中的个人实现用于自学，与 Stanford 无隶属关系。
