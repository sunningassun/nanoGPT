# nanoGPT
nanoGPT 是由 Andrej Karpathy 基于 PyTorch 开发的极简 GPT 训练与微调框架，核心仅约 600 行代码（model.py + train.py），完整复现 GPT‑2 结构与训练流程。它剥离冗余封装，保留 Transformer decoder 核心，兼顾可读性与训练效率，支持单卡 / 多卡训练、混合精度与模型微调，是学习大模型底层原理、快速搭建轻量级文本生成模型的经典工程实现。
