# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core - 手搓神经网络核心引擎 (2026 现代 LLM 全栈版)

纯 NumPy 实现的神经网络与现代 LLM 框架，包含：
    - 激活函数：Sigmoid, ReLU, Tanh, LeakyReLU, Softmax, GELU, SiLU
    - 损失函数：MSE, BinaryCrossEntropy, CategoricalCrossEntropy
    - 网络层：Dense, Dropout, LayerNorm, SwiGLU
    - 优化器：SGD, Momentum, RMSProp, Adam (支持通用参数协议)
    - 初始化：zeros, random, xavier, he
    - 正则化：L1, L2
    - 模型：Sequential
    - 现代 LLM 扩展：
        - BPE 分词器 (BytePairEncoder)
        - 词嵌入与正弦编码 (Embedding, PositionalEncoding)
        - 旋转位置编码 (RotaryPositionalEmbedding / RoPE)
        - 注意力机制 (MultiHeadAttention, GroupedQueryAttention / GQA, CausalMask)
        - 门控前馈网络 (SwiGLU, FeedForward)
        - 结构块与生成模型 (TransformerBlock, TinyGPT, KVCache)
"""

from nn_core.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from nn_core.callbacks import EarlyStopping, ExperimentLogger, TrainingHistory
from nn_core.initializers import he_init, random_init, xavier_init, zeros_init
from nn_core.layers import Dense, Dropout
from nn_core.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from nn_core.model import Sequential
from nn_core.optimizers import SGD, Adam, Momentum, RMSProp
from nn_core.regularizers import L1, L2
from nn_core.tensor import clip_gradients, safe_exp, safe_log, set_seed
from nn_core.embeddings import Embedding, PositionalEncoding, get_mini_vocab, get_pretrained_embeddings
from nn_core.rnn import RNNCell
from nn_core.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from nn_core.layernorm import LayerNorm
from nn_core.gelu import GELU
from nn_core.transformer import FeedForward, TransformerBlock
from nn_core.gpt import TinyGPT

# 2026 现代 LLM 新特性导入
from nn_core.bpe import BytePairEncoder
from nn_core.rope import RotaryPositionalEmbedding, apply_rope, precompute_freqs_cis
from nn_core.gqa import GroupedQueryAttention, repeat_kv
from nn_core.swiglu import SwiGLU, silu
from nn_core.kv_cache import KVCache

__all__ = [
    "L1",
    "L2",
    "MSE",
    "SGD",
    "Adam",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "Dense",
    "Dropout",
    "EarlyStopping",
    "ExperimentLogger",
    "LeakyReLU",
    "Momentum",
    "RMSProp",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "TrainingHistory",
    "clip_gradients",
    "he_init",
    "random_init",
    "safe_exp",
    "safe_log",
    "set_seed",
    "xavier_init",
    "zeros_init",
    "Embedding",
    "PositionalEncoding",
    "get_mini_vocab",
    "get_pretrained_embeddings",
    "RNNCell",
    "MultiHeadAttention",
    "causal_mask",
    "scaled_dot_product_attention",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "TinyGPT",
    "BytePairEncoder",
    "RotaryPositionalEmbedding",
    "apply_rope",
    "precompute_freqs_cis",
    "GroupedQueryAttention",
    "repeat_kv",
    "SwiGLU",
    "silu",
    "KVCache",
]

__version__ = "0.2.0"
__author__ = "Yy1 (yuan278501381)"
