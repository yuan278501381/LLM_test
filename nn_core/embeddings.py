# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.embeddings - 词嵌入与位置编码模块

实现基于查表的词嵌入和正弦位置编码。
"""

import logging
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class Embedding:
    r"""
    词嵌入层 (查表层)。
    
    数学公式:
        $E = W[x]$
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 按照 GPT-2 等标准，初始化通常用正态分布
        self.weights: np.ndarray = np.random.randn(vocab_size, d_model) * 0.02
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.input_cache: np.ndarray | None = None
        
        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] Embedding 层已创建: vocab_size=%d, d_model=%d",
            tid,
            vocab_size,
            d_model,
        )

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """前向传播: 查表提取对应词向量"""
        self.input_cache = token_ids
        return self.weights[token_ids]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播: 稀疏梯度累加到 grad_weights。
        
        Embedding 层的输入是离散 token ids，不需要向后传递梯度。
        """
        if self.input_cache is None:
            raise ValueError("需要先调用 forward() 才可 backward()")
            
        np.add.at(self.grad_weights, self.input_cache, dout)
        # 对前一层的梯度全为 0 (或不需要传播)
        return np.zeros_like(self.input_cache, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Embedding(vocab_size={self.vocab_size}, d_model={self.d_model})"


class PositionalEncoding:
    r"""
    正弦位置编码。
    
    数学公式:
        $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
        $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        self.max_len = max_len
        self.d_model = d_model
        
        # 预计算位置编码矩阵
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float64).reshape(-1, 1)
        # div_term
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # 扩展出 batch 维度供广播: shape (1, max_len, d_model)
        self.pe: np.ndarray = pe[np.newaxis, :, :]
        
        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] PositionalEncoding 已创建: max_len=%d, d_model=%d",
            tid,
            max_len,
            d_model,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: 序列中各位置直接加上预计算的正弦位置编码"""
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播: 加法直接透传梯度。"""
        return dout


def get_mini_vocab() -> dict[str, int]:
    """
    返回预置的迷你英文词表。包含经典语义对。
    """
    words = [
        # 王族
        "king", "queen", "man", "woman", "prince", "princess", "boy", "girl",
        # 动物
        "cat", "dog", "kitten", "puppy", "fish", "bird",
        # 国家
        "china", "japan", "usa", "france", "germany", "italy",
        # 城市
        "beijing", "tokyo", "washington", "paris", "berlin", "rome",
        # 动词
        "run", "walk", "eat", "drink", "sleep", "think", "speak", "write",
        # 形容词
        "big", "small", "hot", "cold", "fast", "slow", "good", "bad", "happy", "sad",
        # 其他常用词
        "the", "a", "an", "is", "are", "was", "were", "on", "in", "at", "to", 
        "and", "or", "not", "he", "she", "it", "they", "we", "I"
    ]
    vocab = {}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def get_pretrained_embeddings(vocab_size: int, d_model: int = 32) -> np.ndarray:
    """
    生成具有一定特定语义关系的预训练词向量。
    
    保证 `king - man + woman ≈ queen` 等基本语义相似性。
    """
    np.random.seed(42)
    embeddings = np.random.randn(vocab_size, d_model) * 0.1
    vocab = get_mini_vocab()
    
    def set_vec(w1: str, w2: str, dim: int, val1: float, val2: float):
        if w1 in vocab:
            embeddings[vocab[w1], dim] = val1
        if w2 in vocab:
            embeddings[vocab[w2], dim] = val2
            
    # 手动设定一些主成分轴 (模拟语义空间)
    # 0 轴：性别轴
    set_vec("man", "woman", 0, 1.0, -1.0)
    set_vec("king", "queen", 0, 1.0, -1.0)
    set_vec("prince", "princess", 0, 1.0, -1.0)
    set_vec("boy", "girl", 0, 1.0, -1.0)
    set_vec("he", "she", 0, 1.0, -1.0)

    # 1 轴：王权轴
    set_vec("king", "man", 1, 1.0, -1.0)
    set_vec("queen", "woman", 1, 1.0, -1.0)

    # 2 轴：国家-首都关系
    set_vec("china", "beijing", 2, 1.0, -1.0)
    set_vec("japan", "tokyo", 2, 1.0, -1.0)
    set_vec("usa", "washington", 2, 1.0, -1.0)
    set_vec("france", "paris", 2, 1.0, -1.0)

    return embeddings
