# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.gpt - 迷你 GPT 模块

实现简化的基于 Transformer 的自回归语言模型。
"""

import logging
import uuid
import numpy as np

from nn_core.embeddings import Embedding, PositionalEncoding
from nn_core.transformer import TransformerBlock
from nn_core.layernorm import LayerNorm
from nn_core.attention import causal_mask

logger = logging.getLogger(__name__)


class TinyGPT:
    """
    迷你版 GPT (生成式预训练 Transformer)。
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None
    ) -> None:
        if d_ff is None:
            d_ff = d_model * 4
            
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_f = LayerNorm(d_model)
        
        # 保存注意力权重用于可视化
        self._attention_weights: list[np.ndarray] = []
        
        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] TinyGPT 已创建: layers=%d, d_model=%d, heads=%d, vocab=%d",
            tid, num_layers, d_model, num_heads, vocab_size
        )

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        前向传播。
        
        Args:
            token_ids: 形状为 (batch_size, seq_len) 的 token 索引。
        """
        batch_size, seq_len = token_ids.shape
        mask = causal_mask(seq_len)
        
        # 1. 词嵌入 + 位置编码
        x = self.embedding.forward(token_ids)
        x = self.pos_encoding.forward(x)
        
        self._attention_weights = []
        
        # 2. Transformer Blocks
        for block in self.blocks:
            x, attn = block.forward(x, mask=mask)
            self._attention_weights.append(attn)
            
        # 3. 最终层归一化
        x = self.ln_f.forward(x)
        
        # 4. 权重绑定 (Weight Tying): 使用 embedding 权重进行输出投影
        logits = x @ self.embedding.weights.T
        
        return logits

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.8, top_k: int | None = None
    ) -> list[int]:
        """
        自回归生成循环。
        """
        generated = list(prompt_ids)
        
        for _ in range(max_new_tokens):
            # 截断以适应最大序列长度
            input_ids = generated[-self.max_seq_len:]
            
            # (1, seq_len)
            x = np.array([input_ids], dtype=np.int32)
            
            # 前向传播得到 logits
            logits = self.forward(x)
            
            # 只取最后一个时间步的 logits
            next_token_logits = logits[0, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Top-K 采样
                if top_k is not None and top_k > 0:
                    indices_to_remove = np.argsort(next_token_logits)[:-top_k]
                    next_token_logits[indices_to_remove] = -1e9
                
                # Softmax 转概率
                shifted_logits = next_token_logits - np.max(next_token_logits)
                exp_logits = np.exp(shifted_logits)
                probs = exp_logits / np.sum(exp_logits)
                
                next_token = int(np.random.choice(self.vocab_size, p=probs))
            else:
                # 贪心解码
                next_token = int(np.argmax(next_token_logits))
                
            generated.append(next_token)
            
        return generated

    def get_all_attention_weights(self) -> list[np.ndarray]:
        """返回所有层的注意力权重。"""
        return self._attention_weights
