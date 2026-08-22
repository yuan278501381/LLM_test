# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.bpe - 分词工程与字节对编码模块 (Byte-Pair Encoding)

实现 2026 年现代 LLM 标配的纯 Python/NumPy BPE 分词器 (基于 minbpe 极简白盒架构)：
- 字节级初始词表 (Byte-level 0~255)
- 频次统计与自底向上贪心合并 (Greedy Merging)
- 编码 (Encode) 与解码 (Decode)
- 合并历史追踪 (Merge History)，支持交互式可视化逐步分词过程
"""

import logging
import uuid
from itertools import pairwise

logger = logging.getLogger(__name__)


def get_stats(
    ids: list[int], counts: dict[tuple[int, int], int] | None = None
) -> dict[tuple[int, int], int]:
    """统计序列中相邻 ID 对的出现频次"""
    counts = {} if counts is None else counts
    for pair in pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """将序列中所有出现的指定 (p0, p1) 对替换为新的 new_id"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BytePairEncoder:
    """
    字节级 BPE (Byte-Pair Encoding) 分词器。

    工作原理:
    1. 将原始文本按 UTF-8 编码为原始字节序列 [0~255]；
    2. 统计出现频率最高的相邻字节/Token 对；
    3. 将最高频对合并分配一个新 Token ID (>= 256)；
    4. 循环直到达到目标词表大小或无重复对。
    """

    def __init__(self, vocab_size: int = 300) -> None:
        self.vocab_size = vocab_size
        # 初始词表为 256 个单字节: {id: bytes}
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        # 合并规则: {(id0, id1): new_id}
        self.merges: dict[tuple[int, int], int] = {}
        # 记录训练合并历史用于可视化: [{"step": int, "pair": (str, str), "new_id": int, "count": int}]
        self.merge_history: list[dict] = []

        tid = uuid.uuid4().hex[:8]
        logger.info("[%s] BytePairEncoder 已创建: target_vocab=%d", tid, vocab_size)

    def train(self, text: str, verbose: bool = False) -> None:
        """
        在给定文本语料上训练 BPE 分词规则。
        """
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        num_merges = max(0, self.vocab_size - 256)

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.merge_history = []

        for step in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            # 找到最高频对
            best_pair = max(stats, key=lambda pair: stats[pair])
            count = stats[best_pair]
            if count <= 1 and step > 10:
                # 出现次数过低则提前终止
                break

            new_id = 256 + step
            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id

            # 构建新 token 的字节表示
            token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = token_bytes

            # 记录历史
            pair_repr = (
                self.vocab[best_pair[0]].decode("utf-8", errors="replace"),
                self.vocab[best_pair[1]].decode("utf-8", errors="replace"),
            )
            merged_repr = token_bytes.decode("utf-8", errors="replace")

            self.merge_history.append(
                {
                    "step": step + 1,
                    "pair_ids": best_pair,
                    "pair_str": pair_repr,
                    "merged_str": merged_repr,
                    "new_id": new_id,
                    "freq": count,
                }
            )

            if verbose:
                logger.debug(
                    "Merge #%d: %s -> %s (id=%d, count=%d)",
                    step + 1,
                    pair_repr,
                    merged_repr,
                    new_id,
                    count,
                )

    def encode(self, text: str) -> list[int]:
        """将输入文本编码为 Token ID 列表"""
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            # 找到当前 ids 中所有 pair 里在 merges 中最先被训练的 pair (new_id 最小)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # 没有更多可合并的 pair
            new_id = self.merges[pair]
            ids = merge(ids, pair, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """将 Token ID 列表还原解码为文本"""
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                part_bytes.append(f"<unk:{idx}>".encode())
        full_bytes = b"".join(part_bytes)
        return full_bytes.decode("utf-8", errors="replace")

    def tokenize_visual_chunks(self, text: str) -> list[tuple[str, int]]:
        """
        返回文本切分为各个 Token 的文本切片及其 ID (用于 UI 炫彩着色展示)。
        """
        token_ids = self.encode(text)
        chunks = []
        for tid in token_ids:
            chunk_bytes = self.vocab.get(tid, b"")
            chunk_str = chunk_bytes.decode("utf-8", errors="replace")
            chunks.append((chunk_str, tid))
        return chunks
