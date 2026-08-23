# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_paged_kv_cache.py - PagedAttention 显存分页与前缀共享单元测试集
"""

import numpy as np
import pytest

from nn_core.paged_kv_cache import PagedKVCache


def test_paged_kv_cache_basic_allocation_and_read():
    """验证基础物理块分配与跨块虚拟读取重组"""
    mgr = PagedKVCache(total_blocks=8, block_size=4, num_kv_heads=2, head_dim=8)

    # 初始 7 个 token -> 应占用 2 个物理块 (4 + 3)
    k_prompt = np.random.randn(2, 7, 8)
    v_prompt = np.random.randn(2, 7, 8)

    block_ids = mgr.allocate_sequence("req1", k_prompt, v_prompt)
    assert len(block_ids) == 2
    assert len(mgr.free_block_ids) == 6

    # 验证读取数据一致性
    k_read, v_read = mgr.read_full_kv("req1")
    np.testing.assert_allclose(k_read, k_prompt)
    np.testing.assert_allclose(v_read, v_prompt)


def test_paged_kv_cache_append_token_and_block_expansion():
    """验证逐 Token 追加与跨块边界自动扩容"""
    mgr = PagedKVCache(total_blocks=8, block_size=4, num_kv_heads=2, head_dim=8)

    k_prompt = np.random.randn(2, 3, 8)
    v_prompt = np.random.randn(2, 3, 8)
    mgr.allocate_sequence("req1", k_prompt, v_prompt)

    # 当前块有 3 个 token，追加第 4 个填满当前块
    tok_k1 = np.random.randn(2, 8)
    tok_v1 = np.random.randn(2, 8)
    bid1, cow1 = mgr.append_token("req1", tok_k1, tok_v1)
    assert not cow1
    assert len(mgr.block_tables["req1"]) == 1

    # 追加第 5 个 token，应自动分配新物理块
    tok_k2 = np.random.randn(2, 8)
    tok_v2 = np.random.randn(2, 8)
    bid2, cow2 = mgr.append_token("req1", tok_k2, tok_v2)
    assert not cow2
    assert len(mgr.block_tables["req1"]) == 2
    assert bid2 != bid1

    k_read, _ = mgr.read_full_kv("req1")
    assert k_read.shape == (2, 5, 8)
    np.testing.assert_allclose(k_read[:, 4, :], tok_k2)


def test_paged_kv_cache_prefix_caching_and_cow():
    """验证 Prefix Caching 前缀共享与写时复制 (Copy-on-Write)"""
    mgr = PagedKVCache(total_blocks=8, block_size=4, num_kv_heads=2, head_dim=8)

    # Req A 分配 4 个 token (刚好占满 1 个物理块)
    k_prompt = np.random.randn(2, 4, 8)
    v_prompt = np.random.randn(2, 4, 8)
    mgr.allocate_sequence("reqA", k_prompt, v_prompt)
    b_id = mgr.block_tables["reqA"][0]

    # Req B 共享 Req A 的前缀
    mgr.fork_sequence_prefix("reqA", "reqB")
    assert mgr.block_tables["reqB"] == [b_id]
    assert mgr.physical_blocks[b_id].ref_count == 2

    # Req B 追加 Token -> 由于当前块满且 ref_count=2，应触发新物理块分配并保持原共享块不变
    tok_k = np.random.randn(2, 8)
    tok_v = np.random.randn(2, 8)
    _new_bid, _ = mgr.append_token("reqB", tok_k, tok_v)

    assert len(mgr.block_tables["reqB"]) == 2
    assert mgr.physical_blocks[b_id].ref_count == 2  # 前一块仍共享

    stats = mgr.get_memory_stats()
    assert stats["shared_saved_tokens"] > 0


def test_paged_kv_cache_incomplete_shared_block_cow():
    """验证共享未满物理块时追加 Token 触发真实的写时复制 (CoW)"""
    mgr = PagedKVCache(total_blocks=8, block_size=4, num_kv_heads=2, head_dim=8)

    # Req A 分配 2 个 token (物理块未填满, num_filled=2 < 4)
    k_prompt = np.random.randn(2, 2, 8)
    v_prompt = np.random.randn(2, 2, 8)
    mgr.allocate_sequence("reqA", k_prompt, v_prompt)
    b_id = mgr.block_tables["reqA"][0]

    # Req B 共享未填满的 Req A
    mgr.fork_sequence_prefix("reqA", "reqB")
    assert mgr.physical_blocks[b_id].ref_count == 2

    # Req B 追加 Token -> 写入未满的共享块，应触发 CoW 拷贝至新私有块
    tok_k = np.random.randn(2, 8)
    tok_v = np.random.randn(2, 8)
    new_bid, triggered_cow = mgr.append_token("reqB", tok_k, tok_v)

    assert triggered_cow is True
    assert new_bid != b_id
    assert mgr.physical_blocks[b_id].ref_count == 1
    assert mgr.physical_blocks[new_bid].ref_count == 1
    assert mgr.physical_blocks[new_bid].num_filled == 3


def test_paged_kv_cache_free_sequence():
    """验证序列释放与物理块归还空闲池"""
    mgr = PagedKVCache(total_blocks=4, block_size=4, num_kv_heads=1, head_dim=4)
    k_data = np.zeros((1, 8, 4))
    v_data = np.zeros((1, 8, 4))

    mgr.allocate_sequence("req1", k_data, v_data)
    assert len(mgr.free_block_ids) == 2

    mgr.free_sequence("req1")
    assert len(mgr.free_block_ids) == 4
    assert "req1" not in mgr.block_tables


def test_paged_kv_cache_oom_and_errors():
    """验证显存耗尽 (OOM) 与非法操作防御"""
    mgr = PagedKVCache(total_blocks=2, block_size=4, num_kv_heads=1, head_dim=4)
    k_data = np.zeros((1, 16, 4))  # 需要 4 个块，但总共只有 2 个块
    v_data = np.zeros((1, 16, 4))

    with pytest.raises(RuntimeError, match="物理显存不足"):
        mgr.allocate_sequence("req_oom", k_data, v_data)

    with pytest.raises(KeyError, match="未找到请求 ID"):
        mgr.read_full_kv("req_nonexistent")
