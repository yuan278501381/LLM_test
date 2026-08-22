# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_video.py - 视频感知与世界模型单元测试
"""

import numpy as np

from nn_core.video import (
    SpatioTemporalPatchEmbed,
    VideoFrameSampler,
    compute_frame_change_magnitude,
    compute_frame_difference,
    compute_motion_vector,
    generate_synthetic_video,
)
from nn_core.world_model import (
    DiffusionScheduler,
    NextFramePredictor,
    visualize_diffusion_process,
)


def test_generate_synthetic_video():
    """测试 32x32 合成视频生成维度与动态特征"""
    n_frames = 8
    size = 32
    vid_bounce = generate_synthetic_video(n_frames=n_frames, size=size, motion="bounce")
    assert vid_bounce.shape == (n_frames, 1, size, size)
    assert np.min(vid_bounce) >= 0.0
    assert np.max(vid_bounce) <= 1.0

    # 验证确实存在物体运动（相邻帧不相同）
    diffs = compute_frame_difference(vid_bounce)
    assert len(diffs) == n_frames - 1
    assert np.all(diffs > 0.0), "视频相邻帧无变化"

    # 验证新函数与别名
    mags = compute_frame_change_magnitude(vid_bounce)
    assert len(mags) == n_frames - 1
    assert np.all(mags > 0.0)

    motion_vecs = compute_motion_vector(vid_bounce)
    np.testing.assert_allclose(mags, motion_vecs)


def test_video_frame_sampler():
    """测试视频均匀与关键帧抽样器"""
    video = generate_synthetic_video(n_frames=10, size=32, motion="slide")
    sampler_u = VideoFrameSampler(strategy="uniform")
    sampled_u, idx_u = sampler_u.sample(video, n_sample=4)
    assert sampled_u.shape == (4, 1, 32, 32)
    assert len(idx_u) == 4

    sampler_k = VideoFrameSampler(strategy="keyframe")
    sampled_k, idx_k = sampler_k.sample(video, n_sample=4)
    assert sampled_k.shape == (4, 1, 32, 32)
    assert len(idx_k) == 4


def test_spatio_temporal_patch_embed():
    """测试 3D 时空图块嵌入维度"""
    B = 2
    T = 8
    C = 1
    H = W = 32
    P = 8
    d_model = 32
    video = np.random.randn(B, T, C, H, W).astype(np.float32)

    st_embed = SpatioTemporalPatchEmbed(
        img_size=H, patch_size=P, n_frames=T, in_channels=C, d_model=d_model
    )
    tokens = st_embed.forward(video)
    num_spatial = (H // P) ** 2  # 16
    expected_tokens = T * num_spatial  # 128
    assert tokens.shape == (B, expected_tokens, d_model)


def test_diffusion_scheduler():
    """测试 Diffusion 方差表与前向加噪过程"""
    num_steps = 20
    scheduler = DiffusionScheduler(num_steps=num_steps)
    sched = scheduler.get_schedule()

    alphas_cumprod = sched["alphas_cumprod"]
    # 验证 alpha_bar 单调递减
    assert np.all(np.diff(alphas_cumprod) < 0.0)
    assert alphas_cumprod[0] > 0.95
    assert alphas_cumprod[-1] < 0.90

    # 验证时间步加噪
    x_0 = np.ones((1, 32, 32), dtype=np.float32)
    x_clean = scheduler.add_noise(x_0, t=-1)
    np.testing.assert_allclose(x_clean, x_0)

    # 验证扩散序列生成
    snapshots = visualize_diffusion_process(x_0, scheduler, steps_to_show=5)
    assert len(snapshots) == 5
    assert snapshots[0][0] == 0
    assert snapshots[-1][0] == num_steps - 1


def test_next_frame_predictor():
    """测试世界模型自回归下一帧预测与 MSE 损失"""
    predictor = NextFramePredictor(d_model=32, frame_pixels=1024)
    context = np.random.randn(1, 32)
    pred = predictor.forward(context)
    assert pred.shape == (1, 1024)
    assert np.all((pred >= 0.0) & (pred <= 1.0))

    true_f = np.random.rand(1024)
    loss = predictor.reconstruction_loss(pred[0], true_f)
    assert loss >= 0.0
