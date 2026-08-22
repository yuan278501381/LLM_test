# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.video - 纯 NumPy 合成视频生成、时空图块嵌入 (Spatio-Temporal Patch) 与帧间动力学特征提取

包含：
- `generate_synthetic_video`: 32x32 多模式连续运动灰度视频序列合成 (反射弹跳/平移/扩散膨胀/圆周旋转/弧形平移)
- `VideoFrameSampler`: 均匀抽样与关键帧运动突变差异抽样器
- `SpatioTemporalPatchEmbed`: 教学级 3D 视频时空切片投影层
- `compute_frame_difference` / `compute_frame_change_magnitude`: 帧间差分能量与全局像素扰动范数计算
"""

import logging

import numpy as np

logger = logging.getLogger("nn_core.video")


def generate_synthetic_video(
    n_frames: int = 8, size: int = 32, motion: str = "bounce"
) -> np.ndarray:
    """
    合成指定帧数与运动模式的 32x32 灰度视频序列。
    返回 shape: (n_frames, 1, size, size)，值域 [0.0, 1.0]。
    """
    video = np.zeros((n_frames, 1, size, size), dtype=np.float32)
    center = size // 2
    radius = size // 8

    for t in range(n_frames):
        img = np.zeros((size, size), dtype=np.float32)
        if motion == "bounce" or "弹跳" in motion:
            # 垂直碰壁反弹：三角波往复运动 (物理反射弹跳)
            phase = (t / max(1, n_frames - 1)) * 2.0  # 完整往复周期
            tri_wave = 1.0 - abs(2.0 * (phase - np.floor(phase + 0.5)))
            cy = int(radius + tri_wave * (size - 2 * radius))
            cx = int(center + (size // 4) * np.sin(2 * np.pi * t / n_frames))
            y, x = np.ogrid[:size, :size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            img[mask] = 1.0
        elif motion == "circular" or "圆周" in motion or "旋转" in motion:
            # 匀速圆周运动轨迹
            cx = int(center + (size // 3) * np.sin(2 * np.pi * t / n_frames))
            cy = int(center + (size // 3) * np.cos(2 * np.pi * t / n_frames))
            y, x = np.ogrid[:size, :size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            img[mask] = 1.0
        elif motion == "slide" or "滑动" in motion:
            # 物体从左往右匀速平移
            cx = int((size - 2 * radius) * (t / max(1, n_frames - 1))) + radius
            cy = center
            y, x = np.ogrid[:size, :size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            img[mask] = 1.0
        elif motion == "grow" or "膨胀" in motion:
            # 物体从中心逐渐变大
            cur_r = int(radius * (1.0 + 2.0 * t / max(1, n_frames - 1)))
            y, x = np.ogrid[:size, :size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= cur_r**2
            img[mask] = 1.0
        else:  # 矩形方块弧形平移
            cx = int(center + (size // 4) * np.cos(np.pi * t / n_frames))
            cy = int(center + (size // 4) * np.sin(np.pi * t / n_frames))
            pad = radius
            x_min, x_max = max(0, cx - pad), min(size, cx + pad)
            y_min, y_max = max(0, cy - pad), min(size, cy + pad)
            img[y_min:y_max, x_min:x_max] = 1.0

        video[t, 0] = img

    return video


def compute_frame_difference(video: np.ndarray) -> np.ndarray:
    """计算相邻帧之间的均方差 (MSE) 能量"""
    # video: (T, 1, H, W)
    diffs = np.diff(video, axis=0)
    mse = np.mean(diffs**2, axis=(1, 2, 3))
    return mse.astype(np.float32)


def compute_frame_change_magnitude(video: np.ndarray) -> np.ndarray:
    """
    计算每帧相对于前一帧的全局像素变化 Frobenius / L2 范数（非密集光流矢量场）。
    """
    diffs = np.diff(video, axis=0)
    norms = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
    return norms.astype(np.float32)


# 向后兼容别名
compute_motion_vector = compute_frame_change_magnitude


class VideoFrameSampler:
    """
    视频关键帧与时域均匀下采样器。
    """

    def __init__(self, strategy: str = "uniform") -> None:
        self.strategy = strategy

    def sample(self, video: np.ndarray, n_sample: int = 4) -> tuple[np.ndarray, list[int]]:
        """
        从原视频中抽取 n_sample 帧。
        返回 (sampled_video, indices)
        """
        T = video.shape[0]
        if n_sample >= T:
            return video, list(range(T))

        if self.strategy == "keyframe" or "关键帧" in self.strategy:
            # 依据帧间变化率选取能量最大的关键帧
            diffs = compute_frame_difference(video)
            # 首帧必选，其余按变化率排序后重新按时间戳排序
            top_diff_indices = np.argsort(diffs)[::-1][: n_sample - 1] + 1
            indices = sorted(set([0, *top_diff_indices.tolist()]))
            # 若数量不足，用均匀采样补齐
            if len(indices) < n_sample:
                uniform_indices = np.linspace(0, T - 1, n_sample, dtype=int).tolist()
                indices = sorted(list(set(indices + uniform_indices)))[:n_sample]
        else:
            # 均匀等间距采样
            indices = np.linspace(0, T - 1, n_sample, dtype=int).tolist()

        sampled = video[indices]
        return sampled, indices


class SpatioTemporalPatchEmbed:
    """
    时空 3D 视频图块嵌入层 (Spatio-Temporal Patch Embedding)。

    将 5D 视频张量 $X \\in \\mathbb{R}^{B \\times T \\times C \\times H \\times W}$ 按照空间 $P \\times P$
    切分并引入时空双重位置编码：
    $$Z = \\text{Linear}(\\text{Flatten}(\\text{Patch}_{t,p})) + E_{spatial} + E_{temporal}$$
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        n_frames: int = 8,
        in_channels: int = 1,
        d_model: int = 32,
    ) -> None:
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_frames = n_frames
        self.in_channels = in_channels
        self.d_model = d_model

        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.total_tokens = n_frames * self.num_spatial_patches
        self.patch_dim = in_channels * patch_size * patch_size

        # 线性投影矩阵
        self.proj_weights = np.random.randn(self.patch_dim, d_model) * np.sqrt(2.0 / self.patch_dim)
        self.proj_bias = np.zeros(d_model)

        # 空间位置编码与时间位置编码
        self.spatial_pos = np.random.randn(1, 1, self.num_spatial_patches, d_model) * 0.02
        self.temporal_pos = np.random.randn(1, n_frames, 1, d_model) * 0.02

    def forward(self, video: np.ndarray) -> np.ndarray:
        """
        video shape: (B, T, C, H, W)
        返回 token 序列: (B, T * num_spatial_patches, d_model)
        """
        B, T, C, H, W = video.shape
        P = self.patch_size
        gh = H // P
        gw = W // P

        # 空间切片: (B, T, gh, gw, C, P, P) -> (B, T, gh*gw, C*P*P)
        patches = video.reshape(B, T, C, gh, P, gw, P).transpose(0, 1, 3, 5, 2, 4, 6)
        patches = patches.reshape(B, T, gh * gw, C * P * P)

        # 线性映射: (B, T, num_spatial_patches, d_model)
        tokens = np.dot(patches, self.proj_weights) + self.proj_bias

        # 注入空间与时间协同位置编码
        tokens = tokens + self.spatial_pos + self.temporal_pos

        # 展平为单一序列: (B, T * num_spatial_patches, d_model)
        return tokens.reshape(B, T * self.num_spatial_patches, self.d_model)
