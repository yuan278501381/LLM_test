# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 10: 卷积与视觉感知 (CNN -> ViT -> CLIP) - 零基础入门保姆级教学平台

解剖 2D 卷积滑动计算、多通道特征图提取、Vision Transformer (ViT) 图块切片与 CLIP 跨模态图文对齐。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _apply_light_theme
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.conv2d import Conv2D, im2col
from nn_core.clip import get_pretrained_clip_data, contrastive_loss
from nn_core.vit import PatchEmbedding

st.set_page_config(
    page_title="Vision Perception · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="卷积与视觉感知架构",
    subtitle="从空间感受野到多模态统一：解剖 2D 卷积 $Y = X * W$、ViT 图像切片 Token 化与 CLIP 图文对比对齐",
    badge_text="MILESTONE 10 // VISION PERCEPTION",
    badge_type="emerald",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="视觉模型与多模态感知入门",
    plain_intro=(
        "<b>为什么处理文字的 Transformer 能看懂图片？</b><br>"
        "在传统计算机视觉中，<b>卷积核 (CNN)</b> 像一个放大镜，在图像上一步步滑动，提取边缘、纹理等局部特征；<br>"
        "而 2020 年诞生的 <b>Vision Transformer (ViT)</b> 提出了革命性的思想：把图片切成一个个网格小图块 (Patch)，"
        "每个图块就像一个'词'，直接送入标准的 Transformer 中阅读！<br>"
        "随后 <b>CLIP</b> 更是让文字和图片投影到了同一个语义向量空间，彻底打通了多模态大模型的任督二脉！"
    ),
    hyperparams_desc=(
        "• <b>卷积核类型</b>：边缘检测 (Sobel)、锐化 (Sharpen)、高斯模糊 (Gaussian) 等经典空间滤波矩阵。<br>"
        "• <b>ViT Patch 尺寸</b>：将图像切分为 $P \\times P$ 大小的图块。尺寸越小，Token 序列越长，捕捉细节越精细。<br>"
        "• <b>合成图像形状</b>：选择内置的合成几何测试流形（圆形/方形/三角形）。"
    ),
    telemetry_desc=(
        "• <b>卷积输出特征图</b>：空间滑动滤波后的梯度与响应强度分布。<br>"
        "• <b>ViT 图块网格与维度</b>：图像空间切片到序列向量空间的精确映射关系。<br>"
        "• <b>CLIP 余弦相似度矩阵</b>：8 组文本与图像在统一高维语义空间中的对齐得分。"
    ),
    experiments=[
        "<b>第 1 步【体验卷积滤波】</b>：在左侧切换【卷积核类型】，观察 Sobel-X 如何精准提取<b>左右垂直轮廓</b>（水平梯度 ∂I/∂x），Sobel-Y 如何提取<b>上下水平轮廓</b>（垂直梯度 ∂I/∂y）！",
        "<b>第 2 步【体验 ViT 切片】</b>：将【Patch Size】从 8 改为 4，观察图像被细分为 64 个 Token，直观理解图像如何转化为语言模型的输入序列！",
        "<b>第 3 步【观测 CLIP 图文对齐】</b>：滚动至 Section 4，观察对角线上明亮的黄色高分点，验证图文双塔如何将'猫咪文字'与'猫咪图片'自动拉近！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

img_choice = st.sidebar.selectbox(
    "合成测试图像",
    options=["圆形 (Circle)", "方形 (Square)", "三角形 (Triangle)"],
    index=0,
)

kernel_choice = st.sidebar.selectbox(
    "卷积核类型 (Filter Kernel)",
    options=[
        "Sobel 水平边缘 (Sobel-X)",
        "Sobel 垂直边缘 (Sobel-Y)",
        "拉普拉斯全向边缘 (Laplacian)",
        "图像锐化 (Sharpen)",
        "高斯模糊 (Gaussian Blur)",
    ],
    index=0,
)

vit_patch_size = st.sidebar.select_slider(
    "ViT Patch 尺寸 (Patch Size)",
    options=[4, 8, 16],
    value=8,
    help="将 32x32 图像切分为 PxP 的小方块",
)


# ---------------------------------------------------------------------------
# 辅助函数：生成 32x32 合成灰度图与卷积核
# ---------------------------------------------------------------------------
def _generate_synthetic_image(name: str, size: int = 32) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    if "Circle" in name or "圆形" in name:
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 3) ** 2
        img[mask] = 1.0
    elif "Square" in name or "方形" in name:
        pad = size // 4
        img[pad : size - pad, pad : size - pad] = 1.0
    else:  # Triangle
        for i in range(size // 4, 3 * size // 4):
            width = i - size // 4
            img[i, center - width : center + width + 1] = 1.0
    return img


KERNELS_DICT = {
    "Sobel 水平边缘 (Sobel-X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Sobel 垂直边缘 (Sobel-Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "拉普拉斯全向边缘 (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "图像锐化 (Sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "高斯模糊 (Gaussian Blur)": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0,
}

raw_img = _generate_synthetic_image(img_choice, size=32)
# 构造 4D 张量: (1, 1, 32, 32)
input_tensor = raw_img.reshape(1, 1, 32, 32)

# 执行卷积
cur_kernel = KERNELS_DICT[kernel_choice]
conv_op = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_op.weights = cur_kernel.reshape(1, 1, 3, 3).astype(np.float64)
conv_op.biases = np.zeros(1, dtype=np.float64)

feature_map = conv_op.forward(input_tensor)[0, 0]

# ViT Patch 嵌入
patch_embed_op = PatchEmbedding(img_size=32, patch_size=vit_patch_size, in_channels=1, d_model=32)
vit_tokens = patch_embed_op.forward(input_tensor)
num_patches = (32 // vit_patch_size) ** 2

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "INPUT RESOLUTION // 输入图像尺寸",
        "32 × 32 PX",
        delta="单通道灰度合成流形",
        delta_type="positive",
        icon_name="eye",
    )
    + render_metric_card(
        "FEATURE MAP // 卷积输出尺寸",
        f"{feature_map.shape[0]} × {feature_map.shape[1]}",
        delta=f"Stride=1, Pad=1 (保持原分辨率)",
        delta_type="positive",
        icon_name="target",
    )
    + render_metric_card(
        "VIT PATCH TOKENS // 序列长度",
        f"{num_patches} + 1 [CLS]",
        delta=f"每个图块 {vit_patch_size}×{vit_patch_size} px",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "TOKEN DIMENSION // 嵌入通道",
        "32-DIM",
        delta="完全对齐 Transformer",
        delta_type="positive",
        icon_name="layers",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 2D 卷积滑动计算与特征图提取
# ---------------------------------------------------------------------------
render_section_heading("2D CONVOLUTION SLIDING WINDOW // 卷积滑动滤波计算演示", icon_name="eye")

col_img_in, col_kernel, col_img_out = st.columns([1.2, 0.8, 1.2])

with col_img_in:
    fig_in = go.Figure(
        data=go.Heatmap(
            z=raw_img,
            colorscale="Viridis",
            showscale=False,
            hovertemplate="X: %{x}, Y: %{y}<br>像素强度: %{z:.2f}<extra></extra>",
        )
    )
    fig_in.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_in = _apply_light_theme(fig_in, f"输入原图 ({img_choice})")
    st.plotly_chart(fig_in, use_container_width=True)

with col_kernel:
    with st.container(border=True):
        st.markdown(f"#### [FILTER // 当前卷积核]\n**{kernel_choice}**")
        st.caption("3×3 感受野离散差分矩阵：")
        st.code(str(cur_kernel), language="text")
        st.markdown(
            "$$Y_{i,j} = \\sum_{m=0}^2 \\sum_{n=0}^2 X_{i+m, j+n} \\cdot W_{m,n}$$"
        )

with col_img_out:
    fig_out = go.Figure(
        data=go.Heatmap(
            z=feature_map,
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(thickness=10, len=0.8),
            hovertemplate="X: %{x}, Y: %{y}<br>特征激活响应: %{z:.2f}<extra></extra>",
        )
    )
    fig_out.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_out = _apply_light_theme(fig_out, "输出特征响应图 (Feature Map)")
    st.plotly_chart(fig_out, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: 多通道特征图对比
# ---------------------------------------------------------------------------
render_section_heading("MULTI-CHANNEL FEATURE MAPS // 经典空间滤波核特征响应矩阵", icon_name="target")

sub_cols = st.columns(4)
sample_kernels = [
    ("Sobel-X (水平梯度)", KERNELS_DICT["Sobel 水平边缘 (Sobel-X)"]),
    ("Sobel-Y (垂直梯度)", KERNELS_DICT["Sobel 垂直边缘 (Sobel-Y)"]),
    ("Laplacian (全向高频)", KERNELS_DICT["拉普拉斯全向边缘 (Laplacian)"]),
    ("Sharpen (高通锐化)", KERNELS_DICT["图像锐化 (Sharpen)"]),
]

for idx, (k_name, k_mat) in enumerate(sample_kernels):
    with sub_cols[idx]:
        c_op = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        c_op.weights = k_mat.reshape(1, 1, 3, 3).astype(np.float64)
        c_op.biases = np.zeros(1, dtype=np.float64)
        f_map = c_op.forward(input_tensor)[0, 0]

        sub_fig = go.Figure(
            data=go.Heatmap(
                z=f_map,
                colorscale="Magma",
                showscale=False,
                hovertemplate=f"{k_name}<br>强度: %{{z:.2f}}<extra></extra>",
            )
        )
        sub_fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
            margin=dict(l=5, r=5, t=25, b=5),
        )
        sub_fig = _apply_light_theme(sub_fig, k_name)
        st.plotly_chart(sub_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: ViT 图块切片与 Token 化
# ---------------------------------------------------------------------------
render_section_heading("VISION TRANSFORMER (ViT) PATCHING // 图像切片与 Token 化流水线", icon_name="cpu")

col_vit_img, col_vit_info = st.columns([1.2, 1])

with col_vit_img:
    # 绘制带 Patch 网格的原图
    fig_grid = go.Figure(
        data=go.Heatmap(
            z=raw_img,
            colorscale="Viridis",
            showscale=False,
        )
    )
    # 添加 Patch 分割网格线
    P = vit_patch_size
    shapes = []
    for line_pos in range(P, 32, P):
        shapes.append(
            dict(type="line", x0=line_pos - 0.5, x1=line_pos - 0.5, y0=-0.5, y1=31.5, line=dict(color="#ffffff", width=1.5, dash="dot"))
        )
        shapes.append(
            dict(type="line", x0=-0.5, x1=31.5, y0=line_pos - 0.5, y1=line_pos - 0.5, line=dict(color="#ffffff", width=1.5, dash="dot"))
        )
    fig_grid.update_layout(
        shapes=shapes,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_grid = _apply_light_theme(fig_grid, f"ViT 切片网格 (Patch={P}×{P}, 共 {num_patches} 个图块)")
    st.plotly_chart(fig_grid, use_container_width=True)

with col_vit_info:
    with st.container(border=True):
        st.markdown(
            f"""
            #### [VIT PIPELINE // 图像转 Token]
            - **原始图像尺寸**：`1 × 32 × 32` (1024 像素)
            - **单图块维度**：`{vit_patch_size} × {vit_patch_size} = {vit_patch_size**2}` 个数值
            - **线性投影层**：$W_{{proj}} \\in \\mathbb{{R}}^{{{vit_patch_size**2} \\times 32}}$
            - **输出序列张量**：`({vit_tokens.shape[0]}, {vit_tokens.shape[1]}, {vit_tokens.shape[2]})`
              - `Token 0`: **[CLS] 类别全局聚合标记**
              - `Token 1~{num_patches}`: **Patch 1~{num_patches} 空间局部图块**
            - **位置编码**：叠加 1D 正弦绝对位置编码注入空间几何位置信息。
            """
        )

# ---------------------------------------------------------------------------
# Section 4: CLIP 图文跨模态对齐空间 (2026 前沿)
# ---------------------------------------------------------------------------
render_section_heading("CLIP MULTI-MODAL ALIGNMENT // 跨模态图文对齐空间", icon_name="activity")

labels_clip, texts_clip, sim_matrix = get_pretrained_clip_data()
loss_val = contrastive_loss(sim_matrix, temperature=0.07)

col_clip_mat, col_clip_desc = st.columns([1.3, 1])

with col_clip_mat:
    fig_clip = go.Figure(
        data=go.Heatmap(
            z=sim_matrix,
            x=labels_clip,
            y=labels_clip,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Cosine Sim", thickness=10, len=0.8),
            hovertemplate="图像: %{y}<br>文本: %{x}<br>余弦相似度: %{z:.3f}<extra></extra>",
        )
    )
    fig_clip.update_layout(
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=30, t=30, b=80),
    )
    fig_clip = _apply_light_theme(fig_clip, "CLIP 跨模态余弦相似度矩阵 (对角线为正样本对)")
    st.plotly_chart(fig_clip, use_container_width=True)

with col_clip_desc:
    with st.container(border=True):
        st.markdown(
            f"""
            #### [CONTRASTIVE LEARNING // InfoNCE 损失]
            - **当前温度系数**：$\\tau = 0.07$
            - **当前对齐损失**：`{loss_val:.4f}` (越低对齐越好)
            - **核心公式**：
            $$L = -\\frac{{1}}{{2N}} \\sum_{{i=1}}^N \\left( \\log \\frac{{e^{{S_{{ii}}/\\tau}}}}{{\\sum_j e^{{S_{{ij}}/\\tau}}}} + \\log \\frac{{e^{{S_{{ii}}/\\tau}}}}{{\\sum_j e^{{S_{{ji}}/\\tau}}}} \\right)$$
            - **几何直觉**：通过对比学习将同语义的图文对（对角线）在超球面上相互拉近，将不匹配的图文对推远。
            """
        )

# ---------------------------------------------------------------------------
# 零基础进阶：视觉多模态核心公式逐字拆解与名词通俗速查
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 视觉卷积、ViT 与 CLIP 核心公式拆解与大白话全解", expanded=True):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：2D 空间卷积滤波
        $$(I * K)(i, j) = \\sum_{m=-1}^1 \\sum_{n=-1}^1 I(i+m, j+n) \\cdot K(m, n)$$
        
        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$I$** | **输入图像像素矩阵 (Image)** | $H \\times W$ (如 $32 \\times 32$) | 原始图片的灰度或彩色数值矩阵。 |
        | **$K$** | **卷积核滤镜 (Kernel / Filter)** | $3 \\times 3$ (小滑块) | **特征探针**。比如 Sobel-X 算子左负右正，专门在图片上寻找“从左到右剧烈明暗变化”的垂直边缘。 |
        | **$* (星号)$** | **2D 卷积滑动内积** | 运算符号 | 卷积核在整张图片上从左往右、从上往下像扫描仪一样滑动，对应位置相乘并累加。 |
        | **$(I * K)$** | **输出特征图 (Feature Map)** | $H \\times W$ | 经过滤镜扫描提取后的特征地图（边缘变亮，平坦区域变黑）。 |
        
        ---
        
        ### 1. 什么是【ViT (Vision Transformer)】？—— “把图片剪成九宫格喂给语言模型”
        * **生活比喻**：以前的 CNN 像一个手持放大镜逐行扫描的侦探；而 ViT 像一个剪刀手，直接把整张照片剪成 $4 \\times 4$ 共 16 个小纸片（Patch），给每个小纸片贴上编号（位置编码），然后当成一句话里的 16 个单词直接喂给标准的 Transformer！
        * **[CLS] Token 的妙用**：在所有图块前面额外插入一个特殊的“班长代表 [CLS]”，它在自注意力中跟所有小纸片交谈，最后单代表整张图片汇报分类结果！
        
        ---
        
        ### 2. 什么是【CLIP 图文对齐】？—— “连连看游戏”
        * **双塔架构**：一个视觉塔（处理图片）和一个文本塔（处理文字），分别把图片和文字映射到同一个高维向量空间。
        * **目标**：“猫咪的照片”和“写着'一只猫'的文字”，在空间里的距离被拉得极近；而“猫咪照片”和“写着'汽车'的文字”被推得极远！
        """
    )

