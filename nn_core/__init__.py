# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core - 手搓神经网络核心引擎 (2026 现代 LLM、多模态与世界模型全生命周期版)

纯 NumPy 实现的神经网络、现代 LLM、多模态感知与对齐评估框架，包含：
    - 基础篇：激活函数、损失函数、全连接层、Dropout、正则化、优化器、Sequential 训练容器
    - 语言篇：BPE 分词器、词嵌入、正弦位置编码、RoPE 旋转位置编码、MHA、GQA、SwiGLU、TransformerBlock、TinyGPT、KV-Cache
    - 视觉与音频篇：Conv2D、MaxPool2D、ViT、CLIP、STFT、MelFilterbank、SpectrogramFramePatcher
    - 视频与世界模型篇：SpatioTemporalPatchEmbed、视频帧采样、教学级 NextFramePredictor、DDPM 前向加噪调度
    - 预训练与对齐篇：MLM (BERT)、CLM (GPT)、MAE (视觉自编码)、NT-Xent (对比学习)、RewardModel、PPO-Clip、DPO、LoRA (低秩微调)
    - 评估体系篇：Perplexity、EvaluationHarness、MMLU / HellaSwag / GSM8K / Safety 基准考场
"""

from nn_core.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from nn_core.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from nn_core.audio import (
    AudioTokenizer,
    SpectrogramFramePatcher,
    compute_mel_spectrogram,
    generate_chord,
    generate_waveform,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    numpy_to_wav_bytes,
    stft,
)

# 现代 LLM 基础算子
from nn_core.bpe import BytePairEncoder
from nn_core.callbacks import EarlyStopping, ExperimentLogger, TrainingHistory
from nn_core.clip import CLIPDualEncoder, contrastive_loss, get_pretrained_clip_data

# 进阶多模态感知与世界模型
from nn_core.conv2d import Conv2D, MaxPool2D, col2im, im2col
from nn_core.embeddings import (
    Embedding,
    PositionalEncoding,
    get_mini_vocab,
    get_pretrained_embeddings,
)
from nn_core.evaluation import (
    BenchmarkQuestion,
    BenchmarkTask,
    EvaluationHarness,
    compute_accuracy,
    compute_f1,
    compute_perplexity,
    get_mini_gsm8k,
    get_mini_hellaswag,
    get_mini_mmlu,
    get_mini_safety,
)
from nn_core.gelu import GELU
from nn_core.gpt import TinyGPT
from nn_core.gqa import GroupedQueryAttention, repeat_kv
from nn_core.initializers import he_init, random_init, xavier_init, zeros_init
from nn_core.kv_cache import KVCache
from nn_core.layernorm import LayerNorm
from nn_core.layers import Dense, Dropout
from nn_core.lora import LoRALayer, compute_param_savings
from nn_core.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from nn_core.model import Sequential
from nn_core.optimizers import SGD, Adam, Momentum, RMSProp
from nn_core.posttraining import AlignmentPipeline, generate_before_after_examples

# 预训练范式、后训练对齐与评测体系
from nn_core.pretraining import (
    CausalLanguageModel,
    ContrastiveLearning,
    DataMixtureEngine,
    MaskedAutoEncoder,
    MaskedLanguageModel,
    PretrainingComparator,
    ScalingLawEngine,
    SimpleBPE,
)
from nn_core.regularizers import L1, L2
from nn_core.rlhf import DPOLoss, PPOClipObjective, RewardModel
from nn_core.rnn import RNNCell
from nn_core.rope import RotaryPositionalEmbedding, apply_rope, precompute_freqs_cis
from nn_core.swiglu import SwiGLU, silu
from nn_core.tensor import clip_gradients, safe_exp, safe_log, set_seed
from nn_core.transformer import FeedForward, TransformerBlock
from nn_core.video import (
    SpatioTemporalPatchEmbed,
    VideoFrameSampler,
    compute_frame_change_magnitude,
    compute_frame_difference,
    compute_motion_vector,
    generate_synthetic_video,
)
from nn_core.vit import PatchEmbedding, VisionTransformer
from nn_core.world_model import (
    DiffusionScheduler,
    NextFramePredictor,
    visualize_diffusion_process,
)

__all__ = [
    "GELU",
    # 基础
    "L1",
    "L2",
    "MSE",
    "SGD",
    "Adam",
    "AlignmentPipeline",
    "AudioTokenizer",
    # 评测体系
    "BenchmarkQuestion",
    "BenchmarkTask",
    "BinaryCrossEntropy",
    "BytePairEncoder",
    "CLIPDualEncoder",
    "CategoricalCrossEntropy",
    "CausalLanguageModel",
    "ContrastiveLearning",
    # 视觉与音频
    "Conv2D",
    "DPOLoss",
    "DataMixtureEngine",
    "Dense",
    "DiffusionScheduler",
    "Dropout",
    "EarlyStopping",
    # 序列与 Transformer
    "Embedding",
    "EvaluationHarness",
    "ExperimentLogger",
    "FeedForward",
    "GroupedQueryAttention",
    "KVCache",
    "LayerNorm",
    "LeakyReLU",
    "LoRALayer",
    "MaskedAutoEncoder",
    # 预训练与后训练对齐
    "MaskedLanguageModel",
    "MaxPool2D",
    "Momentum",
    "MultiHeadAttention",
    "NextFramePredictor",
    "PPOClipObjective",
    "PatchEmbedding",
    "PositionalEncoding",
    "PretrainingComparator",
    "RMSProp",
    "RNNCell",
    "ReLU",
    "RewardModel",
    "RotaryPositionalEmbedding",
    "ScalingLawEngine",
    "Sequential",
    "Sigmoid",
    "SimpleBPE",
    "Softmax",
    "SpatioTemporalPatchEmbed",
    "SpectrogramFramePatcher",
    "SwiGLU",
    "Tanh",
    "TinyGPT",
    "TrainingHistory",
    "TransformerBlock",
    "VideoFrameSampler",
    "VisionTransformer",
    "apply_rope",
    "causal_mask",
    "clip_gradients",
    "col2im",
    "compute_accuracy",
    "compute_f1",
    "compute_frame_change_magnitude",
    "compute_frame_difference",
    "compute_mel_spectrogram",
    "compute_motion_vector",
    "compute_param_savings",
    "compute_perplexity",
    "contrastive_loss",
    "generate_before_after_examples",
    "generate_chord",
    # 视频与世界模型
    "generate_synthetic_video",
    "generate_waveform",
    "get_mini_gsm8k",
    "get_mini_hellaswag",
    "get_mini_mmlu",
    "get_mini_safety",
    "get_mini_vocab",
    "get_pretrained_clip_data",
    "get_pretrained_embeddings",
    "he_init",
    "hz_to_mel",
    "im2col",
    "mel_filterbank",
    "mel_to_hz",
    "numpy_to_wav_bytes",
    "precompute_freqs_cis",
    "random_init",
    "repeat_kv",
    "safe_exp",
    "safe_log",
    "scaled_dot_product_attention",
    "set_seed",
    "silu",
    "stft",
    "visualize_diffusion_process",
    "xavier_init",
    "zeros_init",
]

__version__ = "0.3.0"
__author__ = "Yy1 (yuan278501381)"
