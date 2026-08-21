# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
# =============================================================================
# 多阶段 Docker 构建 — 最小化镜像体积 + 安全加固
# =============================================================================
# 阶段 1: 构建层 (安装依赖)
# 阶段 2: 运行层 (仅复制必要文件)
# =============================================================================

# ---- 阶段 1: 构建依赖 ----
FROM python:3.14-slim AS builder

# 安装 uv (极速包管理器)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先复制依赖声明 (利用 Docker 层缓存)
COPY pyproject.toml uv.lock* ./

# 安装依赖到虚拟环境 (不安装项目本身)
RUN uv sync --frozen --no-install-project --no-dev

# 复制项目源码
COPY nn_core/ nn_core/
COPY datasets/ datasets/
COPY dashboard/ dashboard/

# 安装项目
RUN uv sync --frozen --no-dev


# ---- 阶段 2: 精简运行镜像 ----
FROM python:3.14-slim AS runtime

# 安全加固: 使用非 root 用户
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# 从构建阶段复制虚拟环境和源码
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/nn_core /app/nn_core
COPY --from=builder /app/datasets /app/datasets
COPY --from=builder /app/dashboard /app/dashboard
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# 创建日志目录
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# 环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=dark

# 暴露端口
EXPOSE 8501

# 健康检查 — 每 30 秒检查 Streamlit 是否响应
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 切换到非 root 用户
USER appuser

# 启动 Streamlit
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

# 元数据标签 (OCI 标准)
LABEL org.opencontainers.image.title="NN Playground" \
      org.opencontainers.image.description="手搓神经网络可视化实验平台" \
      org.opencontainers.image.authors="Yy1 (yuan278501381)" \
      org.opencontainers.image.url="https://github.com/yuan278501381" \
      org.opencontainers.image.licenses="MIT"
