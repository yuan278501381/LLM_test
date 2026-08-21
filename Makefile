# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
# =============================================================================
# Makefile — 跨平台开发者工作流入口
# =============================================================================
.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-cov build docker-build docker-run clean pre-commit

# 颜色定义
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## 📋 显示帮助信息
	@echo ""
	@echo "$(CYAN)🧠 NN Playground — 开发者命令$(RESET)"
	@echo "────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# 📦 环境管理
# ---------------------------------------------------------------------------
install: ## 📦 安装所有依赖 (含 dev)
	uv sync

lock: ## 🔒 锁定依赖版本
	uv lock

# ---------------------------------------------------------------------------
# 🔍 代码质量
# ---------------------------------------------------------------------------
lint: ## 🔍 运行 Ruff 代码检查
	uv run ruff check .

format: ## 🎨 自动格式化代码
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## 🏷️ Pyright 类型检查
	uv run pyright nn_core/ datasets/

quality: lint typecheck ## 🏆 运行全部代码质量检查

# ---------------------------------------------------------------------------
# 🧪 测试
# ---------------------------------------------------------------------------
test: ## 🧪 运行全部测试
	uv run pytest tests/ -v --tb=short

test-fast: ## ⚡ 快速测试 (跳过慢速测试)
	uv run pytest tests/ -v --tb=short -m "not slow"

test-cov: ## 📊 运行测试 (带覆盖率报告)
	uv run pytest tests/ -v --tb=short \
		--cov=nn_core --cov=datasets \
		--cov-report=term-missing \
		--cov-report=html:reports/htmlcov \
		--cov-report=xml:reports/coverage.xml

test-gradients: ## 🧮 仅运行梯度校验测试
	uv run pytest tests/test_gradients.py -v

# ---------------------------------------------------------------------------
# 📦 构建 & 发布
# ---------------------------------------------------------------------------
build: ## 📦 构建 wheel & sdist
	uv build

# ---------------------------------------------------------------------------
# 🐳 Docker
# ---------------------------------------------------------------------------
docker-build: ## 🐳 构建 Docker 镜像
	docker build -t nn-playground:latest .

docker-run: ## 🐳 启动 Docker 容器
	docker compose up -d
	@echo "$(GREEN)✅ 仪表板已启动: http://localhost:8501$(RESET)"

docker-stop: ## 🐳 停止 Docker 容器
	docker compose down

docker-logs: ## 🐳 查看容器日志
	docker compose logs -f

# ---------------------------------------------------------------------------
# 🚀 启动
# ---------------------------------------------------------------------------
dashboard: ## 🚀 启动 Streamlit 仪表板
	uv run streamlit run dashboard/app.py --server.port 8501

# ---------------------------------------------------------------------------
# 🔒 安全
# ---------------------------------------------------------------------------
audit: ## 🔒 依赖安全审计
	uv run pip-audit --strict

security: ## 🔒 Bandit 安全静态分析
	uv run bandit -r nn_core/ datasets/ -f screen

# ---------------------------------------------------------------------------
# 🛠️ 工具
# ---------------------------------------------------------------------------
pre-commit: ## 🛠️ 安装 pre-commit hooks
	uv run pre-commit install
	@echo "$(GREEN)✅ Pre-commit hooks 已安装$(RESET)"

pre-commit-all: ## 🛠️ 对所有文件运行 pre-commit
	uv run pre-commit run --all-files

# ---------------------------------------------------------------------------
# 🧹 清理
# ---------------------------------------------------------------------------
clean: ## 🧹 清理构建产物和缓存
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .pyright
	rm -rf reports/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "$(GREEN)✅ 清理完成$(RESET)"

# ---------------------------------------------------------------------------
# 🏆 完整 CI 模拟 (本地)
# ---------------------------------------------------------------------------
ci: quality test-cov build ## 🏆 本地完整 CI 模拟
	@echo "$(GREEN)✅ 所有 CI 阶段通过$(RESET)"
