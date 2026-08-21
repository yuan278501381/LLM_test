# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.styles.theme - 世界级纯净现代亮色视觉引擎 (Light Mode Design System)

提供高对比度、无遮挡、极度清晰的现代亮色 UI 规范（Stripe / Apple / Linear 质感）：
- 纯净白卡片与多维高对比排版（杜绝任何浅色字发虚、被白层遮挡的问题）
- 侧边栏与主工作区完全亮色统一（深色文字 + 浅色底板）
- 极简高精矢量 SVG 图标
- 严格杜绝 Markdown 多行缩进解析冲突
"""

import streamlit as st
from dashboard.styles.icons import svg_icon


def apply_custom_theme() -> None:
    """
    注入全局现代极简高对比度亮色 CSS 样式表。
    全面适配 High-DPI / 4K 屏幕与全操作系统缩放（Windows / macOS / Linux）。
    """
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* -------------------------------------------------------------------------
   全局根变量 (纯净亮色高对比度基底)
------------------------------------------------------------------------- */
:root {
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --bg-sidebar: #f1f5f9;
    --border-card: #e2e8f0;
    --border-hover: #94a3b8;
    --text-primary: #0f172a;    /* 极深板岩灰，保证 100% 锐利可读 */
    --text-secondary: #334155;  /* 正文高对比灰 */
    --text-muted: #64748b;      /* 辅助信息灰 */
    --brand-blue: #1d4ed8;      /* 纯正皇家蓝 */
    --accent-emerald: #047857;  /* 森林深绿 */
    --accent-purple: #6d28d9;   /* 深紫 */
    --accent-amber: #b45309;    /* 琥珀深棕 */
    --accent-rose: #be123c;     /* 艳红 */
}

/* 全局背景与文字重置 */
html, body, .stApp {
    background-color: var(--bg-page) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    color: var(--text-primary) !important;
}

/* 隐藏 Streamlit 冗余元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background-color: transparent !important;}

/* 主工作区内边距 */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1440px !important;
}

/* -------------------------------------------------------------------------
   侧边栏完全亮色化与导航条文字锐利化
------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-card) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

[data-testid="stSidebarNav"] span {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-size: 0.92rem !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] span {
    color: var(--brand-blue) !important;
    font-weight: 700 !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: rgba(37, 99, 235, 0.08) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important;
    border-color: #cbd5e1 !important;
}

/* -------------------------------------------------------------------------
   纯白卡片体系 (杜绝任何暗色底板与发虚白字)
------------------------------------------------------------------------- */
.cyber-card {
    background: #ffffff !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 12px !important;
    padding: 1.35rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
}

.cyber-card:hover {
    border-color: var(--border-hover) !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
    transform: translateY(-2px) !important;
}

/* -------------------------------------------------------------------------
   交互式导航卡片 (Clickable Bento Navigation Card)
------------------------------------------------------------------------- */
.cyber-nav-card {
    background: #ffffff !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 14px !important;
    padding: 1.4rem !important;
    margin-bottom: 1.1rem !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-decoration: none !important;
    display: block !important;
    cursor: pointer !important;
    color: inherit !important;
    position: relative !important;
}

.cyber-nav-card:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 12px 28px -4px rgba(37, 99, 235, 0.14) !important;
    transform: translateY(-3px) !important;
}

.cyber-nav-card .nav-action-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--brand-blue);
    margin-top: 0.8rem;
    transition: all 0.2s ease;
}

.cyber-nav-card:hover .nav-action-link {
    transform: translateX(4px);
    color: #1d4ed8;
}

/* -------------------------------------------------------------------------
   标题与 Hero Header (高对比度深色渐变)
------------------------------------------------------------------------- */
.hero-title {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 60%, #4338ca 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.3rem !important;
    line-height: 1.2 !important;
}

.hero-subtitle {
    font-size: 1.05rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    margin-bottom: 1.5rem !important;
    line-height: 1.6 !important;
}

/* -------------------------------------------------------------------------
   行内代码与公式块 (高对比度背景)
------------------------------------------------------------------------- */
code, pre, .stCode {
    font-family: 'JetBrains Mono', Consolas, "Fira Code", monospace !important;
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.4rem !important;
    font-size: 0.88em !important;
}

/* -------------------------------------------------------------------------
   指标卡 (Telemetry Metric Cards)
------------------------------------------------------------------------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.telemetry-card {
    background: #ffffff !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.2s ease !important;
}

.telemetry-card:hover {
    border-color: var(--brand-blue) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08) !important;
}

.telemetry-label {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
}

.telemetry-value {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin: 0.25rem 0 !important;
    letter-spacing: -0.02em !important;
}

.telemetry-delta {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
}

.delta-positive { color: var(--accent-emerald) !important; }
.delta-negative { color: var(--accent-rose) !important; }
.delta-neutral  { color: var(--text-muted) !important; }

/* -------------------------------------------------------------------------
   徽章与发光标签 (Pill Badges)
------------------------------------------------------------------------- */
.pill-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    padding: 0.25rem 0.75rem !important;
    border-radius: 9999px !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.pill-blue {
    background: #eff6ff !important;
    color: var(--brand-blue) !important;
    border: 1px solid #bfdbfe !important;
}

.pill-emerald {
    background: #ecfdf5 !important;
    color: var(--accent-emerald) !important;
    border: 1px solid #a7f3d0 !important;
}

.pill-purple {
    background: #f5f3ff !important;
    color: var(--accent-purple) !important;
    border: 1px solid #ddd6fe !important;
}

.pill-amber {
    background: #fffbeb !important;
    color: var(--accent-amber) !important;
    border: 1px solid #fde68a !important;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    background-color: currentColor !important;
}

/* -------------------------------------------------------------------------
   按钮与交互控件
------------------------------------------------------------------------- */
.stButton>button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.2rem !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: var(--text-primary) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
}

.stButton>button:hover {
    border-color: var(--brand-blue) !important;
    color: var(--brand-blue) !important;
    background: #f8fafc !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12) !important;
    transform: translateY(-1px) !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
}
</style>""",
        unsafe_allow_html=True,
    )


def render_hero_header(
    title: str,
    subtitle: str,
    badge_text: str = "PURE NUMPY · ZERO BLACKBOX",
    badge_type: str = "blue",
) -> None:
    """渲染纯净亮色高对比度 Hero 头部"""
    badge_class = f"pill-{badge_type}"
    html = (
        f'<div style="margin-bottom: 1.8rem;">'
        f'<div style="margin-bottom: 0.6rem;">'
        f'<span class="pill-badge {badge_class}"><span class="status-dot"></span>{badge_text}</span>'
        f'</div>'
        f'<h1 class="hero-title">{title}</h1>'
        f'<div class="hero-subtitle">{subtitle}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_heading(title: str, icon_name: str = "activity", subtext: str | None = None) -> None:
    """渲染带极简矢量图标的分区标题"""
    icon_html = svg_icon(icon_name, size=18, color="#1d4ed8")
    sub_html = f'<div style="color: #64748b; font-size: 0.85rem; margin-top: 0.2rem; font-weight: 500;">{subtext}</div>' if subtext else ""
    html = (
        f'<div style="margin-top: 1.4rem; margin-bottom: 0.8rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.5rem;">'
        f'{icon_html}'
        f'<span style="font-size: 1.15rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em;">{title}</span>'
        f'</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "neutral",
    icon_name: str = "activity",
) -> str:
    """生成带矢量 SVG 图标的亮色指标卡 HTML"""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="telemetry-delta {delta_class}">{delta}</div>' if delta else ""
    icon_svg = svg_icon(icon_name, size=14, color="#1d4ed8")
    return (
        f'<div class="telemetry-card">'
        f'<div class="telemetry-label"><span>{icon_svg}</span> {label}</div>'
        f'<div class="telemetry-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def render_preset_badge(preset_name: str, desc: str) -> None:
    """渲染预设方案的说明卡片"""
    icon_zap = svg_icon("zap", size=14, color="#1d4ed8")
    html = (
        f'<div style="background: #eff6ff; border-left: 4px solid #1d4ed8; padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; font-size: 0.88rem; color: #1e3a8a; border: 1px solid #dbeafe; border-left-width: 4px;">'
        f'<div style="display: flex; align-items: center; gap: 0.4rem; font-weight: 700; color: #1d4ed8; margin-bottom: 0.2rem;">'
        f'{icon_zap} 预设方案: {preset_name}'
        f'</div>'
        f'<span style="color: #334155;">{desc}</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
