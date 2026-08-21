# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.styles.theme - 世界级 Neo-Bento 赛博深色玻璃拟态视觉引擎

提供全局 CSS 注入、高分屏排版优化、发光卡片、状态指示灯、指标卡等世界级 UI 组件。
全系采用极简高精矢量 SVG 图标，杜绝一切 Emoji 元素。
"""

import streamlit as st
from dashboard.styles.icons import svg_icon


def apply_custom_theme() -> None:
    """
    注入全局现代深色玻璃拟态 CSS 样式表。
    全面适配 High-DPI / 4K 屏幕与全操作系统缩放（Windows / macOS / Linux）。
    """
    st.markdown(
        """
        <style>
        /* -------------------------------------------------------------------------
           全局字体与背景基底
        ------------------------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg-deep: #080c14;
            --bg-card: rgba(15, 23, 42, 0.65);
            --bg-card-hover: rgba(30, 41, 59, 0.8);
            --border-glow: rgba(56, 189, 248, 0.25);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --neon-blue: #38bdf8;
            --neon-indigo: #818cf8;
            --neon-purple: #c084fc;
            --neon-emerald: #34d399;
            --neon-rose: #fb7185;
            --neon-amber: #fbbf24;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
        }

        code, pre, .stCode {
            font-family: 'JetBrains Mono', Consolas, "Fira Code", monospace !important;
        }

        /* 隐藏 Streamlit 冗余元素，净化视野 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: transparent !important;}

        /* -------------------------------------------------------------------------
           主工作区容器与 Bento 网格
        ------------------------------------------------------------------------- */
        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: 1440px !important;
        }

        /* 侧边栏玻璃拟态美化 */
        [data-testid="stSidebar"] {
            background-color: rgba(8, 12, 20, 0.88) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        [data-testid="stSidebar"] hr {
            margin: 1.2rem 0 !important;
            border-color: rgba(255, 255, 255, 0.06) !important;
        }

        /* -------------------------------------------------------------------------
           卡片与面板 (Glassmorphic Cards)
        ------------------------------------------------------------------------- */
        .cyber-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .cyber-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
        }

        .cyber-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 16px 40px -12px rgba(56, 189, 248, 0.15);
            transform: translateY(-2px);
        }

        /* -------------------------------------------------------------------------
           标题与 Hero Header
        ------------------------------------------------------------------------- */
        .hero-title {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
            margin-bottom: 0.3rem !important;
            line-height: 1.2 !important;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: var(--text-muted);
            font-weight: 400;
            margin-bottom: 1.5rem;
            line-height: 1.6;
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
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.2s ease;
        }

        .telemetry-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            background: rgba(30, 41, 59, 0.7);
        }

        .telemetry-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94a3b8;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .telemetry-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'JetBrains Mono', monospace;
            margin: 0.25rem 0;
            letter-spacing: -0.02em;
        }

        .telemetry-delta {
            font-size: 0.78rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .delta-positive { color: #34d399; }
        .delta-negative { color: #fb7185; }
        .delta-neutral  { color: #94a3b8; }

        /* -------------------------------------------------------------------------
           徽章与发光标签 (Glowing Badges & Pills)
        ------------------------------------------------------------------------- */
        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        .pill-blue {
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        .pill-emerald {
            background: rgba(52, 211, 153, 0.12);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        .pill-purple {
            background: rgba(192, 132, 252, 0.12);
            color: #c084fc;
            border: 1px solid rgba(192, 132, 252, 0.3);
        }

        .pill-amber {
            background: rgba(251, 191, 36, 0.12);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 6px currentColor;
        }

        /* -------------------------------------------------------------------------
           按钮交互与现代阴影
        ------------------------------------------------------------------------- */
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.02em !important;
            padding: 0.5rem 1.2rem !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)) !important;
            color: #f8fafc !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }

        .stButton>button:hover {
            border-color: rgba(56, 189, 248, 0.5) !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(129, 140, 248, 0.3)) !important;
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25) !important;
            transform: translateY(-1px) !important;
        }

        .stButton>button:active {
            transform: translateY(1px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header(
    title: str,
    subtitle: str,
    badge_text: str = "PURE NUMPY · ZERO BLACKBOX",
    badge_type: str = "blue",
) -> None:
    """渲染世界级科技感 Hero 头部"""
    badge_class = f"pill-{badge_type}"
    st.markdown(
        f"""
        <div style="margin-bottom: 1.8rem;">
            <div style="margin-bottom: 0.6rem;">
                <span class="pill-badge {badge_class}">
                    <span class="status-dot"></span>
                    {badge_text}
                </span>
            </div>
            <h1 class="hero-title">{title}</h1>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, icon_name: str = "activity", subtext: str | None = None) -> None:
    """渲染带极简矢量图标的分区标题"""
    icon_html = svg_icon(icon_name, size=18, color="#38bdf8")
    sub_html = f'<div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div style="margin-top: 1.4rem; margin-bottom: 0.8rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                {icon_html}
                <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">{title}</span>
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "neutral",
    icon_name: str = "activity",
) -> str:
    """生成带矢量 SVG 图标的科技感遥测指标卡 HTML"""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="telemetry-delta {delta_class}">{delta}</div>' if delta else ""
    icon_svg = svg_icon(icon_name, size=14, color="#38bdf8")
    return f"""
    <div class="telemetry-card">
        <div class="telemetry-label"><span>{icon_svg}</span> {label}</div>
        <div class="telemetry-value">{value}</div>
        {delta_html}
    </div>
    """


def render_preset_badge(preset_name: str, desc: str) -> None:
    """渲染预设方案的说明卡片"""
    icon_zap = svg_icon("zap", size=14, color="#38bdf8")
    st.markdown(
        f"""
        <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; font-size: 0.85rem; color: #cbd5e1;">
            <div style="display: flex; align-items: center; gap: 0.4rem; font-weight: 600; color: #38bdf8; margin-bottom: 0.2rem;">
                {icon_zap} 预设方案: {preset_name}
            </div>
            <span>{desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
