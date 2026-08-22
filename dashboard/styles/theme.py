# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.styles.theme - 统一的现代亮色视觉系统 (Light Mode Design System)

提供高对比度、无遮挡、极度清晰的现代亮色 UI 规范（Stripe / Apple / Linear 质感）：
- 纯净白卡片与多维高对比排版
- 侧边栏与主工作区完全亮色统一
- 极简高精矢量 SVG 图标
- 针对零基础初学者定制的保姆级新手教学指引系统 (Zero-Barrier Beginner Pedagogy)
"""

import contextlib
import importlib
import json
import sys
from functools import wraps
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from dashboard.styles.icons import svg_icon
from nn_core.observability import (
    generate_trace_id,
    get_logger,
    get_trace_id,
    set_trace_id,
    setup_logging,
)


def reload_runtime_modules() -> None:
    """在 Streamlit 热重载时安全同步 nn_core 与 dashboard.constants 核心模块，杜绝长时间运行下的模块导入缓存问题"""
    for prefix in ("nn_core.", "dashboard.constants.", "dashboard.components."):
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith(prefix) or mod_name == prefix.rstrip("."):
                mod = sys.modules.get(mod_name)
                if mod is not None:
                    with contextlib.suppress(Exception):
                        importlib.reload(mod)


def _install_plotly_playback() -> None:
    """让全站普通 Plotly 图表默认拥有无重跑的读图播放控制。"""
    if getattr(st.plotly_chart, "_nn_playback_enabled", False):
        return

    from dashboard.components.charts import add_chart_playback

    original_plotly_chart = st.plotly_chart

    @wraps(original_plotly_chart)
    def animated_plotly_chart(figure_or_data: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(figure_or_data, object) and figure_or_data.__class__.__name__ == "Figure":
            figure_or_data = add_chart_playback(figure_or_data)
        return original_plotly_chart(figure_or_data, *args, **kwargs)

    animated_plotly_chart._nn_playback_enabled = True  # type: ignore[attr-defined]
    st.plotly_chart = animated_plotly_chart


def apply_custom_theme() -> None:
    """
    注入全局现代极简高对比度亮色 CSS 样式表与全链路 TraceID 上下文。
    全面适配 High-DPI / 4K 屏幕与全操作系统缩放（Windows / macOS / Linux）。
    采用 0ms 本地原生高精字体栈，彻底根绝 FOUT (Flash of Unstyled Text) 字体粗细闪烁与页面跳动。
    """
    setup_logging()
    if "session_trace_id" not in st.session_state:
        st.session_state["session_trace_id"] = generate_trace_id("st")
    set_trace_id(str(st.session_state["session_trace_id"]))
    logger = get_logger("dashboard.theme")
    logger.debug("页面主题与 Trace 上下文加载完成 [TraceID: %s]", get_trace_id())

    _install_plotly_playback()
    st.markdown(
        """<style>
/* -------------------------------------------------------------------------
   全局根变量 (纯净亮色高对比度基底 & 0ms 本地极速字体栈)
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
    --font-system: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
    --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "PingFang SC", "Microsoft YaHei", monospace;
}

/* 全局背景与文字重置 (0ms 本地原生字体栈，杜绝 FOUT 字体粗细突变与闪烁) */
html, body, .stApp {
    background-color: var(--bg-page) !important;
    font-family: var(--font-system) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    color: var(--text-primary) !important;
}

/* 保证所有文本与表单继承稳固字体，消除挂载瞬间字号跳动 */
p, span, label, input, textarea, select, button {
    font-family: inherit;
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
   侧边栏完全亮色化与导航条文字锐利化 (消除切页字重闪烁)
------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
   世界级双栏/单栏智能自适应侧边栏 (Content-Aware Dynamic Layout)
   - 当页面无超参数控制台时 (如 M00 / 首页)：自动收拢为 240px 极简单栏，导航 100% 满宽，杜绝大片空白！
   - 当页面有超参数控制台时 (如 M01~M17)：自动展开为双栏（180px 课程罗盘 + 310px 参数微观控制台）！
------------------------------------------------------------------------- */
/* 1. 单栏模式 (无超参数控件时，如 M00 / 首页) */
/* 1. 侧边栏整体与主导航罗盘 (全站统一 Flex Row 双栏架构) */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    display: flex !important;
    flex-direction: row !important;
    height: 100vh !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* 栏位 1：主导航航母罗盘 (全站统一 210px 紧凑罗盘，0 抖动，0 高度差) */
[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
    width: 210px !important;
    min-width: 210px !important;
    max-width: 210px !important;
    height: 100% !important;
    overflow-y: auto !important;
    border-right: 1px solid #e2e8f0 !important;
    padding: 2.8rem 0.4rem 1.5rem 0.4rem !important;
    background-color: #f8fafc !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
}

/* 栏位 2：微观超参数控制台 (有内容时展开 410px 充裕舒展空间) */
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    display: block !important;
    flex: 1 1 410px !important;
    width: 410px !important;
    min-width: 410px !important;
    max-width: 410px !important;
    height: 100% !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
    padding: 2.8rem 1.2rem 1.5rem 1.2rem !important;
    background-color: #ffffff !important;
    box-sizing: border-box !important;
}

/* 当栏位 2 无控件时 (如 M00 / 首页)，自动隐藏并收拢为单栏 260px */
[data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])),
[data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])) > div {
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
    background-color: #f8fafc !important;
}

[data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])) [data-testid="stSidebarNav"] {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    border-right: none !important;
}

[data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])) [data-testid="stSidebarUserContent"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    padding: 0 !important;
}

/* 当栏位 2 有实际控件时 (M01~M17)，展开为双栏 620px */
@media (min-width: 1260px) {
    [data-testid="stSidebar"]:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]),
    [data-testid="stSidebar"]:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]) > div {
        width: 620px !important;
        min-width: 620px !important;
        max-width: 620px !important;
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
}

/* 控制台内部多列与滑块自适应容器 */
[data-testid="stSidebarUserContent"] [data-testid="stHorizontalBlock"] {
    width: 100% !important;
    gap: 0.5rem !important;
}

[data-testid="stSidebarUserContent"] [data-testid="column"] {
    min-width: 0 !important;
    flex: 1 1 calc(50% - 0.25rem) !important;
}

/* 3. 响应式折叠模式 (屏幕 < 1260px 时弹性回退为垂直单列) */
@media (max-width: 1259px) {
    [data-testid="stSidebar"] {
        width: 330px !important;
        min-width: 330px !important;
        max-width: 360px !important;
        background-color: #f8fafc !important;
    }

    [data-testid="stSidebarContent"] {
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        padding: 2.8rem 1rem 1.5rem 1rem !important;
        overflow-y: auto !important;
    }

    [data-testid="stSidebarNav"] {
        width: 100% !important;
        margin-bottom: 1rem !important;
        padding-bottom: 1rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebarUserContent"] {
        width: 100% !important;
    }
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* 展开完整侧边栏并彻底隐藏 View more / View less 按钮，杜绝页面跳转时按钮闪烁挤压布局 */
[data-testid="stSidebarNavItems"] {
    max-height: none !important;
    overflow-y: visible !important;
}

[data-testid="stSidebarNav"] button,
[data-testid="stSidebarNav"] [data-testid="stSidebarNavShowMore"],
[data-testid="stSidebarNavSeparator"] {
    display: none !important;
}

/* 绝对恒定几何尺寸：确保所有页面无论单双栏，菜单容器与每一项高度 100% 绝对一致 (36px)，0 高度差 */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNav"] ul {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    row-gap: 2px !important;
    padding: 0 !important;
    margin: 0 !important;
    list-style: none !important;
}

[data-testid="stSidebarNav"] li {
    padding: 0 !important;
    margin: 0 !important;
    height: 36px !important;
    min-height: 36px !important;
    max-height: 36px !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavLink"] {
    min-height: 36px !important;
    max-height: 36px !important;
    height: 36px !important;
    margin: 0 !important;
    padding: 0 0.5rem !important;
    display: flex !important;
    align-items: center !important;
    border-radius: 6px !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    contain: layout style !important;
    text-decoration: none !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNavLink"] span {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    line-height: 36px !important;
    transition: color 0.15s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: rgba(37, 99, 235, 0.08) !important;
    border-left: 3px solid #2563eb !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: var(--brand-blue) !important;
    font-weight: 600 !important;
}

/* Tier-2 参数微观控制台内部组件精致卡片化 */
[data-testid="stSidebarUserContent"] .stMarkdown h3,
[data-testid="stSidebarUserContent"] .stMarkdown h4 {
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    color: #1e40af !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
}

[data-testid="stSidebarUserContent"] [data-testid="stSlider"],
[data-testid="stSidebarUserContent"] [data-testid="stSelectbox"],
[data-testid="stSidebarUserContent"] [data-testid="stRadio"],
[data-testid="stSidebarUserContent"] [data-testid="stNumberInput"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stSidebarUserContent"] [data-testid="stSlider"]:hover,
[data-testid="stSidebarUserContent"] [data-testid="stSelectbox"]:hover,
[data-testid="stSidebarUserContent"] [data-testid="stRadio"]:hover,
[data-testid="stSidebarUserContent"] [data-testid="stNumberInput"]:hover {
    border-color: #93c5fd !important;
    box-shadow: 0 3px 8px rgba(37, 99, 235, 0.06) !important;
}

[data-testid="stSidebarUserContent"] label {
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    color: #1e293b !important;
}

[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important;
    border-color: #cbd5e1 !important;
}

/* -------------------------------------------------------------------------
   纯白卡片容器体系 (原生 Streamlit Container 与自定义卡片统一)
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

/* Streamlit 原生 Border Container 极客化封装 */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border-card) !important;
    border-radius: 14px !important;
    background-color: #ffffff !important;
    padding: 0.5rem !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 1rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 10px 24px -2px rgba(37, 99, 235, 0.12) !important;
    transform: translateY(-2px) !important;
}

/* Streamlit PageLink 按钮组件增强 */
[data-testid="stPageLink-NavLink"] {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    color: var(--brand-blue) !important;
    padding: 0.45rem 0.9rem !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
}

[data-testid="stPageLink-NavLink"]:hover {
    background: #dbeafe !important;
    border-color: var(--brand-blue) !important;
    transform: translateX(3px) !important;
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
    margin-bottom: 1.2rem !important;
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
    font-size: 1.5rem !important;
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

/* 全局平滑平移滚动 */
html {
    scroll-behavior: smooth !important;
}

/* 动态教学片段不得触发浏览器滚动锚定，否则播放时视口会被上下补偿。 */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    overflow-anchor: none !important;
}

.st-key-m1_player_controls,
.st-key-m1_boundary_shell,
.st-key-m1_loss_shell,
.st-key-m1_trajectory_shell {
    overflow-anchor: none !important;
    contain: layout paint !important;
}

/* 全局图表抗闪烁与 GPU 硬件加速 */
.stPlotlyChart {
    min-height: 380px !important;
    contain: layout style paint !important;
    transform: translateZ(0) !important;
    backface-visibility: hidden !important;
    will-change: transform !important;
}

/* Plotly 读图播放 4 按钮控制器与工具条：仅当鼠标悬浮 (hover) 在图表区域时显示，平时隐藏保持视觉极致干净 */
.stPlotlyChart .updatemenu-container,
.stPlotlyChart .updatemenu-button,
.stPlotlyChart g.updatemenu-container,
.stPlotlyChart g.updatemenu-button,
[data-testid="stPlotlyChart"] .updatemenu-container,
[data-testid="stPlotlyChart"] .updatemenu-button,
[data-testid="stPlotlyChart"] g.updatemenu-container,
[data-testid="stPlotlyChart"] g.updatemenu-button,
.js-plotly-plot .updatemenu-container,
.js-plotly-plot .updatemenu-button,
.js-plotly-plot g.updatemenu-container,
.js-plotly-plot g.updatemenu-button,
.plot-container .updatemenu-container,
.svg-container .updatemenu-container,
g.updatemenu-container,
g.updatemenu-button {
    opacity: 0 !important;
    visibility: hidden !important;
    transition: opacity 0.25s cubic-bezier(0.4, 0, 0.2, 1), visibility 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    pointer-events: none !important;
}

.stPlotlyChart:hover .updatemenu-container,
.stPlotlyChart:hover .updatemenu-button,
.stPlotlyChart:hover g.updatemenu-container,
.stPlotlyChart:hover g.updatemenu-button,
[data-testid="stPlotlyChart"]:hover .updatemenu-container,
[data-testid="stPlotlyChart"]:hover .updatemenu-button,
[data-testid="stPlotlyChart"]:hover g.updatemenu-container,
[data-testid="stPlotlyChart"]:hover g.updatemenu-button,
.js-plotly-plot:hover .updatemenu-container,
.js-plotly-plot:hover .updatemenu-button,
.js-plotly-plot:hover g.updatemenu-container,
.js-plotly-plot:hover g.updatemenu-button,
.plot-container:hover .updatemenu-container,
.svg-container:hover .updatemenu-container {
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
}

/* 文本防抖动：等宽数字排版 */
.tabular-nums, .hero-metric-value, .metric-value, [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums !important;
    font-feature-settings: "tnum" 1 !important;
}

/* 教学指代聚焦：高能量视觉呼吸脉冲 + 饱满光晕，确保视线从屏幕右侧扫视过来时清晰捕获 */
.interactive-region {
    border-radius: 10px !important;
    position: relative !important;
    transition: background-color 0.4s ease, border-color 0.4s ease,
                box-shadow 0.4s ease, transform 0.4s ease !important;
    scroll-margin-top: 80px !important;
}

.nn-focus-target,
.interactive-region:target {
    animation: region-focus-enter 4.2s cubic-bezier(0.16, 1, 0.3, 1) both !important;
    border-left: 6px solid #1d4ed8 !important;
}

@keyframes region-focus-enter {
    0% {
        opacity: 0.85;
        background-color: #dbeafe;
        box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.42), 0 16px 45px rgba(37, 99, 235, 0.28);
        border-color: #2563eb;
        transform: translateY(6px) scale(1.006);
    }
    20% {
        opacity: 1;
        background-color: #bfdbfe;
        box-shadow: 0 0 0 10px rgba(37, 99, 235, 0.32), 0 20px 50px rgba(37, 99, 235, 0.35);
        border-color: #1d4ed8;
        transform: translateY(0) scale(1.008);
    }
    45% {
        opacity: 1;
        background-color: #dbeafe;
        box-shadow: 0 0 0 8px rgba(37, 99, 235, 0.22), 0 14px 36px rgba(37, 99, 235, 0.22);
        border-color: #3b82f6;
        transform: translateY(0) scale(1.004);
    }
    75% {
        opacity: 1;
        background-color: #eff6ff;
        box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.12), 0 8px 24px rgba(37, 99, 235, 0.12);
        border-color: #60a5fa;
        transform: translateY(0) scale(1);
    }
    100% {
        opacity: 1;
        background-color: #ffffff;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        border-color: #e2e8f0;
        transform: translateY(0) scale(1);
    }
}

.nn-focus-chip {
    position: absolute !important;
    left: 0.85rem !important;
    right: auto !important;
    top: -0.9rem !important;
    z-index: 9999 !important;
    padding: 0.28rem 0.85rem !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 999px !important;
    background: #0f172a !important;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.45), 0 0 0 3px rgba(56, 189, 248, 0.35) !important;
    color: #ffffff !important;
    font-size: 0.76rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    pointer-events: none !important;
    animation: focus-chip-enter 4.2s cubic-bezier(0.16, 1, 0.3, 1) both !important;
}

@keyframes focus-chip-enter {
    0% { opacity: 0; transform: translateY(10px) scale(0.92); }
    12%, 80% { opacity: 1; transform: translateY(0) scale(1); }
    100% { opacity: 0; transform: translateY(-4px) scale(0.96); }
}

@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto !important; }
    .nn-focus-target, .interactive-region:target, .nn-focus-chip {
        animation: none !important;
    }
    .nn-focus-target {
        background: #eff6ff !important;
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
}

/* 浮动导航只在宽屏显示，避免覆盖中等视口的教学图表。 */
@media (max-width: 2199px) {
    #nn-floating-spatial-hud { display: none !important; }
}

/* 空间映射语义锚点胶囊 (Spatial Semantic Badges) */
.anchor-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0.15rem 0.55rem !important;
    border-radius: 6px !important;
    vertical-align: middle !important;
    margin: 0 0.15rem !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
    text-decoration: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.anchor-badge:hover {
    transform: translateY(-1px) scale(1.03) !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08) !important;
}
.anchor-badge-amber {
    background: #fffbeb !important;
    color: #92400e !important;
    border: 1px solid #fde68a !important;
}
.anchor-badge-amber:hover {
    background: #fef3c7 !important;
    border-color: #f59e0b !important;
}
.anchor-badge-blue {
    background: #eff6ff !important;
    color: #1e40af !important;
    border: 1px solid #bfdbfe !important;
}
.anchor-badge-blue:hover {
    background: #dbeafe !important;
    border-color: #3b82f6 !important;
}
.anchor-badge-emerald {
    background: #ecfdf5 !important;
    color: #065f46 !important;
    border: 1px solid #a7f3d0 !important;
}
.anchor-badge-emerald:hover {
    background: #d1fae5 !important;
    border-color: #10b981 !important;
}
.anchor-badge-purple {
    background: #f5f3ff !important;
    color: #5b21b6 !important;
    border: 1px solid #ddd6fe !important;
}
.anchor-badge-purple:hover {
    background: #ede9fe !important;
    border-color: #8b5cf6 !important;
}
.anchor-badge-rose {
    background: #fff1f2 !important;
    color: #9f1239 !important;
    border: 1px solid #fecdd3 !important;
}
.anchor-badge-rose:hover {
    background: #ffe4e6 !important;
    border-color: #f43f5e !important;
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

/* -------------------------------------------------------------------------
   交互折叠框与读图指南 (Interactive Reading Guide / Explainer Cards)
------------------------------------------------------------------------- */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-left: 3.5px solid #2563eb !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05) !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.9rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"]:hover {
    border-color: #3b82f6 !important;
    border-left: 3.5px solid #1d4ed8 !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.12) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stExpander"] summary {
    background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%) !important;
    padding: 0.65rem 1rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    color: #1e3a8a !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stExpander"] summary:hover {
    color: #1d4ed8 !important;
    background: #dbeafe !important;
}

[data-testid="stExpander"] summary svg {
    color: #2563eb !important;
    stroke-width: 2.5 !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 1rem 1.25rem !important;
    background: #ffffff !important;
    border-top: 1px solid #e2e8f0 !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    color: #1e293b !important;
}
</style>""",
        unsafe_allow_html=True,
    )

    sidebar_nav_js = """<!doctype html><html><head><meta charset="utf-8"></head><body>
    <script>
    (function() {
        try {
            var doc = window.parent.document;
            if (!doc) return;

            var labelMap = {
                'app': '首页 · 导航大厅',
                '数学基础': 'M00 · 数学基础',
                '0 数学基础': 'M00 · 数学基础',
                '0_数学基础': 'M00 · 数学基础',
                '单神经元感知器': 'M01 · 单神经元感知器',
                '1 单神经元感知器': 'M01 · 单神经元感知器',
                '1_单神经元感知器': 'M01 · 单神经元感知器',
                '多层网络': 'M02 · 多层网络',
                '2 多层网络': 'M02 · 多层网络',
                '2_多层网络': 'M02 · 多层网络',
                '优化器对比': 'M03 · 优化器对比',
                '3 优化器对比': 'M03 · 优化器对比',
                '3_优化器对比': 'M03 · 优化器对比',
                '参数实验室': 'M04 · 参数实验室',
                '4 参数实验室': 'M04 · 参数实验室',
                '4_参数实验室': 'M04 · 参数实验室',
                '词嵌入空间': 'M05 · 词嵌入空间',
                '5 词嵌入空间': 'M05 · 词嵌入空间',
                '5_词嵌入空间': 'M05 · 词嵌入空间',
                '序列记忆': 'M06 · 序列记忆',
                '6 序列记忆': 'M06 · 序列记忆',
                '6_序列记忆': 'M06 · 序列记忆',
                '注意力机制': 'M07 · 注意力机制',
                '7 注意力机制': 'M07 · 注意力机制',
                '7_注意力机制': 'M07 · 注意力机制',
                'Transformer': 'M08 · Transformer',
                '8 Transformer': 'M08 · Transformer',
                '8_Transformer': 'M08 · Transformer',
                'Mini_GPT': 'M09 · Mini-GPT',
                'Mini GPT': 'M09 · Mini-GPT',
                '9 Mini_GPT': 'M09 · Mini-GPT',
                '9 Mini GPT': 'M09 · Mini-GPT',
                '9_Mini_GPT': 'M09 · Mini-GPT',
                '视觉感知': 'M10 · 视觉感知',
                '10 视觉感知': 'M10 · 视觉感知',
                '10_视觉感知': 'M10 · 视觉感知',
                '音频感知': 'M11 · 音频感知',
                '11 音频感知': 'M11 · 音频感知',
                '11_音频感知': 'M11 · 音频感知',
                '视频与世界模型': 'M12 · 视频与世界模型',
                '12 视频与世界模型': 'M12 · 视频与世界模型',
                '12_视频与世界模型': 'M12 · 视频与世界模型',
                '预训练范式': 'M13 · 预训练范式',
                '13 预训练范式': 'M13 · 预训练范式',
                '13_预训练范式': 'M13 · 预训练范式',
                '后训练工程': 'M14 · 后训练工程',
                '14 后训练工程': 'M14 · 后训练工程',
                '14_后训练工程': 'M14 · 后训练工程',
                '评估基准': 'M15 · 评估基准',
                '15 评估基准': 'M15 · 评估基准',
                '15_评估基准': 'M15 · 评估基准',
                '强化学习': 'M16 · 强化学习',
                '16 强化学习': 'M16 · 强化学习',
                '16_强化学习': 'M16 · 强化学习',
                '工程陷阱与Harness': 'M17 · 工程陷阱与Harness',
                '17 工程陷阱与Harness': 'M17 · 工程陷阱与Harness',
                '17_工程陷阱与Harness': 'M17 · 工程陷阱与Harness',
                '17_工程陷阱与harness': 'M17 · 工程陷阱与Harness'
            };

            // 彻底清理之前残留的 invalid splitter / resizer 节点与旧样式表（如果存在）
            var oldSplitter = doc.getElementById('nn-sidebar-col-splitter');
            if (oldSplitter && oldSplitter.parentNode) oldSplitter.parentNode.removeChild(oldSplitter);
            var oldResizer = doc.getElementById('nn-sidebar-right-resizer');
            if (oldResizer && oldResizer.parentNode) oldResizer.parentNode.removeChild(oldResizer);

            var oldStyles = doc.querySelectorAll('style#nn-persistent-sidebar-style');
            oldStyles.forEach(function(s) { if (s && s.parentNode) s.parentNode.removeChild(s); });

            // 注入持久化 Head 样式表，保证即使在 Streamlit 页面重载空白期，侧边栏样式也 100% 永久保持
            var headStyle = doc.createElement('style');
            headStyle.id = 'nn-persistent-sidebar-style';
            headStyle.textContent = `
                    /* 1. 侧边栏整体与主导航罗盘 (全站统一 Flex Row 双栏架构) */
                    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                        display: flex !important;
                        flex-direction: row !important;
                        height: 100vh !important;
                        padding: 0 !important;
                        overflow: hidden !important;
                    }

                    /* 栏位 1：主导航航母罗盘 (全站统一 210px 紧凑罗盘，0 抖动，0 高度差) */
                    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
                        width: 210px !important;
                        min-width: 210px !important;
                        max-width: 210px !important;
                        height: 100% !important;
                        overflow-y: auto !important;
                        border-right: 1px solid #e2e8f0 !important;
                        padding: 2.8rem 0.4rem 1.5rem 0.4rem !important;
                        background-color: #f8fafc !important;
                        box-sizing: border-box !important;
                        flex-shrink: 0 !important;
                    }

                    /* 栏位 2：微观超参数控制台 (有内容时展开 410px 充裕舒展空间) */
                    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
                        display: block !important;
                        flex: 1 1 410px !important;
                        width: 410px !important;
                        min-width: 410px !important;
                        max-width: 410px !important;
                        height: 100% !important;
                        overflow-y: auto !important;
                        overflow-x: auto !important;
                        padding: 2.8rem 1.2rem 1.5rem 1.2rem !important;
                        background-color: #ffffff !important;
                        box-sizing: border-box !important;
                    }

                    /* 当栏位 2 无控件时 (如 M00 / 首页)，自动隐藏并收拢为单栏 260px */
                    [data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])),
                    [data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])) > div {
                        width: 260px !important;
                        min-width: 260px !important;
                        max-width: 260px !important;
                        background-color: #f8fafc !important;
                    }

                    [data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])) [data-testid="stSidebarNav"] {
                        width: 100% !important;
                        min-width: 100% !important;
                        max-width: 100% !important;
                        border-right: none !important;
                    }

                    [data-testid="stSidebar"]:not(:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"])) [data-testid="stSidebarUserContent"] {
                        display: none !important;
                        width: 0 !important;
                        min-width: 0 !important;
                        max-width: 0 !important;
                        padding: 0 !important;
                    }

                    /* 当栏位 2 有实际控件时 (M01~M17)，展开为双栏 620px */
                    @media (min-width: 1260px) {
                        [data-testid="stSidebar"]:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]),
                        [data-testid="stSidebar"]:has([data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]) > div {
                            width: 620px !important;
                            min-width: 620px !important;
                            max-width: 620px !important;
                            background-color: #f8fafc !important;
                            border-right: 1px solid #e2e8f0 !important;
                        }
                    }

                    [data-testid="stSidebarNav"] [data-testid="stSidebarNavShowMore"],
                    [data-testid="stSidebarNavSeparator"] {
                        display: none !important;
                    }

                    [data-testid="stSidebarNav"] ul {
                        padding: 0 !important;
                        margin: 0 !important;
                        list-style: none !important;
                        display: flex !important;
                        flex-direction: column !important;
                        gap: 2px !important;
                    }
                    [data-testid="stSidebarNav"] li {
                        padding: 0 !important;
                        margin: 0 !important;
                        list-style: none !important;
                    }
                    [data-testid="stSidebarNav"] a,
                    [data-testid="stSidebarNavLink"] {
                        display: flex !important;
                        align-items: center !important;
                        padding: 0.42rem 0.65rem !important;
                        margin: 0 !important;
                        border-radius: 8px !important;
                        color: #475569 !important;
                        font-size: 0.84rem !important;
                        font-weight: 500 !important;
                        line-height: 1.25 !important;
                        height: 36px !important;
                        min-height: 36px !important;
                        max-height: 36px !important;
                        box-sizing: border-box !important;
                        text-decoration: none !important;
                        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        border: 1px solid transparent !important;
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                    }
                    [data-testid="stSidebarNav"] a span {
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                        white-space: nowrap !important;
                        font-size: 0.84rem !important;
                        line-height: 1.25 !important;
                        font-family: inherit !important;
                    }
                    [data-testid="stSidebarNav"] a:hover {
                        background: #f1f5f9 !important;
                        color: #0f172a !important;
                        transform: translateX(2px) !important;
                    }
                    [data-testid="stSidebarNav"] a[aria-current="page"] {
                        background: #eff6ff !important;
                        color: #1d4ed8 !important;
                        font-weight: 600 !important;
                        border: 1px solid #bfdbfe !important;
                        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08) !important;
                    }
                    [data-testid="stSidebarNav"] a svg {
                        display: none !important;
                    }

                    [data-testid="stSidebarUserContent"] .stMarkdown h3,
                    [data-testid="stSidebarUserContent"] .stMarkdown h4 {
                        font-size: 0.72rem !important;
                        font-weight: 700 !important;
                        text-transform: uppercase !important;
                        letter-spacing: 0.08em !important;
                        color: #475569 !important;
                        margin: 0.9rem 0 0.4rem 0 !important;
                        padding-bottom: 0.25rem !important;
                        border-bottom: 1px solid #e2e8f0 !important;
                    }
                    [data-testid="stSidebarUserContent"] [data-testid="stSlider"],
                    [data-testid="stSidebarUserContent"] [data-testid="stSelectbox"],
                    [data-testid="stSidebarUserContent"] [data-testid="stRadio"],
                    [data-testid="stSidebarUserContent"] [data-testid="stNumberInput"] {
                        background: #f8fafc !important;
                        border: 1px solid #e2e8f0 !important;
                        border-radius: 8px !important;
                        padding: 0.5rem 0.75rem !important;
                        margin-bottom: 0.6rem !important;
                        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02) !important;
                        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
                    }
                    [data-testid="stSidebarUserContent"] label {
                        font-weight: 700 !important;
                        font-size: 0.82rem !important;
                        color: #1e293b !important;
                    }
                `;
                doc.head.appendChild(headStyle);

            var isFormatting = false;
            function updateLayoutAndLabels() {
                if (isFormatting) return;
                isFormatting = true;
                try {
                    var links = doc.querySelectorAll('[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"]');
                    links.forEach(function(a) {
                        var spans = a.querySelectorAll('span');
                        spans.forEach(function(span) {
                            var text = span.textContent.trim();
                            if (labelMap[text] && span.textContent !== labelMap[text]) {
                                span.textContent = labelMap[text];
                            }
                        });
                    });
                } finally {
                    isFormatting = false;
                }
            }

            updateLayoutAndLabels();

            if (!doc.__nnSidebarObserver) {
                var observerTimeout = null;
                var observer = new window.parent.MutationObserver(function() {
                    if (observerTimeout) clearTimeout(observerTimeout);
                    observerTimeout = setTimeout(updateLayoutAndLabels, 60);
                });
                var sidebarEl = doc.querySelector('[data-testid="stSidebar"]') || doc.body;
                observer.observe(sidebarEl, { childList: true, subtree: true });
                doc.__nnSidebarObserver = observer;
            }
        } catch(e) {}
    })();
    </script></body></html>"""
    components.html(sidebar_nav_js, height=1, width=1)


def render_hero_header(
    title: str,
    subtitle: str,
    badge_text: str = "PURE NUMPY · ZERO BLACKBOX",
    badge_type: str = "blue",
) -> None:
    """渲染纯净亮色高对比度 Hero 头部"""
    badge_class = f"pill-{badge_type}"
    html = (
        f'<div style="margin-bottom: 1.4rem;">'
        f'<div style="margin-bottom: 0.5rem;">'
        f'<span class="pill-badge {badge_class}"><span class="status-dot"></span>{badge_text}</span>'
        f"</div>"
        f'<h1 class="hero-title">{title}</h1>'
        f'<div class="hero-subtitle">{subtitle}</div>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_heading(
    title: str, icon_name: str = "activity", subtext: str | None = None
) -> None:
    """渲染带极简矢量图标的分区标题"""
    icon_html = svg_icon(icon_name, size=18, color="#1d4ed8")
    sub_html = (
        f'<div style="color: #64748b; font-size: 0.85rem; margin-top: 0.2rem; font-weight: 500;">{subtext}</div>'
        if subtext
        else ""
    )
    html = (
        f'<div style="margin-top: 1.4rem; margin-bottom: 0.8rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.5rem;">'
        f"{icon_html}"
        f'<span style="font-size: 1.15rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em;">{title}</span>'
        f"</div>"
        f"{sub_html}"
        f"</div>"
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
        f"{delta_html}"
        f"</div>"
    )


def render_preset_badge(preset_name: str, desc: str) -> None:
    """渲染预设方案的说明卡片 (纯矢量高精风格)"""
    icon_zap = svg_icon("zap", size=14, color="#1d4ed8")
    html = (
        f'<div style="background: #eff6ff; border-left: 4px solid #1d4ed8; padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; font-size: 0.88rem; color: #1e3a8a; border: 1px solid #dbeafe; border-left-width: 4px;">'
        f'<div style="display: flex; align-items: center; gap: 0.4rem; font-weight: 700; color: #1d4ed8; margin-bottom: 0.2rem;">'
        f"{icon_zap} [PRESET // 预设方案]: {preset_name}"
        f"</div>"
        f'<span style="color: #334155;">{desc}</span>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_page_guide(
    title: str,
    plain_intro: str,
    hyperparams_desc: str,
    telemetry_desc: str,
    experiments: list[str],
    blueprint_sections: list[dict] | None = None,
    guide_region_id: str = "region-b",
) -> None:
    """
    渲染全站统一的高对比度零基础教学指引卡片。
    包含：
    1. [SPATIAL BLUEPRINT] 页面空间交互地图（可选微缩拓扑）
    2. [CORE PRINCIPLE] 通俗原理解析与空间指代
    3. [PARAMETERS VS TELEMETRY] 输入超参数与输出指标的精确色彩映射
    4. [LAB EXPERIMENTS] 结构化探索任务清单
    """
    svg_icon("compass", size=15, color="#1d4ed8")
    icon_bulb = svg_icon("lightbulb", size=15, color="#1d4ed8")
    icon_sliders = svg_icon("sliders", size=14, color="#b45309")
    icon_target = svg_icon("target", size=14, color="#047857")
    icon_terminal = svg_icon("terminal", size=14, color="#0f172a")

    with st.expander(f"[GROWTH GUIDE // 教学指引] {title}", expanded=True):
        render_interactive_region_header(
            guide_region_id,
            "TEACHING GUIDE // 教学指引与实验路径",
            "B",
            "blue",
            "从页面地图进入原理、参数、指标与结构化实验说明",
        )
        # 1. 渲染空间微缩地图 (如果提供)
        if blueprint_sections:
            render_page_blueprint(blueprint_sections)

        items_html = "".join(
            [
                f'<li style="margin-bottom:0.45rem;line-height:1.65;">{exp}</li>'
                for exp in experiments
            ]
        )
        guide_html = (
            f'<div style="color:#0f172a;font-size:0.92rem;line-height:1.7;">'
            f'<div style="background:#ffffff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;padding:0.9rem 1.15rem;border-radius:8px;margin-bottom:1rem;box-shadow:0 2px 10px rgba(37,99,235,0.04);">'
            f'<div style="display:flex;align-items:center;gap:0.45rem;font-weight:800;color:#1e40af;margin-bottom:0.35rem;">'
            f'{icon_bulb} <span style="text-transform:uppercase;font-size:0.8rem;letter-spacing:0.04em;">[CORE PRINCIPLE // 核心原理解析与空间导引]</span>'
            f"</div>"
            f'<div style="color:#1e293b;line-height:1.75;">{plain_intro}</div>'
            f"</div>"
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">'
            f'<div style="background:#ffffff;padding:0.95rem 1.15rem;border-radius:8px;border:1px solid #fde68a;border-left:3.5px solid #b45309;box-shadow:0 2px 8px rgba(180,83,9,0.03);">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;font-weight:800;color:#92400e;font-size:0.84rem;margin-bottom:0.35rem;">'
            f'{icon_sliders} <span style="text-transform:uppercase;letter-spacing:0.04em;">[INPUT CONTROLS // 控制台输入参数]</span>'
            f"</div>"
            f'<div style="font-size:0.84rem;color:#475569;line-height:1.65;">{hyperparams_desc}</div>'
            f"</div>"
            f'<div style="background:#ffffff;padding:0.95rem 1.15rem;border-radius:8px;border:1px solid #a7f3d0;border-left:3.5px solid #047857;box-shadow:0 2px 8px rgba(4,120,87,0.03);">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;font-weight:800;color:#065f46;font-size:0.84rem;margin-bottom:0.35rem;">'
            f'{icon_target} <span style="text-transform:uppercase;letter-spacing:0.04em;">[TELEMETRY // 模型学习遥测成果]</span>'
            f"</div>"
            f'<div style="font-size:0.84rem;color:#475569;line-height:1.65;">{telemetry_desc}</div>'
            f"</div>"
            f"</div>"
            f'<div style="background:#ffffff;padding:0.95rem 1.15rem;border-radius:8px;border:1px solid #cbd5e1;border-left:3.5px solid #0f172a;box-shadow:0 2px 8px rgba(15,23,42,0.03);">'
            f'<div style="display:flex;align-items:center;gap:0.4rem;font-weight:800;color:#0f172a;font-size:0.86rem;margin-bottom:0.5rem;">'
            f'{icon_terminal} <span style="text-transform:uppercase;letter-spacing:0.04em;">[LAB EXPERIMENTS // 结构化探索任务]</span>'
            f"</div>"
            f'<ol style="margin:0;padding-left:1.3rem;color:#334155;font-size:0.84rem;">{items_html}</ol>'
            f"</div>"
            f"</div>"
        )
        st.markdown(guide_html, unsafe_allow_html=True)


def render_sequence_flow(tokens: list[str], hidden_states: list) -> None:
    """渲染 RNN 时序流动流水线（纯矢量、严格单行无缩进 HTML）"""
    import numpy as np

    items = []
    for i, token in enumerate(tokens):
        h_norm = float(np.linalg.norm(hidden_states[i])) if len(hidden_states) > i else 1.0
        alpha = min(0.9, max(0.2, h_norm / 4.0))
        tag_bg = f"rgba(29, 78, 216, {alpha:.3f})"
        items.append(
            f'<div style="flex:1;min-width:72px;background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:0.55rem 0.35rem;text-align:center;box-shadow:0 2px 4px rgba(15,23,42,0.03);">'
            f'<div style="font-size:0.68rem;color:#64748b;font-weight:700;text-transform:uppercase;">STEP {i}</div>'
            f"<div style=\"font-size:0.95rem;font-weight:800;color:#0f172a;margin:0.2rem 0;font-family:'JetBrains Mono';\">{token}</div>"
            f'<div style="background:{tag_bg};color:#ffffff;font-size:0.68rem;font-weight:700;border-radius:4px;padding:0.12rem 0.25rem;">||h||={h_norm:.2f}</div>'
            f"</div>"
        )
    html = f'<div style="display:flex;gap:0.4rem;overflow-x:auto;padding:0.5rem 0;margin-bottom:1.2rem;">{"".join(items)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_vector_equation_card(word_a: str, word_b: str, word_c: str, best_match: str) -> None:
    """渲染语义向量方程卡片（纯矢量、严格单行无缩进 HTML）"""
    icon_math = svg_icon("crosshair", size=14, color="#1d4ed8")
    icon_info = svg_icon("info", size=14, color="#64748b")
    html = (
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:1.3rem;box-shadow:0 2px 8px rgba(15,23,42,0.03);">'
        f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:0.7rem;">'
        f"{icon_math} [PARALLELOGRAM THEOREM // 语义平行四边形矢量公式]"
        f"</div>"
        f"<div style=\"background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:0.9rem;text-align:center;font-family:'JetBrains Mono', monospace;font-size:1.05rem;margin-bottom:0.9rem;\">"
        f'<span style="color:#1d4ed8;font-weight:800;">{word_a}</span><span style="color:#64748b;"> - </span><span style="color:#be123c;font-weight:800;">{word_b}</span><span style="color:#64748b;"> + </span><span style="color:#047857;font-weight:800;">{word_c}</span><span style="color:#64748b;font-weight:800;"> ≈ </span><span style="color:#7c3aed;font-weight:800;background:#f3e8ff;padding:0.2rem 0.5rem;border-radius:4px;">{best_match}</span>'
        f"</div>"
        f'<div style="font-size:0.84rem;color:#475569;line-height:1.6;display:flex;gap:0.4rem;align-items:flex-start;">'
        f"<span>{icon_info}</span>"
        f"<span><b>几何原理解析</b>：向量位移差 <code>{word_a} - {word_b}</code> 提取了纯粹的概念维度偏置；在流形空间中叠加到 <code>{word_c}</code> 上，高精定向导航至 <code>{best_match}</code>。</span>"
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_architecture_flow_card() -> None:
    """渲染 Pre-LN Transformer 结构块流程卡片（纯矢量、严格单行无缩进 HTML）"""
    icon_cpu = svg_icon("cpu", size=14, color="#1d4ed8")
    html = (
        f"<div style=\"background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem;font-family:'JetBrains Mono', monospace;font-size:0.84rem;line-height:1.6;color:#0f172a;box-shadow:0 2px 8px rgba(15,23,42,0.03);\">"
        f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:0.8rem;">'
        f"{icon_cpu} [PIPELINE TOPOLOGY // 残差主干流拓扑]"
        f"</div>"
        f'<div style="display:flex;justify-content:space-around;align-items:center;flex-wrap:wrap;gap:0.8rem;">'
        f'<div style="background:#eff6ff;border:1px solid #bfdbfe;padding:0.7rem 1rem;border-radius:8px;text-align:center;"><span style="color:#1d4ed8;font-weight:800;">INPUT STREAM x</span><br><span style="font-size:0.72rem;color:#64748b;">(残差流主干)</span></div>'
        f'<div style="color:#64748b;font-size:1.3rem;"></div>'
        f'<div style="background:#f8fafc;border:1px solid #cbd5e1;padding:0.7rem 1rem;border-radius:8px;text-align:center;"><span style="color:#7c3aed;font-weight:800;">LayerNorm₁</span>  <span style="color:#2563eb;font-weight:800;">MHA</span><br><span style="color:#059669;font-weight:700;">x = x + MHA(LN₁(x))</span></div>'
        f'<div style="color:#64748b;font-size:1.3rem;"></div>'
        f'<div style="background:#f8fafc;border:1px solid #cbd5e1;padding:0.7rem 1rem;border-radius:8px;text-align:center;"><span style="color:#7c3aed;font-weight:800;">LayerNorm₂</span>  <span style="color:#b45309;font-weight:800;">GELU FFN</span><br><span style="color:#059669;font-weight:700;">x = x + FFN(LN₂(x))</span></div>'
        f'<div style="color:#64748b;font-size:1.3rem;"></div>'
        f'<div style="background:#ecfdf5;border:1px solid #a7f3d0;padding:0.7rem 1rem;border-radius:8px;text-align:center;"><span style="color:#047857;font-weight:800;">OUTPUT STREAM x</span><br><span style="font-size:0.72rem;color:#64748b;">(进入下一层)</span></div>'
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_text_stream_box(tokens: list[str], prompt_len: int) -> None:
    """渲染 GPT 文本流（纯矢量、严格单行无缩进 HTML）"""
    icon_terminal = svg_icon("terminal", size=14, color="#1d4ed8")
    icon_sparkle = svg_icon("sparkles", size=12, color="#1d4ed8")
    badges = []
    for idx, tok in enumerate(tokens):
        if idx < prompt_len:
            badges.append(
                f"<span style=\"background:#f1f5f9;color:#0f172a;border:1px solid #cbd5e1;padding:0.22rem 0.55rem;border-radius:6px;font-weight:700;font-family:'JetBrains Mono';\">{tok}</span>"
            )
        else:
            badges.append(
                f"<span style=\"background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd;padding:0.22rem 0.55rem;border-radius:6px;font-weight:800;font-family:'JetBrains Mono';box-shadow:0 2px 6px rgba(37,99,235,0.12);\">{icon_sparkle} {tok}</span>"
            )

    html = (
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:1.3rem;line-height:2.2;box-shadow:0 2px 8px rgba(15,23,42,0.03);margin-bottom:1.2rem;">'
        f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:0.74rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:0.5rem;">'
        f"{icon_terminal} [CONTEXT WINDOW // 上下文滑动窗口]:"
        f"</div>"
        f'<div style="display:flex;flex-wrap:wrap;gap:0.45rem;align-items:center;">'
        f"{' '.join(badges)}"
        f'<span style="display:inline-block;width:7px;height:16px;background:#1d4ed8;"></span>'
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_live_param_status_bar(
    title: str = "LIVE MATHEMATICAL PARAMETERS // 实时微观参数与数学状态",
    badges: list[dict[str, str]] | None = None,
    metrics: list[tuple[str, str]] | None = None,
    tag: str | None = None,
    tag_color: str = "emerald",
) -> None:
    """
    渲染全站统一的现代 Linear/Stripe 风格实时数学参数与动力学遥测状态栏。

    参数：
    - title: 状态栏左侧标题（中英双语）
    - badges: 高亮数学徽标列表，每个元素形如 {"label": "w₁₁", "value": "0.324", "color": "blue"|"amber"|"emerald"|"purple"|"rose"}
    - metrics: 键值对指标列表，每个元素形如 ("权重均值 ‖W‖", "0.4512")
    - tag: 右侧状态标签（如 "TOTAL PARAMS: 120" 或 "STATUS: CONVERGED"）
    - tag_color: 标签主题色 ("emerald"|"blue"|"amber"|"purple"|"rose")
    """
    color_map = {
        "blue": ("#eff6ff", "#bfdbfe", "#1d4ed8"),
        "amber": ("#fffbeb", "#fde68a", "#92400e"),
        "emerald": ("#ecfdf5", "#a7f3d0", "#047857"),
        "purple": ("#f5f3ff", "#ddd6fe", "#6d28d9"),
        "rose": ("#fff1f2", "#fecdd3", "#be123c"),
    }

    badges_html = []
    if badges:
        for b in badges:
            lbl = b.get("label", "")
            val = b.get("value", "")
            c = b.get("color", "blue")
            bg_c, border_c, text_c = color_map.get(c, color_map["blue"])
            badges_html.append(
                f"<span style=\"background:{bg_c};color:{text_c};border:1px solid {border_c};padding:0.1rem 0.42rem;border-radius:4px;font-weight:700;font-family:'JetBrains Mono', monospace;\">"
                f"{lbl} = {val}"
                f"</span>"
            )

    metrics_html = []
    if metrics:
        for k, v in metrics:
            metrics_html.append(
                f'<span style="color:#64748b;font-family:\'JetBrains Mono\', monospace;">| {k}: <b style="color:#0f172a;">{v}</b></span>'
            )

    tag_html = ""
    if tag:
        t_bg, t_border, t_text = color_map.get(tag_color, color_map["emerald"])
        tag_html = (
            f"<div style=\"color:{t_text};font-weight:700;background:{t_bg};border:1px solid {t_border};padding:0.12rem 0.45rem;border-radius:4px;font-family:'JetBrains Mono', monospace;font-size:0.72rem;letter-spacing:0.02em;\">"
            f"{tag}"
            f"</div>"
        )

    html = (
        f"<div style=\"background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.75rem;font-family:'JetBrains Mono', -apple-system, sans-serif;font-size:0.76rem;color:#1e293b;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;box-shadow:0 1px 3px rgba(15,23,42,0.02);\">"
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.45rem;">'
        f'<span style="color:#1e40af;font-weight:800;letter-spacing:0.02em;">[{title}]</span>'
        f"{''.join(badges_html)}"
        f"{''.join(metrics_html)}"
        f"</div>"
        f"{tag_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def anchor_badge(text: str, color_type: str = "blue", target_id: str | None = None) -> str:
    """
    生成带空间语义色彩的高对比度胶囊徽标 HTML。
    若指定 target_id，徽标将自动变为可点击穿梭平滑滚动的超链接锚点，并激发目标容器聚光灯高光！
    """
    if target_id:
        return f'<a href="#{target_id}" class="anchor-badge anchor-badge-{color_type}">{text}</a>'
    return f'<span class="anchor-badge anchor-badge-{color_type}">{text}</span>'


def render_region_anchor(target_id: str) -> None:
    """在组件正上方注入平滑滚动停靠锚点"""
    st.markdown(
        f'<div id="{target_id}" style="position:relative;top:-30px;visibility:hidden;height:0;margin:0;padding:0;"></div>',
        unsafe_allow_html=True,
    )


def render_interactive_region_header(
    region_id: str,
    title: str,
    badge_letter: str,
    badge_color: str = "blue",
    subtext: str | None = None,
) -> None:
    """
    渲染带空间锚点与低干扰聚焦反馈的目标区域头部（纯 SVG 矢量图标）。
    点击导航直达时，此区域会显示柔和入场光晕与“正在查看这里”定位标签。
    """
    sub_html = (
        f'<div style="color:#64748b;font-size:0.84rem;margin-top:0.25rem;font-weight:500;">{subtext}</div>'
        if subtext
        else ""
    )
    badge_html = anchor_badge(badge_letter, badge_color)
    html = (
        f'<div id="{region_id}" class="interactive-region" style="scroll-margin-top:75px;padding:0.6rem 0.85rem;margin-top:1.2rem;margin-bottom:0.6rem;border-radius:10px;border:1px solid #e2e8f0;background:#ffffff;">'
        f'<div style="display:flex;align-items:center;gap:0.45rem;">'
        f"{badge_html}"
        f'<span style="font-size:1.02rem;font-weight:800;color:#0f172a;letter-spacing:-0.01em;">{title}</span>'
        f"</div>"
        f"{sub_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_page_blueprint(sections: list[dict]) -> None:
    """
    渲染页面空间交互蓝图微缩地图 (Page Spatial Blueprint)。
    每个区域卡片均可直接点击平滑穿梭至目标图表/控制台，并触发低干扰的聚焦提示。
    """
    icon_map = svg_icon("target", size=14, color="#1d4ed8")
    icon_sparkle = svg_icon("sparkles", size=12, color="#1d4ed8")

    blocks_html = []
    color_border_map = {
        "amber": ("#fffbeb", "#fde68a", "#92400e"),
        "blue": ("#eff6ff", "#bfdbfe", "#1e40af"),
        "emerald": ("#ecfdf5", "#a7f3d0", "#065f46"),
        "purple": ("#f5f3ff", "#ddd6fe", "#5b21b6"),
        "rose": ("#fff1f2", "#fecdd3", "#9f1239"),
    }

    normalized_sections = []
    for sec in sections:
        sec_id = sec.get("id") or sec.get("letter") or "A"
        sec_name = sec.get("name") or sec.get("title") or ""
        sec_desc = sec.get("desc") or sec.get("role") or ""
        sec_color = sec.get("color", "blue")
        target_id = sec.get("target_id") or f"region-{sec_id.lower()}"
        bg_c, border_c, text_c = color_border_map.get(sec_color, color_border_map["blue"])
        normalized_sections.append(
            {
                "id": sec_id,
                "name": sec_name,
                "desc": sec_desc,
                "color": sec_color,
                "target_id": target_id,
            }
        )

        block = (
            f'<a href="#{target_id}" style="text-decoration:none;color:inherit;flex:1 1 180px;min-width:150px;">'
            f'<div style="background:{bg_c};border:1px solid {border_c};border-radius:8px;padding:0.6rem 0.8rem;cursor:pointer;">'
            f'<div style="display:flex;align-items:center;gap:0.35rem;margin-bottom:0.25rem;">'
            f'<span class="anchor-badge anchor-badge-{sec_color}" style="font-size:0.72rem;padding:0.1rem 0.35rem;">[{sec_id}]</span>'
            f'<span style="font-weight:800;font-size:0.82rem;color:{text_c};">{sec_name}</span>'
            f"</div>"
            f'<div style="font-size:0.74rem;color:#475569;line-height:1.4;">{sec_desc}</div>'
            f"</div>"
            f"</a>"
        )
        blocks_html.append(block)

    html = (
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(15,23,42,0.03);">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:0.74rem;font-weight:800;color:#1e40af;letter-spacing:0.04em;text-transform:uppercase;">'
        f"{icon_map} [SPATIAL BLUEPRINT // 页面空间交互地图 (点击卡片直达物理区域)]"
        f"</div>"
        f'<div style="font-size:0.74rem;font-weight:700;color:#1d4ed8;display:flex;align-items:center;gap:0.3rem;">{icon_sparkle} [CLICK TO FOCUS // 点击聚焦定位]</div>'
        f"</div>"
        f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem;">'
        f"{''.join(blocks_html)}"
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    render_floating_hud_navigator(normalized_sections)


def render_floating_hud_navigator(sections: list[dict]) -> None:
    """使用 st.iframe 在宿主视窗右侧挂载常驻悬浮微缩罗盘 HUD。"""
    normalized = []
    for s in sections:
        s_id = s.get("id") or s.get("letter") or "A"
        s_name = s.get("name") or s.get("title") or ""
        s_desc = s.get("desc") or s.get("role") or ""
        s_color = s.get("color", "blue")
        s_target = s.get("target_id") or f"region-{s_id.lower()}"
        normalized.append(
            {
                "id": s_id,
                "name": s_name,
                "desc": s_desc,
                "color": s_color,
                "target_id": s_target,
            }
        )
    sec_json = json.dumps(normalized, ensure_ascii=False)
    js = f"""<!doctype html><html><head><meta charset="utf-8"></head><body>
    <script>
    (function() {{
        try {{
            var doc = window.parent.document;
            if (!doc) return;

            var oldHud = doc.getElementById('nn-floating-spatial-hud');
            if (oldHud) oldHud.remove();

            var sections = {sec_json};
            if (!sections || sections.length === 0) return;

            doc.__nnIsNavigating = false;

            function setActiveItem(targetId) {{
                doc.querySelectorAll('.nn-hud-item').forEach(function(el) {{
                    if (el.getAttribute('data-target') === targetId) {{
                        el.classList.add('active');
                        el.style.background = '#eff6ff';
                        el.style.color = '#1d4ed8';
                        el.style.fontWeight = '800';
                        el.style.borderLeft = '3px solid #2563eb';
                    }} else {{
                        el.classList.remove('active');
                        el.style.background = 'transparent';
                        el.style.color = '#334155';
                        el.style.fontWeight = '600';
                        el.style.borderLeft = 'none';
                    }}
                }});
            }}

            function clearFocus() {{
                doc.querySelectorAll('.nn-focus-target').forEach(function(node) {{
                    node.classList.remove('nn-focus-target');
                }});
                doc.querySelectorAll('.nn-focus-chip').forEach(function(node) {{ node.remove(); }});
            }}

            function focusRegion(target) {{
                var targetId = typeof target === 'string' ? target : (target ? target.id : '');
                var el = typeof target === 'string' ? doc.getElementById(target) : target;
                if (!el) return false;
                clearFocus();

                if (targetId) {{
                    setActiveItem(targetId);
                    doc.__nnIsNavigating = true;
                    if (doc.__nnNavTimer) window.parent.clearTimeout(doc.__nnNavTimer);
                    doc.__nnNavTimer = window.parent.setTimeout(function() {{
                        doc.__nnIsNavigating = false;
                        updateActiveSection();
                    }}, 900);
                }}

                var reducedMotion = window.parent.matchMedia &&
                    window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;
                el.scrollIntoView({{ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' }});
                el.classList.remove('nn-focus-target');
                void el.offsetWidth;
                el.classList.add('nn-focus-target');
                var chip = doc.createElement('span');
                chip.className = 'nn-focus-chip';
                chip.setAttribute('role', 'status');
                chip.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#38bdf8;margin-right:7px;box-shadow:0 0 8px #38bdf8;vertical-align:middle;"></span>正在查看此板块';
                chip.style.cssText = 'position:absolute !important; left:0.85rem !important; right:auto !important; top:-0.9rem !important; z-index:9999 !important; padding:0.28rem 0.85rem !important; border:2px solid #38bdf8 !important; border-radius:999px !important; background:#0f172a !important; box-shadow:0 8px 25px rgba(15, 23, 42, 0.45), 0 0 0 3px rgba(56, 189, 248, 0.35) !important; color:#ffffff !important; font-size:0.76rem !important; font-weight:800 !important; letter-spacing:0.04em !important; pointer-events:none !important;';
                el.appendChild(chip);
                window.parent.clearTimeout(doc.__nnFocusTimer);
                doc.__nnFocusTimer = window.parent.setTimeout(
                    clearFocus, reducedMotion ? 2000 : 4200
                );
                return true;
            }}
            doc.__nnFocusRegion = focusRegion;

            function bindCourseAnchors() {{
                doc.querySelectorAll('a[href*="#region-"]').forEach(function(anchor) {{
                    if (anchor.getAttribute('data-nn-focus-bound') === '1') return;
                    anchor.setAttribute('data-nn-focus-bound', '1');
                    anchor.addEventListener('click', function(e) {{
                        var href = anchor.getAttribute('href');
                        if (!href) return;
                        e.preventDefault();
                        e.stopPropagation();
                        e.stopImmediatePropagation();
                        focusRegion(href.substring(href.indexOf('#') + 1));
                    }}, true);
                }});
            }}
            bindCourseAnchors();
            window.parent.setTimeout(bindCourseAnchors, 300);
            window.parent.setTimeout(bindCourseAnchors, 900);

            var hud = doc.createElement('div');
            hud.id = 'nn-floating-spatial-hud';
            hud.style.cssText = 'position:fixed;right:20px;top:115px;z-index:999999;background:rgba(255,255,255,0.92);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid #cbd5e1;border-radius:12px;box-shadow:0 10px 25px -5px rgba(15,23,42,0.08),0 4px 6px -2px rgba(15,23,42,0.04);padding:0.65rem 0.75rem;width:178px;font-family:JetBrains Mono,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:0.75rem;transition:all 0.2s cubic-bezier(0.4,0,0.2,1);';
            if (window.parent.innerWidth < 1280) hud.style.display = 'none';

            var header = doc.createElement('div');
            header.style.cssText = 'font-size:0.68rem;font-weight:800;color:#64748b;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.45rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #f1f5f9;padding-bottom:0.3rem;';
            header.innerHTML = '<span style="display:inline-flex;align-items:center;gap:4px;"><span style="color:#2563eb;font-weight:800;">[NAV]</span> SPATIAL HUD</span><span style="font-size:0.62rem;color:#3b82f6;font-weight:800;background:#eff6ff;border:1px solid #bfdbfe;padding:1px 4px;border-radius:3px;">LIVE</span>';
            hud.appendChild(header);

            var list = doc.createElement('div');
            list.style.cssText = 'display:flex;flex-direction:column;gap:0.25rem;';

            sections.forEach(function(sec) {{
                var secId = sec.id || sec.letter || 'A';
                var secName = sec.name || sec.title || '';
                var targetId = sec.target_id || ('region-' + secId.toLowerCase());
                var item = doc.createElement('a');
                item.href = '#' + targetId;
                item.className = 'nn-hud-item';
                item.setAttribute('data-target', targetId);
                item.style.cssText = 'display:flex;align-items:center;gap:0.4rem;padding:0.3rem 0.45rem;border-radius:6px;text-decoration:none;color:#334155;font-weight:600;transition:all 0.15s ease;cursor:pointer;';
                item.innerHTML = '<span style="font-family:monospace;font-size:0.72rem;font-weight:800;color:#2563eb;background:#eff6ff;border:1px solid #bfdbfe;padding:0.05rem 0.3rem;border-radius:4px;">[' + secId + ']</span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:115px;">' + secName + '</span>';

                item.addEventListener('mouseenter', function() {{
                    if (!this.classList.contains('active')) {{
                        this.style.background = '#f8fafc';
                        this.style.transform = 'translateX(-2px)';
                    }}
                }});
                item.addEventListener('mouseleave', function() {{
                    if (!this.classList.contains('active')) {{
                        this.style.background = 'transparent';
                        this.style.transform = 'none';
                    }}
                }});

                item.addEventListener('click', function(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    focusRegion(targetId);
                }});

                list.appendChild(item);
            }});

            hud.appendChild(list);

            var topBtn = doc.createElement('div');
            topBtn.style.cssText = 'margin-top:0.45rem;padding-top:0.35rem;border-top:1px solid #f1f5f9;text-align:center;color:#64748b;cursor:pointer;font-size:0.7rem;font-weight:700;transition:color 0.15s;';
            topBtn.innerHTML = '↑ 回到顶部 (Top)';
            topBtn.addEventListener('mouseenter', function() {{ this.style.color = '#2563eb'; }});
            topBtn.addEventListener('mouseleave', function() {{ this.style.color = '#64748b'; }});
            topBtn.addEventListener('click', function() {{
                var main = doc.querySelector('[data-testid="stMain"]');
                var reducedMotion = window.parent.matchMedia &&
                    window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;
                if (main) main.scrollTo({{ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' }});
                else window.parent.scrollTo({{ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' }});
            }});
            hud.appendChild(topBtn);

            doc.body.appendChild(hud);

            function updateActiveSection() {{
                if (doc.__nnIsNavigating) return;

                var main = doc.querySelector('[data-testid="stMain"]');
                var viewH = (main ? main.clientHeight : window.parent.innerHeight) || 800;
                var focalLine = viewH * 0.45;

                // 页面触底保护：如果已经滚动到底部，直接高亮最后一项
                if (main && (main.scrollTop + main.clientHeight >= main.scrollHeight - 40)) {{
                    var lastSec = sections[sections.length - 1];
                    if (lastSec) {{
                        setActiveItem(lastSec.target_id || ('region-' + (lastSec.id || '').toLowerCase()));
                        return;
                    }}
                }}

                var currentActive = null;

                for (var i = 0; i < sections.length; i++) {{
                    var sec = sections[i];
                    var targetId = sec.target_id || ('region-' + (sec.id || '').toLowerCase());
                    var el = doc.getElementById(targetId);
                    if (!el) continue;

                    // 忽略侧边栏中的固定元素（如 region-a）
                    if (el.closest && el.closest('[data-testid="stSidebar"]')) {{
                        continue;
                    }}

                    var rect = el.getBoundingClientRect();
                    // 标准 ScrollSpy 判定：顶部到达或穿过焦点线（+50px 容差）
                    if (rect.top <= focalLine + 50) {{
                        currentActive = targetId;
                    }}
                }}

                if (!currentActive) {{
                    currentActive = sections[0].target_id || ('region-' + (sections[0].id || '').toLowerCase());
                }}

                if (currentActive) {{
                    setActiveItem(currentActive);
                }}
            }}

            var scrollContainer = doc.querySelector('[data-testid="stMain"]') || window.parent;
            if (doc.__nnScrollListener && doc.__nnScrollContainer) {{
                doc.__nnScrollContainer.removeEventListener('scroll', doc.__nnScrollListener);
            }}
            doc.__nnScrollListener = updateActiveSection;
            doc.__nnScrollContainer = scrollContainer;
            scrollContainer.addEventListener('scroll', updateActiveSection, {{ passive: true }});

            setTimeout(updateActiveSection, 300);
            setTimeout(updateActiveSection, 800);

        }} catch(e) {{
            console.error('Floating HUD error:', e);
        }}
    }})();
    </script></body></html>
    """
    components.html(js, height=1, width=1)


def render_formula_breakdown_card(
    formula_latex: str,
    math_principle: str,
    params_breakdown: list[dict],
) -> None:
    """渲染公式拆解卡片，展示 LaTeX 公式、数学原理与参数拆解表格。"""
    rows_html = "".join(
        [
            f'<tr><td style="padding:6px 10px;font-family:monospace;font-weight:700;color:#1e40af;border-bottom:1px solid #e2e8f0;">{p.get("param", "")}</td>'
            f'<td style="padding:6px 10px;font-family:monospace;color:#64748b;border-bottom:1px solid #e2e8f0;">{p.get("shape", "")}</td>'
            f'<td style="padding:6px 10px;color:#334155;border-bottom:1px solid #e2e8f0;">{p.get("role", "")}</td></tr>'
            for p in params_breakdown
        ]
    )

    card_html = (
        f'<div style="background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;padding:0.9rem 1.1rem;margin:0.8rem 0;box-shadow:0 2px 6px rgba(0,0,0,0.03);">'
        f'<div style="font-size:0.76rem;font-weight:800;color:#2563eb;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.4rem;">[FORMULA CONTRACT // 数学公式与参数拆解]</div>'
        f'<div style="text-align:center;padding:0.5rem;background:#f8fafc;border-radius:6px;margin-bottom:0.6rem;font-size:1.1rem;">$${formula_latex}$$</div>'
        f'<div style="font-size:0.85rem;color:#475569;margin-bottom:0.7rem;line-height:1.5;">{math_principle}</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;text-align:left;">'
        f'<thead><tr style="background:#f1f5f9;color:#475569;">'
        f'<th style="padding:6px 10px;border-bottom:2px solid #cbd5e1;">参数符号</th>'
        f'<th style="padding:6px 10px;border-bottom:2px solid #cbd5e1;">张量形状 (Shape)</th>'
        f'<th style="padding:6px 10px;border-bottom:2px solid #cbd5e1;">物理角色与维度意义</th>'
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)
