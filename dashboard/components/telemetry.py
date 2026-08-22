# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.telemetry - 全链路可观测性与实时 Trace 探针前端组件

提供实时系统日志、TraceID 过滤、异常堆栈回溯与运行时诊断控制台。
"""

from typing import Any

import streamlit as st

from nn_core.observability import GLOBAL_RING_BUFFER, get_trace_id


def render_live_log_drawer() -> None:
    """
    渲染可折叠的世界级全链路 TraceID 实时可观测性抽屉。
    允许学习者与开发者在页面底部直接检查系统实时日志、执行耗时与 Trace 调用流。
    """
    current_tid = get_trace_id()

    with st.expander(
        f"[TELEMETRY // 全链路 Trace 实时日志与运行时诊断探针 (当前 TraceID: {current_tid})]",
        expanded=False,
    ):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            level_filter = st.selectbox(
                "日志级别过滤",
                ["ALL (全部)", "INFO (常规信息)", "WARN (警告及以上)", "ERROR (错误及以上)"],
                index=0,
                key="telemetry_log_level",
            )
        with c2:
            trace_search = st.text_input(
                "TraceID / 关键词搜索",
                value="",
                placeholder="输入 TraceID 或模块名过滤...",
                key="telemetry_trace_search",
            )
        with c3:
            limit = st.slider(
                "显示条数", min_value=10, max_value=200, value=50, step=10, key="telemetry_limit"
            )
        with c4:
            st.write("")
            st.write("")
            if st.button("清空缓冲区", key="telemetry_clear_btn"):
                GLOBAL_RING_BUFFER.clear()
                st.rerun()

        level_map = {
            "ALL (全部)": "DEBUG",
            "INFO (常规信息)": "INFO",
            "WARN (警告及以上)": "WARN",
            "ERROR (错误及以上)": "ERROR",
        }
        min_lvl = level_map.get(level_filter, "DEBUG")
        search_kw = trace_search.strip() or None

        logs: list[dict[str, Any]] = GLOBAL_RING_BUFFER.get_recent_logs(
            limit=limit,
            trace_id=search_kw,
            min_level=min_lvl,
        )

        if not logs:
            st.info("暂无符合过滤条件的系统日志记录。")
            return

        # 格式化终端输出
        lines = []
        for item in logs:
            lvl = item["level"].upper()
            color_tag = "[INFO ]"
            if lvl == "DEBUG":
                color_tag = "[DEBUG]"
            elif lvl in ("WARN", "WARNING"):
                color_tag = "[WARN ]"
            elif lvl in ("ERROR", "CRITICAL", "FATAL"):
                color_tag = "[ERROR]"

            line = f"{item['timestamp']} {color_tag} [{item['trace_id']}] [{item['logger_name']}:{item['lineno']}] {item['message']}"
            if item.get("exc_text"):
                line += f"\n    [STACK] {item['exc_text']}"
            lines.append(line)

        log_content = "\n".join(lines)
        st.text_area(
            "实时日志流 (倒序呈现最新记录)",
            value=log_content,
            height=260,
            key="telemetry_log_area",
            disabled=True,
        )

        # 统计摘要徽章
        info_count = sum(1 for x in logs if x["level"] == "INFO")
        warn_count = sum(1 for x in logs if x["level"] in ("WARN", "WARNING"))
        err_count = sum(1 for x in logs if x["level"] in ("ERROR", "CRITICAL", "FATAL"))

        st.caption(
            f"缓冲区状态: 总展示 {len(logs)} 条日志 | INFO: {info_count} | WARN: {warn_count} | ERROR: {err_count} | 全局日志文件归档于: logs/app.log 与 logs/error.log"
        )
