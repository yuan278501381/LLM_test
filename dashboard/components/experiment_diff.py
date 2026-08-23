# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.experiment_diff - 双实验并排差分消融对比套件 (Side-by-Side Diff)

提供工业级模型训练与超参数探索的对照消融分析工具：
- 一键捕获当前超参数、指标与损失曲线为基准 (Run A Baseline)
- 动态并排比对 Run A vs Run B 超参数变更矩阵
- 同轴双轨收敛图 (Overlay Convergence Traces)
- Delta 指标增益/衰减量化卡片
"""

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from dashboard.styles.icons import svg_icon


def get_baseline(module_id: str) -> dict[str, Any] | None:
    """获取指定模块的已存基准实验"""
    all_baselines = st.session_state.get("nn_experiment_baselines", {})
    return all_baselines.get(module_id)


def save_baseline(
    module_id: str,
    name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    loss_history: list[float] | None = None,
) -> None:
    """保存当前实验为基准 Run A"""
    if "nn_experiment_baselines" not in st.session_state:
        st.session_state["nn_experiment_baselines"] = {}

    st.session_state["nn_experiment_baselines"][module_id] = {
        "name": name,
        "params": params,
        "metrics": metrics,
        "loss_history": list(loss_history) if loss_history else [],
    }


def clear_baseline(module_id: str) -> None:
    """清除指定模块的基准"""
    if "nn_experiment_baselines" in st.session_state:
        st.session_state["nn_experiment_baselines"].pop(module_id, None)


def render_experiment_diff_controller(
    module_id: str,
    current_name: str,
    current_params: dict[str, Any],
    current_metrics: dict[str, Any],
    current_loss_history: list[float] | None = None,
) -> None:
    """
    在工作区渲染双实验差分对比控制栏与可视化视图。
    """
    baseline = get_baseline(module_id)

    col_btn1, col_btn2, col_info = st.columns([1.8, 1.4, 4.8])

    with col_btn1:
        if st.button(
            "捕获当前为基线 [Run A]",
            key=f"btn_save_baseline_{module_id}",
            use_container_width=True,
            help="将当前超参数设置与指标固化为基准 Run A，调节参数后即可查看并排差分对比",
        ):
            save_baseline(
                module_id=module_id,
                name=current_name,
                params=current_params,
                metrics=current_metrics,
                loss_history=current_loss_history,
            )
            st.rerun()

    with col_btn2:
        if baseline is not None and st.button(
            "清除基线",
            key=f"btn_clear_baseline_{module_id}",
            use_container_width=True,
        ):
            clear_baseline(module_id)
            st.rerun()

    with col_info:
        if baseline is None:
            st.caption(
                "提示：点击上方按钮保存当前配置为 [Run A 基线]，随后修改超参数即可在下方实时观察双轨同轴消融差分对比。"
            )
        else:
            st.caption(
                f"已锚定基准 **[Run A: {baseline['name']}]** vs 当前 **[Run B: {current_name}]**"
            )

    if baseline is None:
        return

    # 展开差分对比主面板
    with st.container(border=True):
        icon_diff = svg_icon("activity", size=18, color="#1d4ed8")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:1.05rem;font-weight:700;color:#0f172a;margin-bottom:0.8rem;">'
            f"{icon_diff} <span>双实验并排消融差分视窗 (Side-by-Side Diff)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1])

        # 1. 超参数差分矩阵表
        with col_left:
            st.markdown(
                '<div style="font-size:0.88rem;font-weight:700;color:#1e40af;margin-bottom:0.4rem;">'
                "HYPERPARAMETER DIFF // 超参数差分表"
                "</div>",
                unsafe_allow_html=True,
            )

            all_keys = sorted(set(baseline["params"].keys()) | set(current_params.keys()))
            diff_rows = []
            for k in all_keys:
                v_a = baseline["params"].get(k, "—")
                v_b = current_params.get(k, "—")
                is_changed = str(v_a) != str(v_b)
                badge = (
                    '<span style="background:#fee2e2;color:#be123c;font-weight:700;padding:2px 6px;border-radius:4px;font-size:0.75rem;">CHANGED</span>'
                    if is_changed
                    else '<span style="color:#94a3b8;font-size:0.75rem;">SAME</span>'
                )
                val_b_styled = (
                    f'<span style="font-weight:700;color:#1d4ed8;">{v_b}</span>'
                    if is_changed
                    else f"<span>{v_b}</span>"
                )
                diff_rows.append(
                    f"<tr>"
                    f'<td style="padding:4px 8px;font-family:monospace;font-size:0.82rem;color:#334155;border-bottom:1px solid #f1f5f9;">{k}</td>'
                    f'<td style="padding:4px 8px;font-size:0.82rem;border-bottom:1px solid #f1f5f9;">{v_a}</td>'
                    f'<td style="padding:4px 8px;font-size:0.82rem;border-bottom:1px solid #f1f5f9;">{val_b_styled}</td>'
                    f'<td style="padding:4px 8px;text-align:center;border-bottom:1px solid #f1f5f9;">{badge}</td>'
                    f"</tr>"
                )

            table_html = (
                '<table style="width:100%;border-collapse:collapse;margin-bottom:0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">'
                '<thead style="background:#f8fafc;font-size:0.78rem;color:#64748b;font-weight:700;text-transform:uppercase;">'
                '<tr><th style="padding:6px 8px;text-align:left;">参数名</th><th style="padding:6px 8px;text-align:left;">Run A (基准)</th><th style="padding:6px 8px;text-align:left;">Run B (当前)</th><th style="padding:6px 8px;text-align:center;">状态</th></tr>'
                "</thead>"
                f"<tbody>{''.join(diff_rows)}</tbody>"
                "</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

        # 2. 指标差分卡片
        with col_right:
            st.markdown(
                '<div style="font-size:0.88rem;font-weight:700;color:#1e40af;margin-bottom:0.4rem;">'
                "METRICS DELTA // 评估指标增益 Delta"
                "</div>",
                unsafe_allow_html=True,
            )

            metric_keys = sorted(set(baseline["metrics"].keys()) | set(current_metrics.keys()))
            m_cols = st.columns(min(3, max(1, len(metric_keys))))
            for idx, mk in enumerate(metric_keys):
                ma = baseline["metrics"].get(mk, 0.0)
                mb = current_metrics.get(mk, 0.0)
                col_target = m_cols[idx % len(m_cols)]

                delta_str = ""
                delta_color = "#64748b"
                if isinstance(ma, (int, float)) and isinstance(mb, (int, float)):
                    delta_val = mb - ma
                    pct = (delta_val / abs(ma) * 100.0) if ma != 0 else 0.0
                    sign = "+" if delta_val > 0 else ""
                    delta_str = f"{sign}{delta_val:.4f} ({sign}{pct:.1f}%)"
                    delta_color = "#047857" if delta_val < 0 else "#be123c"  # Loss 类越小越好

                with col_target:
                    st.markdown(
                        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.4rem;">'
                        f'<div style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;">{mk}</div>'
                        f'<div style="font-size:1.15rem;font-weight:800;color:#0f172a;margin:0.2rem 0;">{mb}</div>'
                        f'<div style="font-size:0.76rem;font-weight:700;color:{delta_color};">vs 基线: {delta_str}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # 3. 同轴双曲线对比图 (如果双方均有 Loss 曲线)
        curve_a = baseline.get("loss_history", [])
        curve_b = current_loss_history or []

        if curve_a and curve_b:
            fig = go.Figure()
            steps_a = list(range(1, len(curve_a) + 1))
            steps_b = list(range(1, len(curve_b) + 1))

            fig.add_trace(
                go.Scatter(
                    x=steps_a,
                    y=curve_a,
                    mode="lines",
                    name=f"Run A 基线: {baseline['name']}",
                    line={"color": "#1d4ed8", "width": 2.5, "dash": "dash"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=steps_b,
                    y=curve_b,
                    mode="lines",
                    name=f"Run B 当前: {current_name}",
                    line={"color": "#be123c", "width": 2.5},
                )
            )

            fig.update_layout(
                title="同轴收敛轨迹差分对比 (Loss Convergence Overlay)",
                xaxis_title="Step / Epoch",
                yaxis_title="Loss",
                template="plotly_white",
                height=320,
                margin={"l": 40, "r": 20, "t": 40, "b": 40},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
            )
            st.plotly_chart(fig, use_container_width=True)
