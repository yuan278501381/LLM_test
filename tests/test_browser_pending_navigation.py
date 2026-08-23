# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_browser_pending_navigation.py - 真实浏览器端到端：加载延迟与 Pending Navigation 重试测试
"""

from playwright.sync_api import sync_playwright


def test_browser_pending_navigation_delayed_element_focus():
    """测试在真实 Chromium 中目标元素尚未进入 DOM 时点击导航，元素挂载后能自动重试并精准聚焦"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 访问正在运行的 Streamlit 页面
        page.goto("http://localhost:8501/工程陷阱与Harness", timeout=30000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # 1. 模拟在目标元素不存在时发起导航点击
        has_pending = page.evaluate(
            """
            () => {
                const doc = window.parent.document;
                if (!doc.__nnFocusRegion) return false;
                // 尝试聚焦一个尚未挂载的延迟目标 ID
                doc.__nnFocusRegion('region-delayed-dynamic-test');
                return doc.__nnPendingTarget === 'region-delayed-dynamic-test';
            }
        """
        )
        assert has_pending, "Expected doc.__nnPendingTarget to record 'region-delayed-dynamic-test'"

        # 2. 模拟 Streamlit 延迟挂载该目标 DOM 节点
        page.evaluate(
            """
            () => {
                const doc = window.parent.document;
                const main = doc.querySelector('[data-testid="stMain"]') || doc.body;
                const delayedEl = doc.createElement('div');
                delayedEl.id = 'region-delayed-dynamic-test';
                delayedEl.className = 'interactive-region';
                delayedEl.style.height = '100px';
                delayedEl.innerHTML = '<b>Delayed Region Target</b>';
                main.appendChild(delayedEl);
            }
        """
        )

        # 3. 等待 MutationObserver 或轮询检测到新节点并触发聚焦
        page.wait_for_timeout(500)

        # 4. 断言目标元素已成功获得 nn-focus-target 类名且 pending 状态被清理
        focus_succeeded = page.evaluate(
            """
            () => {
                const doc = window.parent.document;
                const el = doc.getElementById('region-delayed-dynamic-test');
                const isTarget = el && el.classList.contains('nn-focus-target');
                const pendingCleared = doc.__nnPendingTarget === null;
                return isTarget && pendingCleared;
            }
        """
        )
        assert focus_succeeded, (
            "Expected delayed element to be focused with class 'nn-focus-target' and pending target cleared"
        )

        browser.close()


if __name__ == "__main__":
    test_browser_pending_navigation_delayed_element_focus()
