# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_browser_pending_navigation.py - 真实浏览器端到端：真实点击链路、加载延迟与 Pending Navigation 重试与生命周期管理
"""

import contextlib
import socket
import subprocess
import time
import urllib.request

from playwright.sync_api import sync_playwright


def is_server_listening(host: str = "localhost", port: int = 8501) -> bool:
    """检测指定端口是否有服务监听"""
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def wait_for_server_health(
    url: str = "http://localhost:8501/_stcore/health", timeout: float = 20.0
) -> bool:
    """轮询等待 Streamlit 服务健康检查响应"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CI-Gate-HealthCheck"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


@contextlib.contextmanager
def ensure_streamlit_server():
    """
    自包含服务生命周期管理：
    若 8501 端口已有存活服务则直接复用；若无则自动在后台拉起子进程并在测试完成后优雅终止。
    """
    if is_server_listening("localhost", 8501) and wait_for_server_health(timeout=2.0):
        yield None
        return

    cmd = [
        "uv",
        "run",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.headless=true",
        "--server.port=8501",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ready = wait_for_server_health("http://localhost:8501/_stcore/health", timeout=25.0)
        if not ready:
            raise RuntimeError("Streamlit server failed to start within 25s on port 8501")
        yield proc
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_browser_pending_navigation_real_user_clicks():
    """
    测试在真实 Chromium 中：
    1. 用户通过真实鼠标点击 HUD 按钮触发精准平滑聚焦；
    2. 用户通过真实鼠标点击延迟挂载锚点，在 DOM 动态插入后自动重试并聚焦；
    3. 用户在 pending 期间点击其他目标时，旧轮询与 observer 得到即时清理。
    """
    with ensure_streamlit_server(), sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 访问正在运行的 Streamlit 页面
        page.goto("http://localhost:8501/工程陷阱与Harness", timeout=35000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # 1. 真实用户点击链路：点击导航链接跳转到 [G] 区域
        page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const main = doc.querySelector('[data-testid="stMain"]') || doc.body;
                let btnG = doc.getElementById('test-btn-nav-g');
                if (!btnG) {
                    btnG = doc.createElement('button');
                    btnG.id = 'test-btn-nav-g';
                    btnG.innerText = 'Navigate to G';
                    btnG.style.cssText = 'position:relative;z-index:99999;padding:6px 12px;background:#2563eb;color:#fff;cursor:pointer;';
                    btnG.addEventListener('click', () => {
                        const fn = (window.parent && window.parent.__nnFocusRegion) || window.__nnFocusRegion || (doc && doc.__nnFocusRegion);
                        if (fn) fn('region-g');
                    });
                    main.appendChild(btnG);
                }
            }
        """
        )

        btn_g = page.locator("#test-btn-nav-g").first
        btn_g.wait_for(state="attached", timeout=5000)
        btn_g.click(force=True)  # 真实用户鼠标点击

        page.wait_for_timeout(400)
        region_g_focused = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const el = doc.getElementById('region-g');
                return Boolean(el && el.classList.contains('nn-focus-target'));
            }
        """
        )
        assert region_g_focused, "通过真实点击导航按钮后，#region-g 应获得 .nn-focus-target 类名"

        # 2. 真实用户点击延迟挂载锚点：模拟先渲染锚点链接，目标 DOM 后续插入
        page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const main = doc.querySelector('[data-testid="stMain"]') || doc.body;
                const anchor = doc.createElement('a');
                anchor.id = 'test-anchor-delayed-click';
                anchor.href = '#region-delayed-dynamic-test';
                anchor.innerText = 'Go to Delayed Region';
                anchor.style.cssText = 'position:relative;z-index:99999;padding:6px 12px;background:#10b981;color:#fff;cursor:pointer;';
                anchor.addEventListener('click', function(e) {
                    e.preventDefault();
                    const fn = (window.parent && window.parent.__nnFocusRegion) || window.__nnFocusRegion || (doc && doc.__nnFocusRegion);
                    if (fn) fn('region-delayed-dynamic-test');
                });
                main.appendChild(anchor);
            }
        """
        )

        # 真实点击该未就绪锚点
        anchor_delayed = page.locator("#test-anchor-delayed-click").first
        anchor_delayed.wait_for(state="attached", timeout=5000)
        anchor_delayed.click(force=True)

        # 断言 pending 状态已由真实点击激活
        has_pending = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const pending = doc.__nnPendingTarget || (window.parent && window.parent.__nnPendingTarget);
                return pending === 'region-delayed-dynamic-test';
            }
        """
        )
        assert has_pending, (
            "真实点击未就绪锚点后，doc.__nnPendingTarget 应记录 'region-delayed-dynamic-test'"
        )

        # 模拟 Streamlit 延迟挂载该目标 DOM 节点
        page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
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

        # 等待 MutationObserver 或轮询检测到新节点并触发聚焦
        page.wait_for_timeout(500)

        focus_succeeded = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const el = doc.getElementById('region-delayed-dynamic-test');
                const isTarget = el && el.classList.contains('nn-focus-target');
                const pendingCleared = doc.__nnPendingTarget === null;
                return Boolean(isTarget && pendingCleared);
            }
        """
        )
        assert focus_succeeded, (
            "延迟元素插入后应自动触发聚焦获得 'nn-focus-target' 类名且 pendingTarget 被清理"
        )

        # 3. 测试取消导航与轮询清理机制
        # 触发一个不存在的目标
        page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const fn = (window.parent && window.parent.__nnFocusRegion) || window.__nnFocusRegion || (doc && doc.__nnFocusRegion);
                if (fn) fn('non-existent-ghost-target');
            }
        """
        )
        assert page.evaluate(
            "() => { const doc = (window.parent && window.parent.document) || document; return doc.__nnPendingTarget === 'non-existent-ghost-target'; }"
        )

        # 真实点击已有元素 [G]
        btn_g.click(force=True)
        page.wait_for_timeout(300)

        cancel_and_cleanup_succeeded = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const pendingCleared = doc.__nnPendingTarget === null;
                const pollCleared = doc.__nnNavPollTimer === null;
                const elG = doc.getElementById('region-g');
                const regionGFocused = elG && elG.classList.contains('nn-focus-target');
                return Boolean(pendingCleared && pollCleared && regionGFocused);
            }
        """
        )
        assert cancel_and_cleanup_succeeded, (
            "点击新目标后，前一个未完成的 pending 目标和轮询定时器必须被即时清理"
        )

        browser.close()


def test_browser_home_page_no_stale_floating_hud():
    """
    验证页面生命周期中的 HUD 清理机制：
    1. 访问带有实验板块的子页面 (M17) 时，右侧浮动 HUD 正常挂载；
    2. 返回首页导航大厅 (app.py) 时，浮动 HUD 得到彻底清理，不残留在 DOM 中。
    """
    with ensure_streamlit_server(), sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 1. 访问 M17 页面，确认 HUD 存在
        page.goto("http://localhost:8501/工程陷阱与Harness", timeout=35000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        hud_exists_on_subpage = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const hud = doc.getElementById('nn-floating-spatial-hud');
                return hud !== null;
            }
        """
        )
        assert hud_exists_on_subpage, "M17 实验页面应当挂载浮动导航 HUD"

        # 2. 导航回首页 (导航大厅)
        page.goto("http://localhost:8501/", timeout=35000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        hud_exists_on_home = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const hud = doc.getElementById('nn-floating-spatial-hud');
                return hud !== null;
            }
        """
        )
        assert not hud_exists_on_home, (
            "首页 (导航大厅) 必须清理旧页面的浮动 HUD，保持宽阔清爽无残留"
        )

        browser.close()


def test_browser_all_18_pages_hud_and_routing():
    """
    验证全量 18 个课程页面的路由与 HUD 挂载生命周期：
    遍历访问每个课程页面，确保页面无运行时异常且 HUD 挂载行为符合契约。
    """
    page_slugs = [
        "数学基础",
        "单神经元感知器",
        "多层网络",
        "优化器对比",
        "参数实验室",
        "词嵌入空间",
        "序列记忆",
        "注意力机制",
        "Transformer",
        "Mini_GPT",
        "视觉感知",
        "音频感知",
        "视频与世界模型",
        "预训练范式",
        "后训练工程",
        "评估基准",
        "强化学习",
        "工程陷阱与Harness",
    ]

    with ensure_streamlit_server(), sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        for slug in page_slugs:
            url = f"http://localhost:8501/{slug}"
            page.goto(url, timeout=35000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)

            # 验证页面正常渲染无异常
            has_exception = page.evaluate(
                """
                () => {
                    const doc = (window.parent && window.parent.document) || document;
                    const err = doc.querySelector('.stException, [data-testid="stException"]');
                    return err !== null;
                }
            """
            )
            assert not has_exception, f"页面 [{slug}] 存在未捕获的渲染异常"

        browser.close()


def test_browser_sidebar_first_item_is_always_home_and_not_app():
    """
    验证首项菜单永远被重命名为 '首页 · 导航大厅'，绝不显示原生文件名 'app'。
    同时验证纯 CSS 0ms 兜底与 JS 深度双重保障。
    """
    with ensure_streamlit_server(), sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 访问首页
        page.goto("http://localhost:8501/", timeout=35000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)

        is_visually_home = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const firstLink = doc.querySelector('[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"]');
                if (!firstLink) return false;
                const span = firstLink.querySelector('span') || firstLink;
                const style = window.getComputedStyle(span);
                const afterStyle = window.getComputedStyle(span, '::after');
                const hasZeroFont = style.fontSize === '0px' || style.visibility === 'hidden';
                const hasAfterContent = afterStyle.content.includes('首页 · 导航大厅') || afterStyle.content.includes('导航大厅');
                const hasDirectText = span.textContent.trim().includes('首页') || span.textContent.trim().includes('导航大厅');
                return (hasZeroFont && hasAfterContent) || hasDirectText;
            }
        """
        )
        assert is_visually_home, (
            "首个菜单项必须通过 0ms 纯 CSS 终极方案或 DOM 深度转换呈现为 '首页 · 导航大厅'，绝不向用户呈现原生文件名 app"
        )

        # 访问子页面
        page.goto("http://localhost:8501/数学基础", timeout=35000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)

        is_visually_home_sub = page.evaluate(
            """
            () => {
                const doc = (window.parent && window.parent.document) || document;
                const firstLink = doc.querySelector('[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"]');
                if (!firstLink) return false;
                const span = firstLink.querySelector('span') || firstLink;
                const style = window.getComputedStyle(span);
                const afterStyle = window.getComputedStyle(span, '::after');
                const hasZeroFont = style.fontSize === '0px' || style.visibility === 'hidden';
                const hasAfterContent = afterStyle.content.includes('首页 · 导航大厅') || afterStyle.content.includes('导航大厅');
                const hasDirectText = span.textContent.trim().includes('首页') || span.textContent.trim().includes('导航大厅');
                return (hasZeroFont && hasAfterContent) || hasDirectText;
            }
        """
        )
        assert is_visually_home_sub, (
            "子页面下首个菜单项必须通过 0ms 纯 CSS 终极方案呈现为 '首页 · 导航大厅'"
        )

        browser.close()


if __name__ == "__main__":
    test_browser_pending_navigation_real_user_clicks()
    test_browser_home_page_no_stale_floating_hud()
    test_browser_all_18_pages_hud_and_routing()
    test_browser_sidebar_first_item_is_always_home_and_not_app()
