# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_observability.py - 全链路可观测性与 TraceID 日志追踪系统测试套件
"""

import concurrent.futures
import datetime
import logging
import os
import shutil
import tempfile
import time

import pytest

from nn_core.observability import (
    RingBufferLogHandler,
    StandardLogFormatter,
    TraceIdFilter,
    generate_trace_id,
    get_logger,
    get_trace_id,
    purge_expired_logs,
    set_trace_id,
    setup_logging,
    trace_scope,
    traced_span,
)


class TestTraceContextAndGeneration:
    """TraceID 生成与上下文作用域隔离测试"""

    def test_trace_id_generation_format(self):
        tid = generate_trace_id("tr")
        assert tid.startswith("tr-")
        parts = tid.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 8
        assert len(parts[2]) == 8

    def test_trace_scope_nesting_and_restoration(self):
        set_trace_id("tr-root-001")
        assert get_trace_id() == "tr-root-001"

        with trace_scope("tr-child-001") as child_tid:
            assert child_tid == "tr-child-001"
            assert get_trace_id() == "tr-child-001"

            with trace_scope() as grand_tid:
                assert grand_tid.startswith("tr-")
                assert get_trace_id() == grand_tid

            assert get_trace_id() == "tr-child-001"

        assert get_trace_id() == "tr-root-001"

    def test_multithreaded_trace_context_isolation(self):
        def worker(tid: str):
            with trace_scope(tid):
                time.sleep(0.01)
                return get_trace_id()

        expected_ids = [f"tr-thread-{i:03d}" for i in range(10)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(worker, expected_ids))

        assert results == expected_ids


class TestLogFormattingAndHandlers:
    """日志格式化与环形缓冲区测试"""

    def test_trace_id_filter_injects_record_attribute(self):
        filter_ = TraceIdFilter()
        record = logging.LogRecord(
            name="test_mod",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="测试信息",
            args=(),
            exc_info=None,
        )
        with trace_scope("tr-inject-test"):
            filter_.filter(record)
            assert getattr(record, "trace_id", None) == "tr-inject-test"

    def test_standard_formatter_structure(self):
        formatter = StandardLogFormatter()
        record = logging.LogRecord(
            name="nn_core.attention",
            level=logging.INFO,
            pathname="attention.py",
            lineno=42,
            msg="MHA 前向传播完成",
            args=(),
            exc_info=None,
        )
        record.trace_id = "tr-format-123"  # type: ignore[attr-defined]
        formatted = formatter.format(record)
        assert "[INFO ]" in formatted
        assert "[tr-format-123]" in formatted
        assert "[nn_core.attention:42]" in formatted
        assert "MHA 前向传播完成" in formatted

    def test_ring_buffer_filtering_and_capacity(self):
        buf = RingBufferLogHandler(capacity=10)
        formatter = StandardLogFormatter()
        buf.setFormatter(formatter)

        for i in range(15):
            rec = logging.LogRecord(
                name="test",
                level=logging.INFO if i % 2 == 0 else logging.ERROR,
                pathname="test.py",
                lineno=i,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            rec.trace_id = f"tr-ring-{i % 3}"  # type: ignore[attr-defined]
            buf.emit(rec)

        # 验证容量截断为 10
        recent = buf.get_recent_logs(limit=20)
        assert len(recent) == 10

        # 按 trace_id 过滤
        filtered_trace = buf.get_recent_logs(trace_id="tr-ring-1")
        assert all("tr-ring-1" in x["trace_id"] for x in filtered_trace)

        # 按 min_level 过滤
        filtered_level = buf.get_recent_logs(min_level="ERROR")
        assert all(x["level"] == "ERROR" for x in filtered_level)

        buf.clear()
        assert len(buf.get_recent_logs()) == 0


class TestTracedSpanDecorator:
    """函数追踪装饰器测试"""

    def test_traced_span_success_execution(self):
        @traced_span(op_name="test_compute")
        def add(a: int, b: int) -> int:
            return a + b

        with trace_scope("tr-span-test"):
            res = add(2, 3)
            assert res == 5

    def test_traced_span_exception_capture_and_rethrow(self):
        @traced_span(op_name="test_fail")
        def failing():
            raise ValueError("维度不匹配")

        with trace_scope("tr-fail-test"), pytest.raises(ValueError, match="维度不匹配"):
            failing()


class TestFileLoggingRotationAndSetup:
    """按天日期命名与 5 天过期自动淘汰清理测试"""

    def test_setup_logging_writes_daily_date_stamped_files(self):
        temp_dir = tempfile.mkdtemp(prefix="nn_test_daily_logs_")
        today_str = time.strftime("%Y-%m-%d")
        try:
            setup_logging(
                log_dir=temp_dir,
                console_level=logging.DEBUG,
                file_level=logging.DEBUG,
                retention_days=5,
                force=True,
            )
            logger = get_logger("tests.observability")

            with trace_scope("tr-daily-write-001"):
                logger.info("测试按天带日期文件正常日志")
                logger.error("测试按天带日期文件异常日志")

            app_log = os.path.join(temp_dir, f"app-{today_str}.log")
            error_log = os.path.join(temp_dir, f"error-{today_str}.log")

            assert os.path.exists(app_log)
            assert os.path.exists(error_log)

            with open(app_log, encoding="utf-8") as f:
                app_content = f.read()
                assert "测试按天带日期文件正常日志" in app_content
                assert "测试按天带日期文件异常日志" in app_content
                assert "tr-daily-write-001" in app_content

            with open(error_log, encoding="utf-8") as f:
                error_content = f.read()
                assert "测试按天带日期文件正常日志" not in error_content
                assert "测试按天带日期文件异常日志" in error_content
                assert "tr-daily-write-001" in error_content
        finally:
            setup_logging(log_dir="logs", force=True)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_purge_expired_logs_5_day_policy(self):
        temp_dir = tempfile.mkdtemp(prefix="nn_test_purge_logs_")
        try:
            today = datetime.date.today()
            # 创建不同日期的模拟日志文件
            # 1. 今天的日志 (不应被删除)
            f_today = os.path.join(temp_dir, f"app-{today.strftime('%Y-%m-%d')}.log")
            # 2. 3 天前的日志 (不应被删除, <= 5天)
            day3 = today - datetime.timedelta(days=3)
            f_day3 = os.path.join(temp_dir, f"app-{day3.strftime('%Y-%m-%d')}.log")
            # 3. 5 天前的日志 (不应被删除, <= 5天)
            day5 = today - datetime.timedelta(days=5)
            f_day5 = os.path.join(temp_dir, f"app-{day5.strftime('%Y-%m-%d')}.log")
            # 4. 7 天前的日志 (应该被自动删除, > 5天)
            day7 = today - datetime.timedelta(days=7)
            f_day7 = os.path.join(temp_dir, f"app-{day7.strftime('%Y-%m-%d')}.log")
            # 5. 15 天前的错误日志 (应该被自动删除, > 5天)
            day15 = today - datetime.timedelta(days=15)
            f_day15 = os.path.join(temp_dir, f"error-{day15.strftime('%Y-%m-%d')}.log")

            for p in (f_today, f_day3, f_day5, f_day7, f_day15):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("mock log line\n")

            # 执行 5 天淘汰策略清理
            purged = purge_expired_logs(log_dir=temp_dir, retention_days=5)

            # 验证过期文件被成功清理
            assert f_day7 in purged
            assert f_day15 in purged
            assert not os.path.exists(f_day7)
            assert not os.path.exists(f_day15)

            # 验证 5 天之内的文件完好保留
            assert os.path.exists(f_today)
            assert os.path.exists(f_day3)
            assert os.path.exists(f_day5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
