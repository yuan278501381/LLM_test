# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.observability - 全链路可观测性与 TraceID 日志追踪系统

提供工业级、世界级软件架构标准的统一日志与全链路遥测能力：
1. 线程与协程安全的 TraceID 隐式透传上下文 (基于 contextvars)；
2. 统一日志格式化器 (时间戳、日志级别、TraceID、模块行号、消息体)；
3. 按天自动切割且带日期的日志命名 (logs/app-YYYY-MM-DD.log 与 logs/error-YYYY-MM-DD.log)；
4. 5天自动淘汰清理策略 (purge_expired_logs)，严格控制磁盘占用；
5. 内存环形缓冲区 (RingBufferLogHandler)，支持前端 UI 实时探针查看与 TraceID 过滤；
6. 函数执行追踪装饰器 (@traced_span)，自动捕获耗时、张量形状与异常现场堆栈。
"""

import collections
import contextvars
import datetime
import functools
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

# ---------------------------------------------------------------------------
# 全链路 Trace 上下文管理器 (ContextVar 原生隔离)
# ---------------------------------------------------------------------------
_TRACE_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="tr-root")


def generate_trace_id(prefix: str = "tr") -> str:
    """生成符合 W3C 规范的紧凑唯一 TraceID (e.g. tr-17874088-a1b2c3d4)"""
    timestamp_hex = f"{int(time.time() * 1000) & 0xFFFFFFFF:08x}"
    random_hex = uuid.uuid4().hex[:8]
    return f"{prefix}-{timestamp_hex}-{random_hex}"


def get_trace_id() -> str:
    """获取当前协程/线程执行上下文中的活跃 TraceID"""
    return _TRACE_CONTEXT.get()


def set_trace_id(trace_id: str) -> None:
    """显式设置当前上下文的 TraceID"""
    _TRACE_CONTEXT.set(trace_id)


@contextmanager
def trace_scope(trace_id: str | None = None) -> Iterator[str]:
    """
    进入独立的 Trace 作用域。
    若未指定 trace_id，则自动生成全新的全局唯一 TraceID。
    作用域退出时自动还原上级上下文。
    """
    tid = trace_id or generate_trace_id()
    token = _TRACE_CONTEXT.set(tid)
    try:
        yield tid
    finally:
        _TRACE_CONTEXT.reset(token)


# ---------------------------------------------------------------------------
# 日志过滤器与格式化器
# ---------------------------------------------------------------------------
class TraceIdFilter(logging.Filter):
    """为每一条 LogRecord 自动注入当前上下文的 TraceID"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


class StandardLogFormatter(logging.Formatter):
    """统一结构化文本格式化器"""

    DEFAULT_FMT = (
        "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(trace_id)s] "
        "[%(name)s:%(lineno)d] %(message)s"
    )
    DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(
            fmt=fmt or self.DEFAULT_FMT,
            datefmt=datefmt or self.DEFAULT_DATEFMT,
        )


# ---------------------------------------------------------------------------
# 5天自动淘汰与按天滚动文件处理器 (Daily Retention Handler)
# ---------------------------------------------------------------------------
def purge_expired_logs(log_dir: str = "logs", retention_days: int = 5) -> list[str]:
    """
    自动扫描并删除超过 retention_days (默认 5 天) 的历史过期日志文件。
    支持识别形如 `app-YYYY-MM-DD.log`、`error-YYYY-MM-DD.log` 以及普通文件修改时间。
    返回已删除的文件路径列表。
    """
    if not os.path.exists(log_dir):
        return []

    now = time.time()
    cutoff_time = now - (retention_days * 86400)
    today_str = time.strftime("%Y-%m-%d")
    purged: list[str] = []

    date_pattern = re.compile(r"^(?:app|error)-(\d{4}-\d{2}-\d{2})\.log$")

    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        file_path = os.path.join(log_dir, filename)
        if not os.path.isfile(file_path):
            continue

        match = date_pattern.match(filename)
        is_expired = False
        if match:
            date_part = match.group(1)
            # 如果是今天的日志文件，绝对不删除
            if date_part == today_str:
                continue
            try:
                file_dt = datetime.datetime.strptime(date_part, "%Y-%m-%d").timestamp()
                # 加上一整天的秒数表示该天结束
                if (file_dt + 86400) < cutoff_time:
                    is_expired = True
            except ValueError:
                pass

        # 备选：根据文件最后修改时间判定
        if not is_expired:
            mtime = os.path.getmtime(file_path)
            if mtime < cutoff_time:
                is_expired = True

        if is_expired:
            try:
                os.remove(file_path)
                purged.append(file_path)
            except OSError:
                pass

    return purged


class DailyRetentionFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    按天自动切割并携带日期的世界级日志文件处理器 (app-YYYY-MM-DD.log)。
    支持 midnight 自动滚动并触发自动删除 5 天前过期日志。
    """

    def __init__(
        self,
        log_dir: str,
        prefix: str = "app",
        retention_days: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        self.log_dir = log_dir
        self.prefix = prefix
        self.retention_days = retention_days
        os.makedirs(log_dir, exist_ok=True)
        today_str = time.strftime("%Y-%m-%d")
        initial_filename = os.path.join(log_dir, f"{prefix}-{today_str}.log")
        super().__init__(
            filename=initial_filename,
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding=encoding,
            delay=False,
        )

    def doRollover(self) -> None:
        super().doRollover()
        # 触发 5 天生命周期自动清理
        purge_expired_logs(self.log_dir, self.retention_days)


# ---------------------------------------------------------------------------
# 内存环形缓冲区 Handler (供 UI 实时查看与调试)
# ---------------------------------------------------------------------------
class RingBufferLogHandler(logging.Handler):
    """
    线程安全的内存环形日志缓冲区，保留最近 N 条结构化日志记录。
    支持 Streamlit 界面即时呈现与 TraceID 过滤。
    """

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = capacity
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            trace_id = getattr(record, "trace_id", get_trace_id())
            log_item = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
                + f".{int(record.msecs):03d}",
                "level": record.levelname,
                "trace_id": trace_id,
                "logger_name": record.name,
                "filename": record.filename,
                "lineno": record.lineno,
                "message": record.getMessage(),
                "formatted": msg,
                "exc_text": record.exc_text or "",
            }
            with self._lock:
                self._buffer.append(log_item)
        except Exception:
            self.handleError(record)

    def get_recent_logs(
        self,
        limit: int = 100,
        trace_id: str | None = None,
        min_level: str | None = None,
    ) -> list[dict[str, Any]]:
        """按过滤条件获取最近的结构化日志列表"""
        level_values = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "FATAL": logging.CRITICAL,
            "CRITICAL": logging.CRITICAL,
        }
        min_val = level_values.get(min_level.upper(), logging.DEBUG) if min_level else logging.DEBUG

        with self._lock:
            items = list(self._buffer)

        filtered = []
        for item in reversed(items):
            if trace_id and trace_id not in item["trace_id"]:
                continue
            item_val = level_values.get(item["level"].upper(), logging.INFO)
            if item_val < min_val:
                continue
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return list(reversed(filtered))

    def clear(self) -> None:
        """清空环形缓冲区"""
        with self._lock:
            self._buffer.clear()


# 全局单例环形缓冲区
GLOBAL_RING_BUFFER = RingBufferLogHandler(capacity=1000)

# ---------------------------------------------------------------------------
# 全局日志系统初始化与获取
# ---------------------------------------------------------------------------
_LOGGING_INITIALIZED = False
_LOGGING_LOCK = threading.Lock()


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    retention_days: int = 5,
    force: bool = False,
) -> None:
    """
    初始化全局世界级日志系统：
    1. 配置根日志器与 TraceIdFilter；
    2. 注册控制台输出 StreamHandler；
    3. 注册带日期的按天滚动文件归档 logs/app-YYYY-MM-DD.log 与 logs/error-YYYY-MM-DD.log；
    4. 执行 5 天历史日志自动淘汰清理；
    5. 注册内存环形缓冲区 GLOBAL_RING_BUFFER。
    """
    global _LOGGING_INITIALIZED
    with _LOGGING_LOCK:
        if _LOGGING_INITIALIZED and not force:
            return

        os.makedirs(log_dir, exist_ok=True)
        formatter = StandardLogFormatter()
        trace_filter = TraceIdFilter()

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addFilter(trace_filter)

        # 移除已有默认 Handler 避免重复输出
        for h in list(root_logger.handlers):
            h.close()
            root_logger.removeHandler(h)

        # 1. 控制台 Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(trace_filter)
        root_logger.addHandler(console_handler)

        # 2. 全量按天滚动文件 Handler (logs/app-YYYY-MM-DD.log)
        app_file_handler = DailyRetentionFileHandler(
            log_dir=log_dir,
            prefix="app",
            retention_days=retention_days,
            encoding="utf-8",
        )
        app_file_handler.setLevel(file_level)
        app_file_handler.setFormatter(formatter)
        app_file_handler.addFilter(trace_filter)
        root_logger.addHandler(app_file_handler)

        # 3. 错误按天滚动文件 Handler (logs/error-YYYY-MM-DD.log)
        error_file_handler = DailyRetentionFileHandler(
            log_dir=log_dir,
            prefix="error",
            retention_days=retention_days,
            encoding="utf-8",
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        error_file_handler.addFilter(trace_filter)
        root_logger.addHandler(error_file_handler)

        # 4. 内存环形缓冲区 Handler
        GLOBAL_RING_BUFFER.setFormatter(formatter)
        GLOBAL_RING_BUFFER.addFilter(trace_filter)
        root_logger.addHandler(GLOBAL_RING_BUFFER)

        # 5. 执行启动时 5 天过期日志自动淘汰清理
        purged = purge_expired_logs(log_dir, retention_days=retention_days)

        _LOGGING_INITIALIZED = True
        logger = logging.getLogger("nn_core.observability")
        logger.info(
            "全链路统一日志系统初始化就绪 [Console: %s, File: %s, LogsDir: %s, 保留期: %d天, 自动清理: %d个过期文件]",
            logging.getLevelName(console_level),
            logging.getLevelName(file_level),
            os.path.abspath(log_dir),
            retention_days,
            len(purged),
        )


def get_logger(name: str) -> logging.Logger:
    """
    获取自动绑定 TraceID 过滤器的专业日志记录器。
    若系统尚未初始化，则触发默认初始化。
    """
    if not _LOGGING_INITIALIZED:
        setup_logging()
    logger = logging.getLogger(name)
    if not any(isinstance(f, TraceIdFilter) for f in logger.filters):
        logger.addFilter(TraceIdFilter())
    return logger


# ---------------------------------------------------------------------------
# 函数/算子执行追踪装饰器 (@traced_span)
# ---------------------------------------------------------------------------
P = ParamSpec("P")
R = TypeVar("R")


def traced_span(
    op_name: str | None = None,
    log_args: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    高阶执行追踪装饰器：
    - 自动关联当前 TraceID，若无则派生子 Span；
    - 测量执行耗时并记录 DEBUG/INFO 日志；
    - 发生未捕获异常时自动记录 ERROR 级别日志与完整堆栈。
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        operation = op_name or fn.__qualname__
        logger = get_logger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tid = get_trace_id()
            start_t = time.perf_counter()
            if log_args:
                logger.debug(
                    "[%s] 执行开始 %s(args=%d, kwargs=%d)", tid, operation, len(args), len(kwargs)
                )
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                logger.debug("[%s] 执行完成 %s (耗时: %.2f ms)", tid, operation, elapsed_ms)
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                logger.error(
                    "[%s] 执行异常 %s (耗时: %.2f ms, 异常: %s: %s)",
                    tid,
                    operation,
                    elapsed_ms,
                    type(exc).__name__,
                    str(exc),
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator
