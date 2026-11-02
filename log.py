# log.py
import logging
import os
import sys
from pathlib import Path
from loguru import logger as _logger  # use _logger to avoid confusion with the logger exported below


def setup_logging(log_file: str | None = None):
    """
    Configure loguru logging:
    - File + console output
    - Intercept standard library logging / uvicorn / celery logs
    Only takes effect on the first call; subsequent calls return the already configured logger directly.
    """
    # If already configured, reuse it directly
    if getattr(_logger, "_configured_for_api", False):
        return _logger

    # Default log file: papermaster.log in the current directory
    if log_file is None:
        log_dir = Path(__file__).resolve().parent
        log_file = log_dir / "papermaster.log"
    else:
        log_file = Path(log_file)

    # First remove loguru's default handler
    _logger.remove()

    # File output
    _logger.add(
        log_file,
        rotation="10 MB",
        retention=5,
        level="INFO",
        diagnose=False,
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss:SSS} | {level: <8} | API | "
            "{name}:{function}:{line} | {message} | extra={extra}"
        ),
    )

    # Console output (colored)
    _logger.add(
        sys.stdout,
        level="INFO",
        diagnose=False,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss:SSS}</green> | "
            "<level>{level: <8}</level> | <blue>API</blue> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level> | extra={extra}"
        ),
    )

    # === Intercept standard library logging ===
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = _logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            _logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # === Intercept uvicorn / fastapi logs ===
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        log = logging.getLogger(name)
        log.handlers = [InterceptHandler()]
        log.propagate = False

    # === Intercept Celery-related logs ===
    celery_loggers = [
        "celery",
        "celery.app",
        "celery.task",
        "celery.worker",
        "celery.beat",
        "celery.redirected",
        "celery.app.trace",
        "billiard",
        "kombu",
    ]

    for logger_name in celery_loggers:
        celery_logger = logging.getLogger(logger_name)
        if not celery_logger.handlers or not any(
            isinstance(h, InterceptHandler) for h in celery_logger.handlers
        ):
            celery_logger.handlers = [InterceptHandler()]
            celery_logger.propagate = False
            celery_logger.setLevel(logging.INFO)

    # Mark: configuration is complete
    _logger._configured_for_api = True
    return _logger


# Expose loggers for different "roles" (actually the same loguru instance with different extra)
reason_logger = _logger.bind(role="reasoner")
quick_logger = _logger.bind(role="quick_thinker")
manager_logger = _logger.bind(role="manager")

# If you want to use the logger name directly elsewhere, a generic one is also exposed
logger = _logger
