"""
Logging module for the auto_ml package.

This module provides a centralized logging system with different logging levels,
handlers, and formatters. It supports both file and console logging, with
configurable log rotation and formatting options.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict

class AutoMLLogger:
    """
    A configurable logger for the auto_ml package that handles both
    console and file logging with different verbosity levels.
    """
    
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    def __init__(
        self,
        name: str = "auto_ml",
        level: Union[str, int] = logging.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        date_format: Optional[str] = None,
        file_logging: bool = True,
        console_logging: bool = True,
        max_bytes: int = 10_485_760,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the AutoMLLogger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            format_string: Custom format string for log messages
            date_format: Custom date format for log messages
            file_logging: Enable/disable file logging
            console_logging: Enable/disable console logging
            max_bytes: Maximum size of each log file
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent adding handlers multiple times
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        self.format_string = format_string or self.DEFAULT_FORMAT
        self.date_format = date_format or self.DEFAULT_DATE_FORMAT
        self.formatter = logging.Formatter(self.format_string, self.date_format)
        
        if console_logging:
            self._setup_console_handler()
        
        if file_logging:
            self._setup_file_handler(log_dir, max_bytes, backup_count)
    
    def _setup_console_handler(self) -> None:
        """Set up console (stdout) logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(
        self,
        log_dir: Optional[Union[str, Path]],
        max_bytes: int,
        backup_count: int
    ) -> None:
        """Set up rotating file handler for logging."""
        if log_dir is None:
            log_dir = Path.home() / ".auto_ml" / "logs"
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"auto_ml_{datetime.now():%Y%m%d}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self.logger

# Global logger instance with default settings
default_logger = AutoMLLogger().get_logger()

class LoggerMixin:
    """
    A mixin class that provides logging capabilities to other classes
    in the auto_ml package.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the mixin with a logger instance."""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(f"auto_ml.{self.__class__.__name__}")
    
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """
        Generic logging method.
        
        Args:
            level: Logging level
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

def configure_logger(
    config: Dict,
    logger_name: str = "auto_ml"
) -> logging.Logger:
    """
    Configure a logger instance from a configuration dictionary.
    
    Args:
        config: Configuration dictionary containing logger settings
        logger_name: Name for the logger instance
    
    Returns:
        Configured logger instance
    """
    return AutoMLLogger(
        name=logger_name,
        level=config.get("level", logging.INFO),
        log_dir=config.get("log_dir"),
        format_string=config.get("format_string"),
        date_format=config.get("date_format"),
        file_logging=config.get("file_logging", True),
        console_logging=config.get("console_logging", True),
        max_bytes=config.get("max_bytes", 10_485_760),
        backup_count=config.get("backup_count", 5)
    ).get_logger()