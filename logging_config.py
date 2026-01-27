"""
Centralized Logging Configuration for SRDA
============================================
Formats: DATE | MODULE | LEVEL | MESSAGE

Separate log files for:
- General application logs (srda_YYYYMMDD.log)
- Download attempts (downloads_YYYYMMDD.log)
- Gemini AI decisions (gemini_YYYYMMDD.log)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

# Ensure logs directory exists
config.LOGS_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[34m',      # Blue
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Add color to levelname
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{reset}"
        
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        return result


def setup_logging(
    log_level: int = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """
    Configure all loggers for the SRDA application.
    
    Args:
        log_level: Minimum log level (default: INFO)
        enable_console: Whether to output to console
        enable_file: Whether to write to log files
    """
    # Format: DATE | MODULE | LEVEL | MESSAGE
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger for SRDA
    root_logger = logging.getLogger('srda')
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    date_str = datetime.now().strftime('%Y%m%d')
    
    if enable_file:
        # Main application log
        main_handler = logging.FileHandler(
            config.LOGS_DIR / f"srda_{date_str}.log",
            encoding='utf-8'
        )
        main_handler.setFormatter(file_formatter)
        main_handler.setLevel(log_level)
        root_logger.addHandler(main_handler)
    
    if enable_console:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    # Specialized loggers with their own files
    specialized_loggers = {
        'srda.downloads': f"downloads_{date_str}.log",
        'srda.gemini': f"gemini_{date_str}.log",
        'srda.gmail': f"gmail_{date_str}.log"
    }
    
    for logger_name, log_file in specialized_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = True  # Also log to parent
        
        if enable_file:
            handler = logging.FileHandler(
                config.LOGS_DIR / log_file,
                encoding='utf-8'
            )
            handler.setFormatter(file_formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the 'srda' prefix.
    
    Args:
        name: Logger name (will be prefixed with 'srda.')
        
    Returns:
        Configured logger instance
    """
    if not name.startswith('srda'):
        name = f'srda.{name}'
    return logging.getLogger(name)


# Convenience function for quick setup
def quick_setup(debug: bool = False) -> None:
    """Quick setup with sensible defaults."""
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level=level)


# Auto-setup on import if not already configured
if not logging.getLogger('srda').handlers:
    setup_logging()
