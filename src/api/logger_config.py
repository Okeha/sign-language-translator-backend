"""Colored logging configuration for the application"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for terminal output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'            # Reset
    BOLD = '\033[1m'             # Bold
    DIM = '\033[2m'              # Dim
    
    def format(self, record):
        # Get the level color
        levelname = record.levelname
        level_color = self.COLORS.get(levelname, self.RESET)
        
        # Extract short module name (last part of logger name)
        logger_parts = record.name.split('.')
        short_name = logger_parts[-1] if len(logger_parts) > 1 else record.name
        
        # Custom format based on logger name
        if any(key in record.name for key in ['api.main', 'api.videomae', 'api.sentence_generation', 'api.websocket', 'api.session']):
            # Main API-related logs - clean format without full module path
            fmt = (
                f"{self.DIM}%(asctime)s{self.RESET} "
                f"{level_color}{self.BOLD}[%(levelname)s]{self.RESET} "
                f"%(message)s"
            )
        elif 'httpx' in record.name:
            # Httpx logs - completely dimmed (HuggingFace Hub requests)
            fmt = (
                f"{self.DIM}%(asctime)s{self.RESET} "
                f"{self.DIM}[%(levelname)s]{self.RESET} "
                f"{self.DIM}%(message)s{self.RESET}"
            )
        else:
            # Other logs - with compact module name
            fmt = (
                f"{self.DIM}%(asctime)s{self.RESET} "
                f"{level_color}[%(levelname)s]{self.RESET} "
                f"{self.DIM}[%(name)s]{self.RESET} "
                f"%(message)s"
            )
        
        formatter = logging.Formatter(fmt, datefmt='%H:%M:%S')
        return formatter.format(record)


def setup_colored_logging(level: int = logging.INFO) -> None:
    """
    Configure colored logging for the application
    
    Args:
        level: Logging level (default: INFO)
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    
    # Add handler to root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Set specific loggers
    # Suppress verbose httpx logs (HuggingFace Hub requests)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
