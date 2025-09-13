"""Optimized configuration utilities for BIER evaluation."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration file efficiently."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "eval_config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get configured logger instance efficiently."""
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger_instance