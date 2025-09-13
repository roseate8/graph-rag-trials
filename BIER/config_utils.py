"""
Configuration utilities for BIER evaluation.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "eval_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    logger_instance = logging.getLogger(name)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger_instance.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if not logger_instance.handlers:  # Avoid duplicate handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
    
    return logger_instance


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['retrieval', 'evaluation', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate retrieval config
    retrieval_config = config['retrieval']
    required_retrieval = ['embedding_model', 'milvus_profile', 'collection_name']
    for key in required_retrieval:
        if key not in retrieval_config:
            raise ValueError(f"Missing required retrieval config: {key}")
    
    # Validate evaluation config
    eval_config = config['evaluation']
    if 'datasets' not in eval_config:
        raise ValueError("Missing datasets configuration")
    
    if 'metrics' not in eval_config:
        raise ValueError("Missing metrics configuration")
    
    return True