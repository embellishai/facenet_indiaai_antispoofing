"""
Utility functions for InsightFace finetuning project
Includes logging setup, config loading, seed setting, and helper functions
"""

import os
import yaml
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv


def setup_logging(log_dir: str, log_level: str = 'INFO', console: bool = True, file: bool = True) -> logging.Logger:
    """
    Set up comprehensive logging with console and file handlers
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Enable console logging
        file: Enable file logging
        
    Returns:
        Configured logger instance
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir_path / f'training_{timestamp}.log'
    
    logger = logging.getLogger('insightface_finetune')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f'📝 Logging initialized. Log file: {log_file}')
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        raise FileNotFoundError(f'❌ Config file not found: {config_path}')
    
    with open(config_path_obj, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def load_environment_variables(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file (optional)
    """
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
        else:
            logging.warning(f'⚠️ Environment file not found: {env_file}')
    else:
        load_dotenv()


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f'🎲 Random seed set to {seed}')


def get_device(device_name: str = 'cuda', gpu_id: int = 0) -> torch.device:
    """
    Get PyTorch device with proper configuration
    
    Args:
        device_name: Device type (cuda or cpu)
        gpu_id: GPU ID to use if cuda is available
        
    Returns:
        PyTorch device object
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        logging.info(f'🎮 Using GPU: {gpu_name} ({gpu_memory:.2f} GB)')
    else:
        device = torch.device('cpu')
        logging.warning('⚠️ CUDA not available, using CPU')
    
    return device


def create_directory_structure(base_path: str, directories: List[str]) -> None:
    """
    Create multiple directories in the project structure
    
    Args:
        base_path: Base path for the project
        directories: List of directory paths to create
    """
    base_path_obj = Path(base_path)
    
    for directory in directories:
        dir_path = base_path_obj / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f'📁 Directory structure created at {base_path}')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    param_info = {
        'trainable': trainable_params,
        'total': total_params,
        'frozen': total_params - trainable_params
    }
    
    logging.info(f'🔢 Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}, Frozen: {param_info["frozen"]:,}')
    
    return param_info


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    checkpoint_path: str,
    **kwargs
) -> None:
    """
    Save model checkpoint with training state
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_metric: Best validation metric
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path_obj)
    logging.info(f'💾 Checkpoint saved to {checkpoint_path}')


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary with training state
    """
    checkpoint_path_obj = Path(checkpoint_path)
    
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f'❌ Checkpoint not found: {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path_obj, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f'📂 Checkpoint loaded from {checkpoint_path}')
    logging.info(f'📊 Epoch: {checkpoint.get("epoch", "N/A")}, Best metric: {checkpoint.get("best_metric", "N/A")}')
    
    return checkpoint


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        return f'{minutes}m {seconds}s'
    else:
        return f'{seconds}s'


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate classification accuracy
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Accuracy as percentage
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = (correct / total) * 100.0
    
    return accuracy


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    Computes and stores the average and current value
    Useful for tracking metrics during training
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value
        
        Args:
            val: New value to add
            n: Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f'{self.name}: {self.avg:.4f} (current: {self.val:.4f})'


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

