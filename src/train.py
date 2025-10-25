"""
Training script for finetuning InsightFace buffalo_l model
Handles complete training loop with validation and checkpointing
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logging,
    load_config,
    load_environment_variables,
    set_random_seed,
    get_device,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    format_time,
    get_learning_rate,
    AverageMeter,
    merge_configs
)
from src.data_preparation import (
    create_identity_mapping,
    create_data_loaders
)
from src.model import load_insightface_buffalo_model


logger = logging.getLogger('insightface_finetune')


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        Optimizer instance
    """
    training_config = config.get('training', {})
    
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 0.0001)
    weight_decay = training_config.get('weight_decay', 0.0005)
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f'❌ Unsupported optimizer: {optimizer_name}')
    
    logger.info(f'🎯 Optimizer: {optimizer_name.upper()}, LR: {learning_rate}, Weight decay: {weight_decay}')
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    steps_per_epoch: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Training configuration
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        Learning rate scheduler
    """
    training_config = config.get('training', {})
    scheduler_config = training_config.get('scheduler', {})
    
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    num_epochs = training_config.get('num_epochs', 100)
    
    if scheduler_type == 'cosine':
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        min_lr = scheduler_config.get('min_lr', 0.000001)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    else:
        logger.warning(f'⚠️ Unknown scheduler type: {scheduler_type}, using none')
        return None
    
    logger.info(f'📅 Scheduler: {scheduler_type.upper()}')
    
    return scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Args:
        model: Face recognition model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        scaler: GradScaler for mixed precision training
        
    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.train()
    
    training_config = config.get('training', {})
    log_interval = config.get('logging', {}).get('log_interval', 10)
    gradient_clip = training_config.get('gradient_clip', 1.0)
    
    loss_meter = AverageMeter('Loss')
    accuracy_meter = AverageMeter('Accuracy')
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                embeddings, loss = model(images, labels)
            
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings, loss = model(images, labels)
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        # Calculate accuracy (using loss head weights)
        with torch.no_grad():
            # Convert to float32 if mixed precision was used
            embeddings_float = embeddings.float() if scaler is not None else embeddings
            embeddings_normalized = torch.nn.functional.normalize(embeddings_float, p=2, dim=1)
            weight_normalized = torch.nn.functional.normalize(model.loss_head.weight, p=2, dim=1)
            logits = torch.matmul(embeddings_normalized, weight_normalized.t())
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item() * 100.0
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        accuracy_meter.update(accuracy, images.size(0))
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{accuracy_meter.avg:.2f}%'
        })
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            logger.debug(f'📈 Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] - {loss_meter}, {accuracy_meter}')
    
    return loss_meter.avg, accuracy_meter.avg


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Validate model on validation set
    
    Args:
        model: Face recognition model
        val_loader: Validation data loader
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.eval()
    
    loss_meter = AverageMeter('Val Loss')
    accuracy_meter = AverageMeter('Val Accuracy')
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Validation {epoch}', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings, loss = model(images, labels)
            
            # Calculate accuracy - ensure float32 for compatibility
            embeddings_float = embeddings.float() if embeddings.dtype == torch.float16 else embeddings
            embeddings_normalized = torch.nn.functional.normalize(embeddings_float, p=2, dim=1)
            weight_normalized = torch.nn.functional.normalize(model.loss_head.weight, p=2, dim=1)
            logits = torch.matmul(embeddings_normalized, weight_normalized.t())
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item() * 100.0
            
            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            accuracy_meter.update(accuracy, images.size(0))
    
    logger.info(f'🎯 Validation - {loss_meter}, {accuracy_meter}')
    
    return loss_meter.avg, accuracy_meter.avg


def train(
    config: Dict[str, Any],
    resume_checkpoint: Optional[str] = None
) -> None:
    """
    Main training function
    
    Args:
        config: Training configuration dictionary
        resume_checkpoint: Path to checkpoint to resume from (optional)
    """
    logger.info('🚀 ***** Starting training pipeline...')
    
    # Extract configurations
    paths_config = config.get('paths', {})
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    hardware_config = config.get('hardware', {})
    validation_config = config.get('validation', {})
    
    # Set up device
    device = get_device(
        device_name=hardware_config.get('device', 'cuda'),
        gpu_id=hardware_config.get('gpu_id', 0)
    )
    
    # Create identity mapping
    data_root = paths_config.get('data_root', 'data/raw')
    min_images = dataset_config.get('min_images_per_identity', 5)
    
    identity_to_label, label_to_identity = create_identity_mapping(
        data_root=data_root,
        min_images=min_images
    )
    
    num_classes = len(identity_to_label)
    logger.info(f'🎭 Total identities: {num_classes}')
    
    # Create data loaders
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        data_root=data_root,
        identity_to_label=identity_to_label,
        batch_size=training_config.get('batch_size', 64),
        image_size=tuple(dataset_config.get('image_size', [112, 112])),
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
        train_ratio=dataset_config.get('train_split', 0.8),
        val_ratio=dataset_config.get('val_split', 0.1),
        test_ratio=dataset_config.get('test_split', 0.1),
        augmentation_config=dataset_config.get('augmentation', {}),
        random_seed=42
    )
    
    # Create model
    model = load_insightface_buffalo_model(
        num_classes=num_classes,
        config=config,
        pretrained_path=None,
        device=device
    )
    
    # Count parameters
    count_parameters(model)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision training
    scaler = None
    if training_config.get('mixed_precision', False) and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        logger.info('⚡ Mixed precision training enabled')
    
    # Set up tensorboard
    if config.get('logging', {}).get('tensorboard', False):
        log_dir = Path(paths_config.get('log_dir', 'logs')) / 'tensorboard'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        logger.info(f'📊 TensorBoard logging enabled: {log_dir}')
    else:
        writer = None
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_metric = 0.0
    
    if resume_checkpoint:
        checkpoint = load_checkpoint(
            checkpoint_path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        logger.info(f'🔄 Resuming from epoch {start_epoch}')
    
    # Training loop
    num_epochs = training_config.get('num_epochs', 100)
    freeze_epochs = config.get('model', {}).get('freeze_epochs', 0)
    
    logger.info(f'🎬 Starting training for {num_epochs} epochs')
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Freeze/unfreeze backbone
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
        
        logger.info(f'\n{"=" * 80}')
        logger.info(f'🔄 ***** Starting Epoch {epoch}/{num_epochs}...')
        logger.info(f'📚 Learning rate: {get_learning_rate(optimizer):.6f}')
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler
        )
        
        logger.info(f'✅ ***** Epoch {epoch} training done.')
        logger.info(f'📊 Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        
        # Validation
        if epoch % validation_config.get('frequency', 1) == 0:
            val_loss, val_acc = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch
            )
            
            # Check if best model
            metric_name = validation_config.get('metric', 'accuracy')
            current_metric = val_acc if metric_name == 'accuracy' else -val_loss
            
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                logger.info(f'🏆 New best model! {metric_name}: {best_metric:.4f}')
            
            # Save checkpoint
            checkpoint_dir = Path(paths_config.get('checkpoint_dir', 'models/checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save latest checkpoint
            latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                checkpoint_path=str(latest_checkpoint_path),
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )
            
            # Save best checkpoint
            if is_best:
                best_checkpoint_path = checkpoint_dir / 'best_checkpoint.pth'
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    checkpoint_path=str(best_checkpoint_path),
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc
                )
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('Learning_Rate', get_learning_rate(optimizer), epoch)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f'⏱️ Epoch time: {format_time(epoch_time)}')
    
    # Training complete
    total_training_time = time.time() - training_start_time
    logger.info(f'\n{"=" * 80}')
    logger.info(f'✅ ***** Training complete!')
    logger.info(f'⏱️ Total training time: {format_time(total_training_time)}')
    logger.info(f'🏆 Best {validation_config.get("metric", "accuracy")}: {best_metric:.4f}')
    
    # Save final model
    final_model_dir = Path(paths_config.get('final_model_dir', 'models/final'))
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_model_dir / 'final_model.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'embedding_size': config.get('model', {}).get('embedding_size', 512),
        'identity_to_label': identity_to_label,
        'label_to_identity': label_to_identity,
        'best_metric': best_metric
    }, final_model_path)
    
    logger.info(f'💾 Final model saved to {final_model_path}')
    
    if writer:
        writer.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train InsightFace buffalo_l model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_environment_variables()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger_instance = setup_logging(
        log_dir=config.get('paths', {}).get('log_dir', 'logs'),
        log_level=config.get('logging', {}).get('level', 'INFO'),
        console=config.get('logging', {}).get('console', True),
        file=config.get('logging', {}).get('file', True)
    )
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Start training
    try:
        train(config=config, resume_checkpoint=args.resume)
    except KeyboardInterrupt:
        logger.info('⚠️ Training interrupted by user')
    except Exception as e:
        logger.error(f'❌ Training failed with error: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    main()

