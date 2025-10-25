#!/usr/bin/env python3
"""
Verify that training can start without errors
Quick smoke test before running full training
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import logging
from src.utils import setup_logging, load_config, get_device, set_random_seed
from src.data_preparation import create_identity_mapping, create_data_loaders
from src.model import load_insightface_buffalo_model

logger = setup_logging('logs', log_level='INFO', console=True, file=False)


def verify_training_setup():
    """Verify all components work before starting full training"""
    
    logger.info('=' * 80)
    logger.info('🧪 Training Setup Verification')
    logger.info('=' * 80)
    
    try:
        # 1. Load config
        logger.info('\n1️⃣  Loading configuration...')
        config = load_config('config/config.yaml')
        logger.info('✅ Configuration loaded')
        
        # 2. Set device
        logger.info('\n2️⃣  Checking GPU...')
        device = get_device('cuda', 0)
        logger.info(f'✅ Device ready: {device}')
        
        # 3. Set random seed
        logger.info('\n3️⃣  Setting random seed...')
        set_random_seed(42)
        logger.info('✅ Random seed set')
        
        # 4. Check dataset
        logger.info('\n4️⃣  Checking dataset...')
        data_root = config['paths']['data_root']
        identity_to_label, label_to_identity = create_identity_mapping(
            data_root=data_root,
            min_images=5
        )
        num_classes = len(identity_to_label)
        logger.info(f'✅ Dataset ready: {num_classes} identities')
        
        # 5. Create small data loaders for testing
        logger.info('\n5️⃣  Creating data loaders (small batch for testing)...')
        train_loader, val_loader, test_loader, _ = create_data_loaders(
            data_root=data_root,
            identity_to_label=identity_to_label,
            batch_size=8,  # Small batch for testing
            image_size=(112, 112),
            num_workers=2,
            pin_memory=True,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            augmentation_config=config['dataset']['augmentation'],
            random_seed=42
        )
        logger.info('✅ Data loaders created')
        
        # 6. Create model
        logger.info('\n6️⃣  Creating model...')
        model = load_insightface_buffalo_model(
            num_classes=num_classes,
            config=config,
            device=device
        )
        logger.info('✅ Model created')
        
        # 7. Test forward pass with mixed precision
        logger.info('\n7️⃣  Testing forward pass with mixed precision...')
        model.train()
        
        # Get one batch
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)
        
        # Test with mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        with torch.amp.autocast('cuda'):
            embeddings, loss = model(images, labels)
        
        logger.info(f'   Embeddings shape: {embeddings.shape}')
        logger.info(f'   Embeddings dtype: {embeddings.dtype}')
        logger.info(f'   Loss: {loss.item():.4f}')
        
        # Test accuracy calculation (the critical part that was failing)
        with torch.no_grad():
            embeddings_float = embeddings.float()
            embeddings_normalized = torch.nn.functional.normalize(embeddings_float, p=2, dim=1)
            weight_normalized = torch.nn.functional.normalize(model.loss_head.weight, p=2, dim=1)
            logits = torch.matmul(embeddings_normalized, weight_normalized.t())
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item() * 100.0
        
        logger.info(f'   Accuracy calculation: {accuracy:.2f}%')
        logger.info('✅ Forward pass successful with mixed precision')
        
        # 8. Test backward pass
        logger.info('\n8️⃣  Testing backward pass...')
        scaler.scale(loss).backward()
        logger.info('✅ Backward pass successful')
        
        # Success!
        logger.info('\n' + '=' * 80)
        logger.info('🎉 All verification tests passed!')
        logger.info('=' * 80)
        logger.info('')
        logger.info('✅ Configuration is correct')
        logger.info('✅ Dataset is accessible')
        logger.info('✅ Model can be created')
        logger.info('✅ Mixed precision training works')
        logger.info('✅ Forward and backward passes work')
        logger.info('')
        logger.info('🚀 Ready to start training!')
        logger.info('   Run: python src/train.py --config config/config.yaml')
        logger.info('')
        
        return True
        
    except Exception as e:
        logger.error(f'\n❌ Verification failed: {e}', exc_info=True)
        logger.info('\n💡 Please check the error above and fix before starting training')
        return False


if __name__ == '__main__':
    success = verify_training_setup()
    sys.exit(0 if success else 1)

