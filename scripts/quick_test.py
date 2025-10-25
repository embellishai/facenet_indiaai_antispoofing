#!/usr/bin/env python3
"""
Quick test script to verify installation and basic functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
from src.utils import get_device, setup_logging
from src.model import FaceRecognitionModel

logger = setup_logging('logs', log_level='INFO')


def test_imports():
    """Test that all required libraries are installed"""
    logger.info('🔍 ***** Testing imports...')
    
    try:
        import torch
        import torchvision
        import cv2
        import albumentations
        import yaml
        from dotenv import load_dotenv
        from tqdm import tqdm
        
        logger.info('✅ All required libraries imported successfully')
        return True
    except ImportError as e:
        logger.error(f'❌ Import error: {e}')
        return False


def test_cuda():
    """Test CUDA availability"""
    logger.info('🔍 ***** Testing CUDA...')
    
    if torch.cuda.is_available():
        device = get_device('cuda', 0)
        logger.info(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
        logger.info(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
        return True
    else:
        logger.warning('⚠️ CUDA not available, will use CPU')
        return False


def test_model_creation():
    """Test model creation"""
    logger.info('🔍 ***** Testing model creation...')
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = FaceRecognitionModel(
            num_classes=10,
            embedding_size=512,
            depth=100,
            dropout=0.0
        )
        
        model = model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 112, 112).to(device)
        dummy_labels = torch.tensor([0, 1]).to(device)
        
        embeddings, loss = model(dummy_input, dummy_labels)
        
        logger.info(f'✅ Model created successfully')
        logger.info(f'📊 Embedding shape: {embeddings.shape}')
        logger.info(f'📉 Loss: {loss.item():.4f}')
        
        return True
    except Exception as e:
        logger.error(f'❌ Model creation failed: {e}')
        return False


def test_opencv():
    """Test OpenCV functionality"""
    logger.info('🔍 ***** Testing OpenCV...')
    
    try:
        # Create a dummy image
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Test basic operations
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        logger.info('✅ OpenCV working correctly')
        return True
    except Exception as e:
        logger.error(f'❌ OpenCV test failed: {e}')
        return False


def test_config_loading():
    """Test configuration loading"""
    logger.info('🔍 ***** Testing config loading...')
    
    try:
        from src.utils import load_config
        
        config_path = project_root / 'config' / 'config.yaml'
        
        if config_path.exists():
            config = load_config(str(config_path))
            logger.info('✅ Configuration loaded successfully')
            logger.info(f'📊 Batch size: {config.get("training", {}).get("batch_size")}')
            logger.info(f'📊 Num epochs: {config.get("training", {}).get("num_epochs")}')
            return True
        else:
            logger.warning('⚠️ Config file not found, but this is OK for basic testing')
            return True
    except Exception as e:
        logger.error(f'❌ Config loading failed: {e}')
        return False


def main():
    """Run all tests"""
    logger.info('=' * 80)
    logger.info('🧪 Running Installation Tests')
    logger.info('=' * 80)
    
    tests = [
        ('Import Test', test_imports),
        ('CUDA Test', test_cuda),
        ('Model Creation Test', test_model_creation),
        ('OpenCV Test', test_opencv),
        ('Config Loading Test', test_config_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f'\n--- {test_name} ---')
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f'❌ {test_name} failed with exception: {e}')
            results[test_name] = False
    
    # Summary
    logger.info('\n' + '=' * 80)
    logger.info('📊 Test Summary')
    logger.info('=' * 80)
    
    for test_name, result in results.items():
        status = '✅ PASS' if result else '❌ FAIL'
        logger.info(f'{test_name}: {status}')
    
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f'\nTotal: {passed}/{total} tests passed')
    
    if passed == total:
        logger.info('\n🎉 All tests passed! Your environment is ready.')
        return 0
    else:
        logger.warning(f'\n⚠️ {total - passed} test(s) failed. Please check the errors above.')
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

