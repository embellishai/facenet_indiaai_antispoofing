#!/usr/bin/env python3
"""
Quick test to verify data augmentation pipeline works correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data_preparation import create_augmentation_pipeline


def test_augmentation():
    """Test that augmentation pipeline can be created without errors"""
    
    print('🧪 Testing data augmentation pipeline...')
    
    image_size = (112, 112)
    
    # Test train mode with all augmentations
    config = {
        'random_flip': True,
        'random_rotation': 10,
        'color_jitter': True,
        'random_crop': True
    }
    
    try:
        train_transform = create_augmentation_pipeline(
            image_size=image_size,
            mode='train',
            config=config
        )
        print('✅ Train augmentation pipeline created successfully')
        
        # Test with a dummy image
        dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = train_transform(image=dummy_image)
        
        print(f'✅ Train transform works - output shape: {result["image"].shape}')
        
    except Exception as e:
        print(f'❌ Train augmentation failed: {e}')
        return False
    
    # Test validation mode
    try:
        val_transform = create_augmentation_pipeline(
            image_size=image_size,
            mode='val'
        )
        print('✅ Validation augmentation pipeline created successfully')
        
        dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = val_transform(image=dummy_image)
        
        print(f'✅ Val transform works - output shape: {result["image"].shape}')
        
    except Exception as e:
        print(f'❌ Validation augmentation failed: {e}')
        return False
    
    # Test test mode
    try:
        test_transform = create_augmentation_pipeline(
            image_size=image_size,
            mode='test'
        )
        print('✅ Test augmentation pipeline created successfully')
        
        dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = test_transform(image=dummy_image)
        
        print(f'✅ Test transform works - output shape: {result["image"].shape}')
        
    except Exception as e:
        print(f'❌ Test augmentation failed: {e}')
        return False
    
    print('\n🎉 All augmentation tests passed!')
    return True


if __name__ == '__main__':
    success = test_augmentation()
    sys.exit(0 if success else 1)

