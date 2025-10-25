#!/usr/bin/env python3
"""
Example script for preparing and exploring your dataset
Run this before training to verify your data is properly structured
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, load_config
from src.data_preparation import (
    create_identity_mapping,
    split_dataset_by_identity,
    prepare_sample_dataset
)

logger = setup_logging('logs', log_level='INFO')


def analyze_dataset(data_root: str, min_images: int = 5):
    """
    Analyze dataset structure and provide statistics
    
    Args:
        data_root: Path to dataset root directory
        min_images: Minimum images per identity
    """
    logger.info('🔍 ***** Starting dataset analysis...')
    
    data_root_path = Path(data_root)
    
    if not data_root_path.exists():
        logger.error(f'❌ Data root not found: {data_root}')
        return
    
    # Get all identity directories
    identity_dirs = [d for d in data_root_path.iterdir() if d.is_dir()]
    
    logger.info(f'📁 Found {len(identity_dirs)} identity directories')
    
    # Count images per identity
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    identity_stats = []
    
    for identity_dir in identity_dirs:
        image_files = [
            f for f in identity_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        num_images = len(image_files)
        
        identity_stats.append({
            'name': identity_dir.name,
            'num_images': num_images,
            'valid': num_images >= min_images
        })
    
    # Sort by number of images
    identity_stats.sort(key=lambda x: x['num_images'], reverse=True)
    
    # Statistics
    valid_identities = [stat for stat in identity_stats if stat['valid']]
    invalid_identities = [stat for stat in identity_stats if not stat['valid']]
    
    total_images = sum(stat['num_images'] for stat in identity_stats)
    valid_images = sum(stat['num_images'] for stat in valid_identities)
    
    logger.info(f'\n{"=" * 80}')
    logger.info('📊 Dataset Statistics')
    logger.info(f'{"=" * 80}')
    logger.info(f'Total identities: {len(identity_stats)}')
    logger.info(f'Valid identities (>= {min_images} images): {len(valid_identities)}')
    logger.info(f'Invalid identities (< {min_images} images): {len(invalid_identities)}')
    logger.info(f'Total images: {total_images}')
    logger.info(f'Valid images: {valid_images}')
    
    if valid_identities:
        avg_images = sum(stat['num_images'] for stat in valid_identities) / len(valid_identities)
        min_valid = min(stat['num_images'] for stat in valid_identities)
        max_valid = max(stat['num_images'] for stat in valid_identities)
        
        logger.info(f'\nValid identities - Images per identity:')
        logger.info(f'  Average: {avg_images:.1f}')
        logger.info(f'  Minimum: {min_valid}')
        logger.info(f'  Maximum: {max_valid}')
    
    # Show top 10 identities
    logger.info(f'\n📈 Top 10 Identities by Image Count:')
    for i, stat in enumerate(identity_stats[:10], 1):
        status = '✅' if stat['valid'] else '❌'
        logger.info(f'  {i:2d}. {status} {stat["name"]}: {stat["num_images"]} images')
    
    # Show invalid identities if any
    if invalid_identities:
        logger.info(f'\n⚠️ Identities with insufficient images (will be skipped):')
        for stat in invalid_identities[:10]:  # Show first 10
            logger.info(f'  - {stat["name"]}: {stat["num_images"]} images (need {min_images})')
        
        if len(invalid_identities) > 10:
            logger.info(f'  ... and {len(invalid_identities) - 10} more')
    
    logger.info(f'\n{"=" * 80}\n')
    logger.info('✅ ***** Dataset analysis done.')
    
    return {
        'total_identities': len(identity_stats),
        'valid_identities': len(valid_identities),
        'total_images': total_images,
        'valid_images': valid_images,
        'identity_stats': identity_stats
    }


def create_sample_for_testing(data_root: str, sample_dir: str = 'data/sample'):
    """
    Create a small sample dataset for quick testing
    
    Args:
        data_root: Source dataset directory
        sample_dir: Destination for sample dataset
    """
    logger.info('📋 Creating sample dataset for testing...')
    
    try:
        prepare_sample_dataset(
            source_dir=data_root,
            sample_dir=sample_dir,
            num_identities=10,
            images_per_identity=20
        )
        logger.info(f'✅ Sample dataset created at {sample_dir}')
        logger.info('💡 You can test training on this sample before using the full dataset')
    except Exception as e:
        logger.error(f'❌ Failed to create sample dataset: {e}')


def verify_data_splits(data_root: str, min_images: int = 5):
    """
    Verify that data can be properly split into train/val/test
    
    Args:
        data_root: Path to dataset root directory
        min_images: Minimum images per identity
    """
    logger.info('🔀 ***** Verifying data splits...')
    
    # Create identity mapping
    identity_to_label, label_to_identity = create_identity_mapping(
        data_root=data_root,
        min_images=min_images
    )
    
    # Split dataset
    train_mapping, val_mapping, test_mapping = split_dataset_by_identity(
        data_root=data_root,
        identity_to_label=identity_to_label,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
    
    logger.info(f'\n📊 Data Split Summary:')
    logger.info(f'  Train identities: {len(train_mapping)}')
    logger.info(f'  Validation identities: {len(val_mapping)}')
    logger.info(f'  Test identities: {len(test_mapping)}')
    logger.info(f'  Total: {len(identity_to_label)}')
    
    logger.info('✅ ***** Data splits verified.')


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare and analyze dataset')
    parser.add_argument('--data_root', type=str, default='data/raw', help='Path to dataset root')
    parser.add_argument('--min_images', type=int, default=5, help='Minimum images per identity')
    parser.add_argument('--create_sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--sample_dir', type=str, default='data/sample', help='Sample dataset directory')
    
    args = parser.parse_args()
    
    logger.info('=' * 80)
    logger.info('📦 Dataset Preparation Tool')
    logger.info('=' * 80)
    
    # Analyze dataset
    stats = analyze_dataset(args.data_root, args.min_images)
    
    if stats is None:
        logger.error('❌ Failed to analyze dataset. Please check the data_root path.')
        return
    
    # Verify splits
    if stats['valid_identities'] > 0:
        verify_data_splits(args.data_root, args.min_images)
    
    # Create sample if requested
    if args.create_sample:
        create_sample_for_testing(args.data_root, args.sample_dir)
    
    # Recommendations
    logger.info(f'\n{"=" * 80}')
    logger.info('💡 Recommendations')
    logger.info(f'{"=" * 80}')
    
    if stats['valid_identities'] < 10:
        logger.warning('⚠️ Very few identities! Collect more data for better results.')
    elif stats['valid_identities'] < 50:
        logger.info('📊 Small dataset. Training should be quick.')
    elif stats['valid_identities'] < 200:
        logger.info('📊 Medium dataset. Good for most applications.')
    else:
        logger.info('📊 Large dataset. Training may take several hours.')
    
    avg_images = stats['valid_images'] / stats['valid_identities'] if stats['valid_identities'] > 0 else 0
    
    if avg_images < 10:
        logger.warning('⚠️ Few images per identity. Try to collect at least 20 per person.')
    elif avg_images < 30:
        logger.info('✅ Decent number of images per identity.')
    else:
        logger.info('✅ Great! Lots of images per identity for robust training.')
    
    logger.info(f'\n{"=" * 80}')
    logger.info('🚀 Next Steps:')
    logger.info('  1. Review the statistics above')
    logger.info('  2. Adjust min_images_per_identity in config/config.yaml if needed')
    logger.info('  3. Start training with: python src/train.py --config config/config.yaml')
    logger.info(f'{"=" * 80}\n')


if __name__ == '__main__':
    main()

