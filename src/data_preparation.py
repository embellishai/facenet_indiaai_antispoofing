"""
Data preparation and loading module for face recognition dataset
Handles dataset creation, augmentation, and DataLoader setup
"""

import os
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


logger = logging.getLogger('insightface_finetune')


class FaceRecognitionDataset(Dataset):
    """
    PyTorch Dataset for face recognition
    Expects data structure: data_root/identity_name/image_files
    """
    
    def __init__(
        self,
        data_root: str,
        identity_to_label: Dict[str, int],
        image_size: Tuple[int, int] = (112, 112),
        transform: Optional[Any] = None,
        mode: str = 'train'
    ):
        """
        Initialize face recognition dataset
        
        Args:
            data_root: Root directory containing identity folders
            identity_to_label: Mapping from identity name to label ID
            image_size: Target image size (height, width)
            transform: Albumentations transform pipeline
            mode: Dataset mode (train, val, test)
        """
        self.data_root = Path(data_root)
        self.identity_to_label = identity_to_label
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
        
        logger.info(f'📊 {self.mode.upper()} dataset loaded: {len(self.image_paths)} images, {len(set(self.labels))} identities')
    
    def _load_dataset(self) -> None:
        """Load all image paths and corresponding labels"""
        logger.info(f'🔄 ***** Starting {self.mode} dataset loading...')
        
        for identity_name, label in self.identity_to_label.items():
            identity_dir = self.data_root / identity_name
            
            if not identity_dir.exists():
                logger.warning(f'⚠️ Identity directory not found: {identity_dir}')
                continue
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [
                f for f in identity_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            for img_path in image_files:
                self.image_paths.append(str(img_path))
                self.labels.append(label)
        
        logger.info(f'✅ ***** {self.mode.upper()} dataset loading done.')
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from dataset
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        
        if image is None:
            logger.error(f'❌ Failed to load image: {img_path}')
            # Return a blank image if loading fails
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def create_identity_mapping(data_root: str, min_images: int = 5) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create mapping between identity names and label IDs
    
    Args:
        data_root: Root directory containing identity folders
        min_images: Minimum number of images required per identity
        
    Returns:
        Tuple of (identity_to_label, label_to_identity) mappings
    """
    logger.info('🏷️ ***** Starting identity mapping creation...')
    
    data_root_path = Path(data_root)
    
    if not data_root_path.exists():
        raise FileNotFoundError(f'❌ Data root not found: {data_root}')
    
    identity_dirs = [d for d in data_root_path.iterdir() if d.is_dir()]
    
    identity_to_label = {}
    label_to_identity = {}
    label_id = 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for identity_dir in sorted(identity_dirs):
        identity_name = identity_dir.name
        
        # Count images in this identity folder
        image_files = [
            f for f in identity_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        num_images = len(image_files)
        
        if num_images >= min_images:
            identity_to_label[identity_name] = label_id
            label_to_identity[label_id] = identity_name
            label_id += 1
            logger.debug(f'📸 {identity_name}: {num_images} images -> Label {label_id - 1}')
        else:
            logger.warning(f'⚠️ Skipping {identity_name}: only {num_images} images (min: {min_images})')
    
    logger.info(f'✅ ***** Identity mapping done. Total identities: {len(identity_to_label)}')
    
    return identity_to_label, label_to_identity


def split_dataset_by_identity(
    data_root: str,
    identity_to_label: Dict[str, int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Split identities into train, validation, and test sets
    
    Args:
        data_root: Root directory containing identity folders
        identity_to_label: Mapping from identity name to label ID
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_mapping, val_mapping, test_mapping)
    """
    logger.info('🔀 ***** Starting dataset splitting...')
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, '❌ Split ratios must sum to 1.0'
    
    identities = list(identity_to_label.keys())
    
    # First split: train and temp (val + test)
    train_identities, temp_identities = train_test_split(
        identities,
        train_size=train_ratio,
        random_state=random_seed
    )
    
    # Second split: val and test from temp
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_identities, test_identities = train_test_split(
        temp_identities,
        train_size=val_size_adjusted,
        random_state=random_seed
    )
    
    train_mapping = {identity: identity_to_label[identity] for identity in train_identities}
    val_mapping = {identity: identity_to_label[identity] for identity in val_identities}
    test_mapping = {identity: identity_to_label[identity] for identity in test_identities}
    
    logger.info(f'📊 Split complete - Train: {len(train_identities)}, Val: {len(val_identities)}, Test: {len(test_identities)}')
    logger.info(f'✅ ***** Dataset splitting done.')
    
    return train_mapping, val_mapping, test_mapping


def create_augmentation_pipeline(
    image_size: Tuple[int, int],
    mode: str = 'train',
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create albumentations augmentation pipeline
    
    Args:
        image_size: Target image size (height, width)
        mode: Pipeline mode (train, val, test)
        config: Augmentation configuration dictionary
        
    Returns:
        Albumentations compose transform
    """
    if config is None:
        config = {}
    
    if mode == 'train':
        transform_list = [
            A.Resize(height=image_size[0], width=image_size[1]),
        ]
        
        if config.get('random_flip', True):
            transform_list.append(A.HorizontalFlip(p=0.5))
        
        if config.get('random_rotation', False):
            rotation_limit = config.get('random_rotation', 10)
            transform_list.append(A.Rotate(limit=rotation_limit, p=0.3))
        
        if config.get('color_jitter', True):
            transform_list.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        
        if config.get('random_crop', False):
            transform_list.append(A.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), p=0.3))
        
        transform_list.extend([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
    else:  # val or test
        transform_list = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ]
    
    transform = A.Compose(transform_list)
    
    logger.debug(f'🎨 {mode.upper()} augmentation pipeline created')
    
    return transform


def create_data_loaders(
    data_root: str,
    identity_to_label: Dict[str, int],
    batch_size: int,
    image_size: Tuple[int, int],
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    augmentation_config: Optional[Dict[str, Any]] = None,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data_root: Root directory containing identity folders
        identity_to_label: Mapping from identity name to label ID
        batch_size: Batch size for data loaders
        image_size: Target image size (height, width)
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        augmentation_config: Configuration for data augmentation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_to_identity)
    """
    logger.info('🔧 ***** Starting data loader creation...')
    
    # Split dataset
    train_mapping, val_mapping, test_mapping = split_dataset_by_identity(
        data_root=data_root,
        identity_to_label=identity_to_label,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Create label to identity mapping
    label_to_identity = {label: identity for identity, label in identity_to_label.items()}
    
    # Create augmentation pipelines
    train_transform = create_augmentation_pipeline(image_size, mode='train', config=augmentation_config)
    val_transform = create_augmentation_pipeline(image_size, mode='val')
    test_transform = create_augmentation_pipeline(image_size, mode='test')
    
    # Create datasets
    train_dataset = FaceRecognitionDataset(
        data_root=data_root,
        identity_to_label=train_mapping,
        image_size=image_size,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = FaceRecognitionDataset(
        data_root=data_root,
        identity_to_label=val_mapping,
        image_size=image_size,
        transform=val_transform,
        mode='val'
    )
    
    test_dataset = FaceRecognitionDataset(
        data_root=data_root,
        identity_to_label=test_mapping,
        image_size=image_size,
        transform=test_transform,
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f'📦 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}')
    logger.info('✅ ***** Data loader creation done.')
    
    return train_loader, val_loader, test_loader, label_to_identity


def prepare_sample_dataset(
    source_dir: str,
    sample_dir: str,
    num_identities: int = 10,
    images_per_identity: int = 20
) -> None:
    """
    Create a sample dataset for testing (optional helper function)
    
    Args:
        source_dir: Source directory with full dataset
        sample_dir: Destination directory for sample dataset
        num_identities: Number of identities to sample
        images_per_identity: Number of images per identity
    """
    logger.info('📋 ***** Starting sample dataset creation...')
    
    source_path = Path(source_dir)
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)
    
    identity_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if len(identity_dirs) < num_identities:
        logger.warning(f'⚠️ Only {len(identity_dirs)} identities available, using all')
        num_identities = len(identity_dirs)
    
    sampled_identities = np.random.choice(identity_dirs, size=num_identities, replace=False)
    
    for identity_dir in sampled_identities:
        dest_identity_dir = sample_path / identity_dir.name
        dest_identity_dir.mkdir(exist_ok=True)
        
        image_files = list(identity_dir.glob('*.jpg')) + list(identity_dir.glob('*.png'))
        
        num_to_copy = min(len(image_files), images_per_identity)
        sampled_images = np.random.choice(image_files, size=num_to_copy, replace=False)
        
        for img_file in sampled_images:
            dest_file = dest_identity_dir / img_file.name
            if not dest_file.exists():
                import shutil
                shutil.copy2(img_file, dest_file)
        
        logger.debug(f'📁 {identity_dir.name}: copied {num_to_copy} images')
    
    logger.info(f'✅ ***** Sample dataset created at {sample_dir}')

