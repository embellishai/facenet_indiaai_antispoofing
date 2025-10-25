"""
Model module for InsightFace buffalo_l finetuning
Handles model loading, architecture setup, and loss functions
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


logger = logging.getLogger('insightface_finetune')


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for face recognition
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """
        Initialize ArcFace loss
        
        Args:
            embedding_size: Size of face embeddings
            num_classes: Number of identity classes
            scale: Feature scale s
            margin: Angular margin m
            easy_margin: Use easy margin
        """
        super(ArcFaceLoss, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Compute cos(m) and sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_multiplier = math.sin(math.pi - margin) * margin
        
        logger.info(f'🎯 ArcFace loss initialized: scale={scale}, margin={margin}, num_classes={num_classes}')
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss
        
        Args:
            embeddings: Face embeddings (batch_size, embedding_size)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            ArcFace loss value
        """
        # Normalize embeddings and weights
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings_normalized, weight_normalized)
        
        # Compute sine from cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.margin_multiplier)
        
        # Convert labels to one-hot
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace loss for face recognition
    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """
    
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.35
    ):
        """
        Initialize CosFace loss
        
        Args:
            embedding_size: Size of face embeddings
            num_classes: Number of identity classes
            scale: Feature scale s
            margin: Cosine margin m
        """
        super(CosFaceLoss, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        logger.info(f'🎯 CosFace loss initialized: scale={scale}, margin={margin}, num_classes={num_classes}')
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute CosFace loss
        
        Args:
            embeddings: Face embeddings (batch_size, embedding_size)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            CosFace loss value
        """
        # Normalize embeddings and weights
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings_normalized, weight_normalized)
        
        # Apply margin to target class
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = cosine - one_hot * self.margin
        output *= self.scale
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class IResNet(nn.Module):
    """
    Improved ResNet backbone for face recognition
    Based on InsightFace architecture
    """
    
    def __init__(
        self,
        depth: int = 100,
        embedding_size: int = 512,
        dropout: float = 0.0
    ):
        """
        Initialize IResNet backbone
        
        Args:
            depth: Network depth (50, 100, 152)
            embedding_size: Output embedding size
            dropout: Dropout probability
        """
        super(IResNet, self).__init__()
        
        self.depth = depth
        self.embedding_size = embedding_size
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        
        # Residual blocks
        if depth == 50:
            units = [3, 4, 14, 3]
        elif depth == 100:
            units = [3, 13, 30, 3]
        elif depth == 152:
            units = [3, 8, 36, 3]
        else:
            raise ValueError(f'❌ Unsupported depth: {depth}')
        
        self.layer1 = self._make_layer(64, 64, units[0], stride=2)
        self.layer2 = self._make_layer(64, 128, units[1], stride=2)
        self.layer3 = self._make_layer(128, 256, units[2], stride=2)
        self.layer4 = self._make_layer(256, 512, units[3], stride=2)
        
        # Output layers
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
        logger.info(f'🏗️ IResNet-{depth} backbone initialized with embedding size {embedding_size}')
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks"""
        layers = []
        
        # First block with stride
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Face embeddings (batch_size, embedding_size)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for IResNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        
        return out


class FaceRecognitionModel(nn.Module):
    """
    Complete face recognition model with backbone and loss head
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        depth: int = 100,
        dropout: float = 0.0,
        loss_type: str = 'arcface',
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """
        Initialize face recognition model
        
        Args:
            num_classes: Number of identity classes
            embedding_size: Size of face embeddings
            depth: Backbone network depth
            dropout: Dropout probability
            loss_type: Type of loss (arcface, cosface)
            scale: Feature scale for loss
            margin: Margin for loss
            easy_margin: Use easy margin (ArcFace only)
        """
        super(FaceRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.loss_type = loss_type
        
        # Backbone network
        self.backbone = IResNet(depth=depth, embedding_size=embedding_size, dropout=dropout)
        
        # Loss head
        if loss_type.lower() == 'arcface':
            self.loss_head = ArcFaceLoss(
                embedding_size=embedding_size,
                num_classes=num_classes,
                scale=scale,
                margin=margin,
                easy_margin=easy_margin
            )
        elif loss_type.lower() == 'cosface':
            self.loss_head = CosFaceLoss(
                embedding_size=embedding_size,
                num_classes=num_classes,
                scale=scale,
                margin=margin
            )
        else:
            raise ValueError(f'❌ Unsupported loss type: {loss_type}')
        
        logger.info(f'🤖 Face recognition model created: {num_classes} classes, {loss_type} loss')
    
    def forward(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            images: Input images (batch_size, 3, height, width)
            labels: Ground truth labels (batch_size,) - optional for inference
            
        Returns:
            Tuple of (embeddings, loss)
        """
        embeddings = self.backbone(images)
        
        if labels is not None:
            loss = self.loss_head(embeddings, labels)
            return embeddings, loss
        else:
            return embeddings, None
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('❄️ Backbone frozen')
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info('🔥 Backbone unfrozen')


def load_insightface_buffalo_model(
    num_classes: int,
    config: Dict[str, Any],
    pretrained_path: Optional[str] = None,
    device: torch.device = torch.device('cuda')
) -> FaceRecognitionModel:
    """
    Load InsightFace buffalo_l model with optional pretrained weights
    
    Args:
        num_classes: Number of identity classes
        config: Model configuration dictionary
        pretrained_path: Path to pretrained weights (optional)
        device: Device to load model to
        
    Returns:
        Initialized face recognition model
    """
    logger.info('🚀 ***** Starting model initialization...')
    
    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    
    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_size=model_config.get('embedding_size', 512),
        depth=100,  # buffalo_l uses IResNet-100
        dropout=model_config.get('dropout', 0.0),
        loss_type=loss_config.get('type', 'arcface'),
        scale=loss_config.get('scale', 64.0),
        margin=loss_config.get('margin', 0.5),
        easy_margin=loss_config.get('easy_margin', False)
    )
    
    # Load pretrained weights if provided
    if pretrained_path:
        logger.info(f'📥 Loading pretrained weights from {pretrained_path}')
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load weights (may have mismatched keys for classification head)
            model.load_state_dict(state_dict, strict=False)
            logger.info('✅ Pretrained weights loaded successfully')
        except Exception as e:
            logger.warning(f'⚠️ Failed to load pretrained weights: {e}')
            logger.info('🔄 Continuing with random initialization')
    
    model = model.to(device)
    
    logger.info('✅ ***** Model initialization done.')
    
    return model

