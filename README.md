# InsightFace Buffalo_L Finetuning for Indian Celebrity Face Recognition

A comprehensive, production-ready codebase for finetuning InsightFace's Buffalo_L model on custom face recognition datasets, specifically designed for Indian celebrity recognition.

## 🌟 Features

- **Modular Architecture**: Clean, maintainable code following best practices
- **State-of-the-Art Model**: Buffalo_L (IResNet-100) with ArcFace/CosFace loss
- **Advanced Training**: Mixed precision training, gradient clipping, learning rate scheduling
- **Flexible Configuration**: YAML-based config system with environment variable support
- **Comprehensive Logging**: Detailed logging with emojis, TensorBoard integration
- **Production-Ready Inference**: Batch processing, face verification, visualization
- **GPU Optimized**: Designed for NVIDIA A6000 (48GB) but works with any CUDA device

## 📁 Project Structure

```
finetune_models/
├── config/
│   ├── config.yaml              # Main configuration file
│   └── model_config.yaml        # Model-specific configuration
├── data/
│   ├── raw/                     # Raw dataset (identity folders)
│   ├── processed/               # Processed data
│   └── sample/                  # Sample dataset for testing
├── models/
│   ├── checkpoints/             # Training checkpoints
│   ├── final/                   # Final trained models
│   └── cache/                   # Model cache directory
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Utility functions
│   ├── data_preparation.py      # Dataset handling
│   ├── model.py                 # Model architecture
│   ├── train.py                 # Training pipeline
│   └── inference.py             # Inference and recognition
├── logs/                        # Training logs
├── results/                     # Inference results
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd /home/raushan/codebase/ml/india_ai/finetune_models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset in the following structure:

```
data/raw/
├── celebrity_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── celebrity_2/
│   ├── image_001.jpg
│   └── ...
└── celebrity_N/
    └── ...
```

**Requirements**:
- At least 5 images per identity (configurable in `config.yaml`)
- Face-cropped images preferred (112x112 pixels)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

### 3. Configure Training

Edit `config/config.yaml` to customize your training:

```yaml
# Key configurations
paths:
  data_root: 'data/raw'           # Your dataset path

dataset:
  min_images_per_identity: 5      # Minimum images per person
  image_size: [112, 112]          # Input size

training:
  batch_size: 64                  # Adjust based on GPU memory
  num_epochs: 100
  learning_rate: 0.0001
  mixed_precision: true           # Enable for faster training

hardware:
  device: 'cuda'
  gpu_id: 0
```

### 4. Start Training

```bash
# Basic training
python src/train.py --config config/config.yaml

# Resume from checkpoint
python src/train.py --config config/config.yaml --resume models/checkpoints/latest_checkpoint.pth

# Custom random seed
python src/train.py --config config/config.yaml --seed 123
```

### 5. Run Inference

```bash
# Recognize single image
python src/inference.py \
    --model models/final/final_model.pth \
    --image path/to/test_image.jpg \
    --visualize

# Recognize multiple images
python src/inference.py \
    --model models/final/final_model.pth \
    --images image1.jpg image2.jpg image3.jpg

# Process entire directory
python src/inference.py \
    --model models/final/final_model.pth \
    --image_dir path/to/test_images/

# Face verification (1:1 matching)
python src/inference.py \
    --model models/final/final_model.pth \
    --verify image1.jpg image2.jpg

# Save results to JSON
python src/inference.py \
    --model models/final/final_model.pth \
    --image test.jpg \
    --output results/predictions.json
```

## ⚙️ Configuration Reference

### Main Config (`config/config.yaml`)

| Section | Parameter | Description | Default |
|---------|-----------|-------------|---------|
| **dataset** | `train_split` | Training data ratio | 0.8 |
| | `val_split` | Validation data ratio | 0.1 |
| | `test_split` | Test data ratio | 0.1 |
| | `image_size` | Input image dimensions | [112, 112] |
| **training** | `batch_size` | Batch size | 64 |
| | `num_epochs` | Total epochs | 100 |
| | `learning_rate` | Initial learning rate | 0.0001 |
| | `optimizer` | Optimizer type | adam |
| | `mixed_precision` | Use FP16 training | true |
| **model** | `embedding_size` | Face embedding dimension | 512 |
| | `freeze_epochs` | Epochs to freeze backbone | 5 |
| **loss** | `type` | Loss function | arcface |
| | `scale` | Feature scale (s) | 64.0 |
| | `margin` | Angular margin (m) | 0.5 |

## 🎯 Model Architecture

```
Input Image (3 x 112 x 112)
        ↓
IResNet-100 Backbone
        ↓
Face Embedding (512-dim)
        ↓
ArcFace/CosFace Loss Head
        ↓
Identity Classification
```

- **Backbone**: IResNet-100 (Improved ResNet)
- **Embedding**: 512-dimensional L2-normalized features
- **Loss**: ArcFace (additive angular margin) or CosFace (cosine margin)
- **Parameters**: ~65M total

## 📊 Training Features

### Data Augmentation
- Horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Random crop and resize
- Normalization

### Training Techniques
- **Mixed Precision Training**: Faster training with FP16
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpoint Management**: Automatic best model saving
- **TensorBoard Logging**: Real-time training visualization
- **Progressive Unfreezing**: Freeze backbone initially, unfreeze later

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Open `http://localhost:6006` to view:
- Training/validation loss curves
- Accuracy metrics
- Learning rate schedule
- Model architecture

### Log Files

Training logs are saved in `logs/` with timestamps:
```
logs/training_20250101_120000.log
```

## 🔍 Inference Modes

### 1. Face Recognition (1:N)
Identify a person from a database of known identities.

```python
# Example output
{
  "best_match": {
    "identity": "Shah Rukh Khan",
    "confidence": 95.34,
    "is_recognized": true
  },
  "top_matches": [
    {"identity": "Shah Rukh Khan", "confidence": 95.34},
    {"identity": "Aamir Khan", "confidence": 87.21},
    ...
  ]
}
```

### 2. Face Verification (1:1)
Verify if two faces belong to the same person.

```python
# Example output
{
  "similarity": 0.923,
  "confidence": 92.3,
  "is_same_person": true,
  "threshold": 0.6
}
```

## 🎛️ Advanced Usage

### Custom Dataset Preparation

```python
from src.data_preparation import create_identity_mapping, prepare_sample_dataset

# Create sample dataset for testing
prepare_sample_dataset(
    source_dir='data/raw',
    sample_dir='data/sample',
    num_identities=10,
    images_per_identity=20
)

# Create identity mappings
identity_to_label, label_to_identity = create_identity_mapping(
    data_root='data/raw',
    min_images=5
)
```

### Custom Model Loading

```python
from src.model import load_insightface_buffalo_model
import torch

device = torch.device('cuda')

model = load_insightface_buffalo_model(
    num_classes=100,
    config=config,
    pretrained_path='path/to/pretrained.pth',
    device=device
)
```

### Programmatic Inference

```python
from src.inference import FaceRecognizer
import torch

device = torch.device('cuda')

recognizer = FaceRecognizer(
    model_path='models/final/final_model.pth',
    device=device,
    threshold=0.6
)

# Recognize face
result = recognizer.recognize_from_path('test_image.jpg')
print(f"Identity: {result['best_match']['identity']}")
print(f"Confidence: {result['best_match']['confidence']:.2f}%")

# Verify faces
verification = recognizer.verify_faces('image1.jpg', 'image2.jpg')
print(f"Same person: {verification['is_same_person']}")
```

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 32  # or 16, 8
  
# Or disable mixed precision
training:
  mixed_precision: false
```

### Slow Training

```yaml
# Increase number of workers
training:
  num_workers: 8  # adjust based on CPU cores
  
# Enable cudnn benchmark
hardware:
  benchmark: true
```

### Poor Recognition Accuracy

1. **More training data**: Collect more images per identity (20+ recommended)
2. **Longer training**: Increase number of epochs
3. **Data quality**: Ensure faces are properly cropped and aligned
4. **Adjust threshold**: Lower threshold for more lenient matching

## 📝 Best Practices

1. **Data Quality**: Use high-quality, well-lit face images
2. **Data Balance**: Similar number of images per identity
3. **Face Alignment**: Pre-align faces to 112x112 for best results
4. **Validation**: Monitor validation metrics to prevent overfitting
5. **Checkpointing**: Regularly save checkpoints during long training runs
6. **Experimentation**: Try different loss types (ArcFace vs CosFace)

## 🔧 Hardware Requirements

### Minimum
- GPU: 8GB VRAM (NVIDIA GTX 1080 or better)
- RAM: 16GB
- Storage: 20GB

### Recommended (Current Setup)
- GPU: NVIDIA A6000 (48GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

### Training Time Estimates
- 10 identities, 20 images each: ~30 minutes (A6000)
- 100 identities, 50 images each: ~5 hours (A6000)
- 1000 identities, 100 images each: ~2 days (A6000)

## 📚 References

- **InsightFace**: [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- **ArcFace Paper**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **CosFace Paper**: [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)

## 🤝 Contributing

This codebase follows strict coding principles:
- **DRY**: Don't Repeat Yourself
- **Modularity**: Each function does one thing well
- **Type Hints**: All functions have type annotations
- **Logging**: Comprehensive logging with emojis
- **Documentation**: Clear docstrings for all functions

## 📄 License

This project is for educational and research purposes.

## ✨ Author

Created with ❤️ for face recognition research

---

**Happy Training! 🚀**

For issues or questions, please check the logs in the `logs/` directory or review the configuration files.

