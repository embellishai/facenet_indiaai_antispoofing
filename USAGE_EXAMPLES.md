# Usage Examples

This document provides practical examples for using the InsightFace Buffalo_L finetuning system.

## Table of Contents
- [Installation and Setup](#installation-and-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Advanced Usage](#advanced-usage)

---

## Installation and Setup

### Quick Setup

```bash
# Clone/navigate to project
cd /home/raushan/codebase/ml/india_ai/finetune_models

# Run setup script
bash scripts/setup.sh

# Or manual setup:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run quick test to verify everything is installed correctly
python scripts/quick_test.py
```

Expected output:
```
🧪 Running Installation Tests
✅ Import Test: PASS
✅ CUDA Test: PASS
✅ Model Creation Test: PASS
✅ OpenCV Test: PASS
✅ Config Loading Test: PASS
🎉 All tests passed! Your environment is ready.
```

---

## Data Preparation

### Organize Your Dataset

```bash
# Your dataset should look like this:
data/raw/
├── shah_rukh_khan/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── priyanka_chopra/
│   ├── img001.jpg
│   └── ...
└── ...
```

### Analyze Your Dataset

```bash
# Analyze dataset structure and get statistics
python scripts/prepare_data_example.py --data_root data/raw

# Create a sample dataset for quick testing
python scripts/prepare_data_example.py \
    --data_root data/raw \
    --create_sample \
    --sample_dir data/sample
```

Output:
```
📊 Dataset Statistics
Total identities: 100
Valid identities (>= 5 images): 95
Total images: 5,234
Average images per identity: 55.1
```

---

## Training

### Basic Training

```bash
# Train with default configuration
python src/train.py --config config/config.yaml
```

### Training on Sample Dataset (Quick Test)

```bash
# Edit config to use sample data
# Then train for just a few epochs to test
python src/train.py --config config/config.yaml
```

### Custom Configuration Training

```bash
# Train with custom settings
python src/train.py \
    --config config/config.yaml \
    --seed 123
```

### Resume Training from Checkpoint

```bash
# Resume from the latest checkpoint
python src/train.py \
    --config config/config.yaml \
    --resume models/checkpoints/latest_checkpoint.pth
```

### Monitor Training with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs/tensorboard

# Open browser to http://localhost:6006
```

### Expected Training Output

```
🚀 ***** Starting training pipeline...
🎮 Using GPU: NVIDIA A6000 (48.00 GB)
📊 TRAIN dataset loaded: 4000 images, 80 identities
📊 VAL dataset loaded: 500 images, 10 identities
🔢 Model parameters - Total: 65,234,567, Trainable: 65,234,567
🎬 Starting training for 100 epochs

================================================================================
🔄 ***** Starting Epoch 1/100...
📚 Learning rate: 0.000100
Epoch 1: 100%|████████| 62/62 [00:45<00:00]
✅ ***** Epoch 1 training done.
📊 Train - Loss: 4.2341, Accuracy: 12.34%
🎯 Validation - Loss: 3.8765, Accuracy: 15.67%
🏆 New best model! accuracy: 15.67
💾 Checkpoint saved to models/checkpoints/best_checkpoint.pth
⏱️ Epoch time: 1m 23s
```

---

## Inference

### Single Image Recognition

```bash
# Recognize a single face
python src/inference.py \
    --model models/final/final_model.pth \
    --image test_images/person1.jpg
```

Output:
```
================================================================================
FACE RECOGNITION RESULT
================================================================================
Image: test_images/person1.jpg

Best Match:
  Identity: Shah Rukh Khan
  Confidence: 95.34%
  Recognized: ✅ YES

Top 5 Matches:
  1. Shah Rukh Khan: 95.34%
  2. Aamir Khan: 87.21%
  3. Salman Khan: 82.15%
  4. Hrithik Roshan: 78.90%
  5. Akshay Kumar: 75.43%
```

### Visualize Recognition Results

```bash
# Add visualization to the output
python src/inference.py \
    --model models/final/final_model.pth \
    --image test_images/person1.jpg \
    --visualize

# This creates person1_annotated.jpg with the prediction overlaid
```

### Batch Recognition (Multiple Images)

```bash
# Recognize multiple images
python src/inference.py \
    --model models/final/final_model.pth \
    --images img1.jpg img2.jpg img3.jpg img4.jpg
```

### Directory Recognition

```bash
# Process all images in a directory
python src/inference.py \
    --model models/final/final_model.pth \
    --image_dir test_images/

# Save results to JSON
python src/inference.py \
    --model models/final/final_model.pth \
    --image_dir test_images/ \
    --output results/predictions.json
```

### Face Verification (1:1 Matching)

```bash
# Check if two images are of the same person
python src/inference.py \
    --model models/final/final_model.pth \
    --verify image1.jpg image2.jpg
```

Output:
```
================================================================================
FACE VERIFICATION RESULT
================================================================================
Image 1: image1.jpg
Image 2: image2.jpg
Similarity: 0.9234 (92.34%)
Same Person: ✅ YES
```

### Custom Threshold

```bash
# Use a custom similarity threshold (default: 0.6)
python src/inference.py \
    --model models/final/final_model.pth \
    --image test.jpg \
    --threshold 0.7  # More strict
```

---

## Advanced Usage

### Programmatic Training

```python
from src.train import train
from src.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Customize settings
config['training']['batch_size'] = 32
config['training']['num_epochs'] = 50

# Start training
train(config=config)
```

### Programmatic Inference

```python
from src.inference import FaceRecognizer
import torch

# Initialize recognizer
device = torch.device('cuda')
recognizer = FaceRecognizer(
    model_path='models/final/final_model.pth',
    device=device,
    threshold=0.6
)

# Recognize face
result = recognizer.recognize_from_path('test.jpg')

print(f"Identity: {result['best_match']['identity']}")
print(f"Confidence: {result['best_match']['confidence']:.2f}%")

# Face verification
verification = recognizer.verify_faces('img1.jpg', 'img2.jpg')
print(f"Same person: {verification['is_same_person']}")
print(f"Similarity: {verification['similarity']:.4f}")
```

### Extract Embeddings Only

```python
from src.inference import FaceRecognizer
import torch
import cv2

device = torch.device('cuda')
recognizer = FaceRecognizer(
    model_path='models/final/final_model.pth',
    device=device
)

# Load image
image = cv2.imread('test.jpg')

# Extract 512-dimensional embedding
embedding = recognizer.extract_embedding(image)
print(f"Embedding shape: {embedding.shape}")  # (512,)
print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")  # Should be ~1.0
```

### Compare Multiple Faces

```python
from src.inference import FaceRecognizer
import torch
import cv2

device = torch.device('cuda')
recognizer = FaceRecognizer(
    model_path='models/final/final_model.pth',
    device=device
)

# Extract embeddings
image1 = cv2.imread('person1.jpg')
image2 = cv2.imread('person2.jpg')

emb1 = recognizer.extract_embedding(image1)
emb2 = recognizer.extract_embedding(image2)

# Compute similarity
similarity = recognizer.compute_similarity(emb1, emb2)
print(f"Similarity: {similarity:.4f}")
```

### Custom Data Augmentation

Edit `config/config.yaml`:

```yaml
dataset:
  augmentation:
    enabled: true
    random_flip: true
    color_jitter: true
    random_rotation: 15  # Increase rotation
    random_crop: true
```

### Custom Loss Configuration

Edit `config/config.yaml`:

```yaml
loss:
  type: 'cosface'  # Try CosFace instead of ArcFace
  scale: 64.0
  margin: 0.35
```

### Multiple GPU Training

Currently single GPU. For multi-GPU, modify `src/train.py`:

```python
# Wrap model with DataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

---

## Troubleshooting Examples

### Out of Memory

```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 32  # or 16, 8
  mixed_precision: false  # Disable if still issues
```

### Training Too Slow

```yaml
# Optimize settings
training:
  num_workers: 16  # Increase data loading workers
  pin_memory: true

hardware:
  benchmark: true  # Enable cudnn benchmarking
```

### Poor Accuracy

1. **Check data quality**:
```bash
python scripts/prepare_data_example.py --data_root data/raw
```

2. **Increase training**:
```yaml
training:
  num_epochs: 200  # Train longer
  learning_rate: 0.00005  # Try lower LR
```

3. **Adjust loss**:
```yaml
loss:
  margin: 0.5  # Try 0.3 or 0.7
  scale: 64.0  # Try 32.0 or 128.0
```

---

## Production Deployment Example

```python
# production_api.py
from flask import Flask, request, jsonify
from src.inference import FaceRecognizer
import torch
import cv2
import numpy as np

app = Flask(__name__)

# Load model once at startup
device = torch.device('cuda')
recognizer = FaceRecognizer(
    model_path='models/final/final_model.pth',
    device=device,
    threshold=0.6
)

@app.route('/recognize', methods=['POST'])
def recognize():
    # Get image from request
    file = request.files['image']
    
    # Read image
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    # Recognize
    result = recognizer.recognize_face(image)
    
    return jsonify(result['best_match'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Tips and Best Practices

1. **Start Small**: Test with sample dataset first
2. **Monitor Training**: Always watch TensorBoard
3. **Save Checkpoints**: Resume training if interrupted
4. **Data Quality**: Better data > More data
5. **Experiment**: Try different loss functions and hyperparameters
6. **Validation**: Always validate on unseen identities

---

For more information, see [README.md](README.md) or check the logs in `logs/` directory.

