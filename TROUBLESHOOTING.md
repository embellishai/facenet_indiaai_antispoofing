# Troubleshooting Guide

## Common Issues and Solutions

### ✅ FIXED: Mixed Precision dtype Mismatch

**Error:**
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated
```

**Solution:** Fixed in `src/train.py` lines 192, 215, 270
- Updated autocast API: `torch.amp.autocast('cuda')` instead of `torch.cuda.amp.autocast()`
- Convert embeddings to float32 before accuracy calculation in mixed precision mode
- Ensures dtype compatibility between embeddings and model weights

This was due to PyTorch mixed precision creating FP16 embeddings while model weights remain in FP32. **Already fixed!**

---

### ✅ FIXED: RandomResizedCrop Error

**Error:**
```
ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.9, 1.0), 'p'...: None, 'strict': False}
```

**Solution:** Fixed in `src/data_preparation.py` line 257
- Changed from: `A.RandomResizedCrop(height=..., width=...)`
- Changed to: `A.RandomResizedCrop(size=...)`

This was due to albumentations API changes. **Already fixed!**

---

## Other Potential Issues

### Out of Memory (OOM) Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** in `config/config.yaml`:
```yaml
training:
  batch_size: 32  # Down from 64
```

2. **Disable mixed precision** (uses more memory):
```yaml
training:
  mixed_precision: false
```

3. **Reduce number of workers**:
```yaml
training:
  num_workers: 4  # Down from 8
```

---

### Slow Training

**Symptoms:**
- Training taking much longer than expected

**Solutions:**

1. **Enable mixed precision** in `config/config.yaml`:
```yaml
training:
  mixed_precision: true
```

2. **Increase number of workers**:
```yaml
training:
  num_workers: 16  # Increase based on CPU cores
```

3. **Enable cudnn benchmark**:
```yaml
hardware:
  benchmark: true
```

---

### Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt
```

---

### CUDA Not Available

**Symptoms:**
```
⚠️ CUDA not available, using CPU
```

**Solutions:**

1. **Check PyTorch CUDA installation**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Reinstall PyTorch with CUDA**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Data Loading Errors

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**

1. **Verify data path** in `config/config.yaml`:
```yaml
paths:
  data_root: 'data/raw/IMFDB FR dataset/IMFDB FR dataset'
```

2. **Check dataset structure**:
```bash
python scripts/prepare_data_example.py --data_root "data/raw/IMFDB FR dataset/IMFDB FR dataset"
```

---

### Poor Accuracy

**Symptoms:**
- Validation accuracy not improving
- Loss not decreasing

**Solutions:**

1. **Train longer**:
```yaml
training:
  num_epochs: 200  # Increase from 100
```

2. **Adjust learning rate**:
```yaml
training:
  learning_rate: 0.00005  # Try lower
```

3. **Change loss function**:
```yaml
loss:
  type: 'cosface'  # Try CosFace instead of ArcFace
  margin: 0.35
```

4. **Adjust loss margin**:
```yaml
loss:
  margin: 0.3  # Try different values: 0.3, 0.4, 0.5, 0.6
```

---

### Checkpoint Loading Errors

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

1. **Check checkpoint path**:
```bash
ls -lh models/checkpoints/
```

2. **Start fresh training** (if checkpoint is corrupted):
```bash
rm models/checkpoints/*.pth
python src/train.py --config config/config.yaml
```

---

### TensorBoard Not Working

**Symptoms:**
- Can't access TensorBoard at localhost:6006

**Solutions:**

1. **Check if TensorBoard is running**:
```bash
ps aux | grep tensorboard
```

2. **Start TensorBoard**:
```bash
tensorboard --logdir logs/tensorboard --bind_all
```

3. **Use different port**:
```bash
tensorboard --logdir logs/tensorboard --port 6007
```

---

### Image Loading Errors

**Symptoms:**
```
❌ Failed to load image: /path/to/image.jpg
```

**Solutions:**

1. **Check image format**:
```bash
file path/to/image.jpg
```

2. **Verify images are not corrupted**:
```bash
cd "data/raw/IMFDB FR dataset/IMFDB FR dataset"
find . -name "*.jpg" -exec file {} \; | grep -v "JPEG"
```

3. **Remove corrupted images**:
```bash
# The code will skip corrupted images automatically
# Check logs for warnings
```

---

## Quick Diagnostics

### Test Environment
```bash
python scripts/quick_test.py
```

### Test Augmentation Pipeline
```bash
python scripts/test_augmentation.py
```

### Verify Dataset
```bash
python scripts/prepare_data_example.py --data_root "data/raw/IMFDB FR dataset/IMFDB FR dataset"
```

### Check GPU Status
```bash
nvidia-smi
```

### Check Disk Space
```bash
df -h
```

---

## Getting Help

If you encounter an issue not listed here:

1. **Check the logs**:
```bash
tail -f logs/training_*.log
```

2. **Check TensorBoard**:
```bash
tensorboard --logdir logs/tensorboard
```

3. **Run diagnostics**:
```bash
python scripts/quick_test.py
```

4. **Check GitHub Issues**:
   - InsightFace: https://github.com/deepinsight/insightface
   - Albumentations: https://github.com/albumentations-team/albumentations

---

## Contact

For project-specific issues, check:
- `README.md` - Complete documentation
- `USAGE_EXAMPLES.md` - Practical examples
- `DATASET_INFO.md` - Dataset information

---

**Last Updated**: After fixing RandomResizedCrop API issue

