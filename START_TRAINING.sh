#!/bin/bash

# Simple script to start training immediately
# Use this if you've already run QUICK_START.sh

set -e

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Starting Training - IMFDB FR Dataset                                 ║"
echo "║  100 Celebrities, 34,513 Images                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /home/raushan/codebase/ml/india_ai/finetune_models

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: bash QUICK_START.sh first"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if dependencies are installed
python -c "import torch, yaml, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed!"
    echo "   Please run: bash QUICK_START.sh first"
    exit 1
fi

echo "✅ Environment ready"
echo ""

# Show GPU info
echo "🎮 GPU Information:"
python3 << PYEOF
import torch
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  CUDA not available")
PYEOF

echo ""
echo "🚀 Starting training..."
echo "   Config: config/config.yaml"
echo "   Dataset: IMFDB FR (100 celebrities)"
echo "   Epochs: 100"
echo "   Batch size: 64"
echo ""
echo "📊 To monitor training:"
echo "   tensorboard --logdir logs/tensorboard"
echo ""
echo "⏱️  Estimated time: 15-20 hours"
echo ""

# Start training
python src/train.py --config config/config.yaml

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Model saved: models/final/final_model.pth"
echo "📊 Logs: logs/"
echo ""
echo "🔍 Test your model:"
echo "   python src/inference.py --model models/final/final_model.pth --image test.jpg"
echo ""
