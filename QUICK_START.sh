#!/bin/bash

# Quick Start Script for IMFDB FR Dataset Training
# This script will set up the environment and start training

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  InsightFace Buffalo_L Finetuning - Quick Start                       ║"
echo "║  IMFDB FR Dataset: 100 Celebrities, 34,513 Images                     ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_DIR="/home/raushan/codebase/ml/india_ai/finetune_models"
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

echo "✅ All dependencies installed"
echo ""

# Check CUDA availability
echo "🎮 Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available, will use CPU (very slow)")
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete! Choose an option:                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "1️⃣  Quick Test (10 epochs, ~2 hours)"
echo "2️⃣  Standard Training (100 epochs, ~15-20 hours)"
echo "3️⃣  Full Training (200 epochs, ~30-40 hours)"
echo "4️⃣  Analyze Dataset Only"
echo "5️⃣  Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Quick Test (10 epochs)..."
        echo "   This will take approximately 2 hours"
        echo ""
        # Backup original config
        cp config/config.yaml config/config.yaml.bak
        # Update epochs to 10
        sed -i 's/num_epochs: 100/num_epochs: 10/' config/config.yaml
        python src/train.py --config config/config.yaml
        # Restore original config
        mv config/config.yaml.bak config/config.yaml
        ;;
    2)
        echo ""
        echo "🚀 Starting Standard Training (100 epochs)..."
        echo "   This will take approximately 15-20 hours"
        echo "   You can monitor progress with: tensorboard --logdir logs/tensorboard"
        echo ""
        python src/train.py --config config/config.yaml
        ;;
    3)
        echo ""
        echo "🚀 Starting Full Training (200 epochs)..."
        echo "   This will take approximately 30-40 hours"
        echo ""
        # Backup original config
        cp config/config.yaml config/config.yaml.bak
        # Update epochs to 200
        sed -i 's/num_epochs: 100/num_epochs: 200/' config/config.yaml
        python src/train.py --config config/config.yaml
        # Restore original config
        mv config/config.yaml.bak config/config.yaml
        ;;
    4)
        echo ""
        echo "📊 Analyzing dataset..."
        python scripts/prepare_data_example.py --data_root "data/raw/IMFDB FR dataset/IMFDB FR dataset"
        ;;
    5)
        echo ""
        echo "👋 Exiting. To train later, run:"
        echo "   source venv/bin/activate"
        echo "   python src/train.py --config config/config.yaml"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Model saved to: models/final/final_model.pth"
echo "📊 Logs saved to: logs/"
echo ""
echo "🔍 To run inference:"
echo "   python src/inference.py --model models/final/final_model.pth --image test.jpg"
echo ""

