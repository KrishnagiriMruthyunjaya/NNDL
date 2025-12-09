#!/bin/bash

# Setup script for Emotion Detection System
# NNDL Course Project

echo "========================================="
echo "Emotion Detection System Setup"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✓ Found: $python_version"
else
    echo "✗ Python not found! Please install Python 3.7+"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
if [[ -d "emotion_env" ]]; then
    echo "✓ Virtual environment already exists"
else
    python -m venv emotion_env
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source emotion_env/Scripts/activate
else
    # macOS/Linux
    source emotion_env/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "========================================="
echo "✓ Setup completed successfully!"
echo "========================================="
echo ""
echo "To run the emotion detection system:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source emotion_env/Scripts/activate"
else
    echo "   source emotion_env/bin/activate"
fi
echo "2. Run the application:"
echo "   python EmotionDetection.py"
echo ""
echo "Press 'q' to quit the application when running."