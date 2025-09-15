#!/bin/bash

# Setup script for LLaMA-Factory training environment
# This script creates a virtual environment and installs all required dependencies

echo "Setting up LLaMA-Factory training environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv llama-factory-env

# Activate virtual environment
echo "Activating virtual environment..."
source llama-factory-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: source llama-factory-env/bin/activate"
echo "To run training, execute: bash train.sh"