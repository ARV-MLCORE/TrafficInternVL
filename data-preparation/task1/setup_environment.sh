#!/bin/bash

# Setup script for Data Preparation environment
# This script creates a virtual environment and installs all required dependencies

echo "Setting up Data Preparation environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv data-prep-env

# Activate virtual environment
echo "Activating virtual environment..."
source data-prep-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Install additional common packages for data processing
echo "Installing additional data processing packages..."
pip install opencv-python pillow numpy pandas matplotlib

echo "Environment setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. To activate the environment, run: source data-prep-env/bin/activate"
echo "2. To prepare training data, run: bash prepare_data_train.sh"
echo "3. To prepare test data, run: bash prepare_data_test.sh"
echo ""
echo "📁 Make sure to update the data paths in the preparation scripts:"
echo "   - prepare_data_train.sh"
echo "   - prepare_data_test.sh"