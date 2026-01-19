#!/bin/bash
# Setup script for UV package manager

echo "Setting up Human Activity Detection project with UV..."

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Install PyTorch with CUDA (if needed)
echo ""
echo "Note: PyTorch with CUDA should be installed separately:"
echo "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
echo ""

# Verify installation
echo "Verifying installation..."
python scripts/verify_setup.py

echo ""
echo "Setup complete!"
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate  # Linux/Mac"
echo "  .venv\\Scripts\\activate     # Windows"
