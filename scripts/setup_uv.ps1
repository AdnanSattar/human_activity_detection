# Setup script for UV package manager (Windows PowerShell)

Write-Host "Setting up Human Activity Detection project with UV..." -ForegroundColor Green

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
uv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
uv pip install -r requirements.txt

# Install PyTorch with CUDA (if needed)
Write-Host ""
Write-Host "Note: PyTorch with CUDA should be installed separately:" -ForegroundColor Cyan
Write-Host "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Cyan
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
python scripts/verify_setup.py

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To activate the virtual environment:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
