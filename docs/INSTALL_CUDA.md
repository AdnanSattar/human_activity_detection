# CUDA Installation Guide

## Current Status

Your system shows:

- ✅ NVIDIA GPU: GeForce MX450 (2GB VRAM)
- ✅ NVIDIA Driver: 581.83
- ✅ CUDA Driver Support: 13.0
- ❌ PyTorch CUDA: Not installed (currently CPU-only version)

## Installation Steps

### Step 1: Verify CUDA Toolkit

Check if CUDA toolkit is installed:

```bash
nvcc --version
```

If not installed, download from: <https://developer.nvidia.com/cuda-downloads>

### Step 2: Install PyTorch with CUDA Support

**For CUDA 12.4 (Recommended - backward compatible with CUDA 13.0 driver):**

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 11.8 (Alternative):**

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation

```bash
python scripts/verify_setup.py
```

Expected output:

```text
PyTorch Version: 2.x.x+cu124
CUDA Available: True
CUDA Version: 12.4
GPU Count: 1
GPU Name: NVIDIA GeForce MX450
GPU Memory: 2.00 GB
```

### Step 4: Test GPU Training

```bash
python -c "import torch; x = torch.randn(3, 3).cuda(); print('GPU test successful:', x.device)"
```

## Troubleshooting

### Issue: CUDA not available after installation

1. **Check CUDA toolkit version:**

   ```bash
   nvcc --version
   ```

2. **Check PyTorch CUDA version:**

   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall matching versions:**
   - If CUDA toolkit is 12.4, use cu124 PyTorch
   - If CUDA toolkit is 11.8, use cu118 PyTorch

### Issue: Out of Memory (OOM)

Your MX450 has 2GB VRAM. Use these settings:

```yaml
# config/training_config.yaml
training:
  batch_size: 4  # Start with 4, increase if stable
  imgsz: 640
  cache: false  # Disable if OOM occurs
```

### Issue: Driver version mismatch

If you get driver version errors:

- Update NVIDIA drivers: <https://www.nvidia.com/drivers>
- Or use older CUDA version (11.8)

## Performance Tips for MX450 (2GB VRAM)

1. **Use smallest model**: `yolo11n.pt` (nano)
2. **Batch size**: 4-8 (test and adjust)
3. **Image size**: 640 (default) or 512 if OOM
4. **AMP**: if you see instability/NaNs, keep `amp: false` (this repo defaults to `false` for stability)
5. **Cache**: set `cache: false` if RAM is tight or you hit paging-file errors
6. **Workers (Windows)**: keep `workers: 0` or `workers: 1` to avoid paging-file / multiprocessing issues

## Alternative: CPU Training

If CUDA installation fails, you can still train on CPU (much slower):

```yaml
# config/training_config.yaml
training:
  device: "cpu"
  batch_size: 2  # Smaller for CPU
```

Training will be 10-50x slower on CPU, but will work.

## Next Steps

After installing CUDA PyTorch:

1. ✅ Verify: `python scripts/verify_setup.py`
2. ✅ Test: `python scripts/quick_start.py`
3. ✅ Train: `python -m src.train`

---

**Note**: The CUDA driver version (13.0) can support CUDA toolkit 12.4. They are backward compatible.
