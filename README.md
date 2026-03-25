# PyTorch for Intel GPUs

Run PyTorch on Intel Arc and Xe GPUs using OpenCL/SYCL.

## Quick Install

### Linux
```bash
git clone https://github.com/matticusnicholas/pytorch-intel.git
cd pytorch-intel
chmod +x install_intel_gpu.sh
./install_intel_gpu.sh
```

### Windows
```
git clone https://github.com/matticusnicholas/pytorch-intel.git
cd pytorch-intel
install_intel_gpu.bat
```
Right-click the `.bat` file and **Run as Administrator**.

The installer handles Intel GPU drivers, oneAPI toolkit, Python dependencies, and building PyTorch. Takes 30-60 minutes.

## Manual Install

```bash
# 1. Install Intel GPU drivers
#    https://dgpu-docs.intel.com/driver/installation.html

# 2. Install Intel oneAPI (DPC++ compiler + oneMKL)
#    https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

# 3. Source oneAPI
source /opt/intel/oneapi/setvars.sh

# 4. Build
git clone https://github.com/matticusnicholas/pytorch-intel.git
cd pytorch-intel
pip install -r requirements.txt
USE_OPENCL=1 USE_CUDA=0 pip install -e . -v --no-build-isolation
```

## Usage

```python
import torch

x = torch.randn(3, 3, device="opencl")
y = torch.randn(3, 3, device="opencl")
print(x + y)
```

```python
import torch.opencl

print(torch.opencl.is_available())   # True if Intel GPU detected
print(torch.opencl.device_count())   # Number of Intel GPUs
```

## Supported Hardware

- Intel Arc A-series (A770, A750, A580, A380)
- Intel Data Center GPU Flex / Max
- Intel integrated Xe graphics (limited performance)

## What This Is

A fork of PyTorch with 217 CUDA kernels translated to SYCL for Intel GPU support. Includes full dispatch wiring, a caching allocator, stream management, and a `torch.opencl` Python module.

See [INTEL_GPU_README.md](INTEL_GPU_README.md) for technical details, architecture, known limitations, and contributing guide.

## License

BSD-3-Clause (same as PyTorch).
