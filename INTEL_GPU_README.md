# PyTorch OpenCL/SYCL Backend for Intel GPUs

This fork adds an OpenCL/SYCL backend targeting Intel Arc and Xe GPUs via Intel's oneAPI toolchain.

## Status

**Experimental / AI-Translated** — All 217 CUDA kernels have been translated to SYCL and wired into PyTorch's dispatch system. Hardware validation on Intel Arc GPUs is still required.

### What's included

- 217 translated SYCL kernel files (`aten/src/ATen/native/opencl/`)
- 37 SYCL infrastructure headers (`aten/src/ATen/native/sycl/`)
- 20 dispatch registration files
- 849 operator dispatch entries in `native_functions.yaml`
- Full backend wiring: `DispatchStub`, `BackendComponent`, `DeviceType::OPENCL`, `AutogradOpenCL`
- `OpenCLHooksInterface`, `OpenCLCachingAllocator`, stream management
- `torch.opencl` Python module
- `c10/opencl/` and `c10/sycl/` runtime libraries

### Translation status

See `aten/src/ATen/native/opencl/translation_log.json` for per-file status:
- **172 success** — clean translations using PyTorch's `gpu_kernel` abstractions
- **11 partial** — translated with some sections needing manual work
- **34 needs_review** — complex kernels (warp shuffles, shared memory patterns, >500 lines)

## Quick Install (One-Click)

### Linux (Ubuntu 22.04/24.04)
```bash
chmod +x install_intel_gpu.sh
./install_intel_gpu.sh
```

### Windows
```
Right-click install_intel_gpu.bat → Run as Administrator
```

Both scripts handle everything: GPU drivers, Intel oneAPI toolkit, Python deps, and building PyTorch with `USE_OPENCL=1`. Takes 30-60 minutes depending on your system.

## Manual Install

### Prerequisites

### Hardware
- Intel Arc A-series GPU (A770, A750, A580, A380, etc.) or
- Intel Data Center GPU Flex/Max series or
- Intel integrated Xe graphics (different than xe dedicated) (i.e. the ones built into the chips) (limited performance)

### Software
- Linux (Ubuntu 22.04+ or RHEL 9+ recommended)
- Intel oneAPI Base Toolkit (2024.0+): https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- Intel GPU drivers: https://dgpu-docs.intel.com/driver/installation.html

## Installation

### 1. Install Intel GPU drivers

```bash
# Ubuntu 22.04/24.04
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt-get update
sudo apt-get install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2
```

Verify your GPU is detected:
```bash
sudo apt-get install -y clinfo
clinfo | grep "Device Name"
# Should show something like: Intel(R) Arc(TM) A770 Graphics
```

### 2. Install Intel oneAPI toolkit

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel
```

Source the oneAPI environment:
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Clone and build PyTorch with OpenCL support

```bash
git clone https://github.com/matticusnicholas/pytorch-intel.git
cd pytorch-intel
git checkout claude/cuda-to-opencl-translation-kl8Ms

# Install Python dependencies
pip install -r requirements.txt
pip install pyyaml typing_extensions

# Build with OpenCL enabled
export USE_OPENCL=1
export USE_CUDA=0
pip install -e . -v --no-build-isolation
```

Build will take 30-60 minutes depending on your system.

### 4. Verify installation

```python
import torch
import torch.opencl

print(f"OpenCL available: {torch.opencl.is_available()}")
print(f"Device count: {torch.opencl.device_count()}")

# Create a tensor on Intel GPU
x = torch.randn(3, 3, device="opencl")
y = torch.randn(3, 3, device="opencl")
z = x + y
print(z)
```

## Usage

```python
import torch

# Move tensors to Intel GPU
device = torch.device("opencl", 0)
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Operations run on Intel GPU
z = torch.matmul(x, y)
w = torch.nn.functional.relu(z)

# Move back to CPU
result = w.cpu()
```

### Device context manager

```python
with torch.opencl.Device(0):
    x = torch.randn(100, 100, device="opencl")
    # All operations use GPU 0
```

## Known Limitations

1. **BLAS operations** — `torch.matmul`, `torch.mm`, etc. require oneMKL integration (currently stubbed). Use Intel's oneMKL SYCL BLAS for production.

2. **FFT operations** — Spectral ops need oneMKL DFT integration.

3. **JIT compilation** — CUDA's jiterator path is not available. All kernels use ahead-of-time compilation via DPC++.

4. **Complex kernels** — 34 files marked `needs_review` have complex warp-level operations that may need tuning for Intel's subgroup sizes (8/16/32 vs CUDA's fixed 32).

5. **Pinned memory** — Host pinned memory allocation not yet implemented.

6. **Multi-GPU** — Stream and multi-device management are stubbed. Single-GPU only for now.

7. **cuDNN equivalents** — No oneDNN integration yet for optimized convolution/pooling/normalization.

## Architecture

```
c10/opencl/                          # Runtime: allocator, streams, guards, device functions
c10/sycl/                            # SYCL math compatibility layer
aten/src/ATen/native/opencl/         # 217 translated kernels (.cl) + 20 registration (.cpp)
aten/src/ATen/native/sycl/           # 37 SYCL utility headers (kernel launch infra)
aten/src/ATen/detail/                # OpenCLHooksInterface
aten/src/ATen/sycl/                  # SYCL generator for RNG
torch/opencl/                        # Python device management module
torch/testing/_internal/             # Test utilities (common_opencl.py)
```

## Contributing

The biggest areas that need work:

1. **oneMKL BLAS integration** — Replace stubs in `Blas.cpp`, `ScaledBlas.cpp`, `GroupedBlas.cpp` with actual `oneapi::mkl::blas` calls
2. **oneMKL FFT** — Wire up `SpectralOps.cpp` to `oneapi::mkl::dft`
3. **oneDNN** — Add optimized convolution/normalization via oneDNN
4. **Hardware validation** — Test all 217 kernels on actual Intel Arc hardware
5. **Subgroup tuning** — Validate warp-level operations for Intel's subgroup sizes
6. **Memory management** — Replace allocator stubs with `sycl::malloc_device`/`sycl::free`

## License

Same as PyTorch (BSD-3-Clause). SYCL translations were generated by Claude (AI) and require hardware validation.
