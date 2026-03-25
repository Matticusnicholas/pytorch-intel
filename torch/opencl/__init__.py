# AI-TRANSLATED: OpenCL/SYCL device management for Intel GPU (Arc/Xe) support
# Requires hardware validation on Intel Arc GPU.
# Modeled after torch/cuda/__init__.py

import torch
from torch._C import _opencl_isDriverSufficient, _opencl_getDriverVersion  # type: ignore[attr-defined]
from typing import Any, Optional, Union

__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "manual_seed",
    "manual_seed_all",
    "Device",
]

_initialized = False
_device_count = -1


class Device:
    """Context manager for OpenCL device selection."""
    def __init__(self, device: Union[int, str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device):
            if device.type != "opencl":
                raise ValueError(f"Expected opencl device, got {device.type}")
            device = device.index or 0
        self.prev_device = current_device()
        self.device = device

    def __enter__(self):
        set_device(self.device)
        return self

    def __exit__(self, *args):
        set_device(self.prev_device)


def is_available() -> bool:
    """Return True if OpenCL/SYCL (Intel GPU) is available."""
    # SYCL TODO: needs_review - check for actual SYCL runtime and Intel GPU
    try:
        return device_count() > 0
    except Exception:
        return False


def device_count() -> int:
    """Return the number of OpenCL (Intel GPU) devices available."""
    # SYCL TODO: needs_review - implement via C++ binding to SYCL device enumeration
    global _device_count
    if _device_count < 0:
        try:
            # Try to get from C++ backend
            _device_count = torch._C._opencl_getDeviceCount()  # type: ignore[attr-defined]
        except AttributeError:
            # C++ binding not available yet, return 0
            _device_count = 0
    return _device_count


def current_device() -> int:
    """Return the index of the currently selected OpenCL device."""
    # SYCL TODO: needs_review - implement via C++ binding
    try:
        return torch._C._opencl_getCurrentDevice()  # type: ignore[attr-defined]
    except AttributeError:
        return 0


def set_device(device: Union[int, torch.device]) -> None:
    """Set the current OpenCL device."""
    if isinstance(device, torch.device):
        device = device.index or 0
    # SYCL TODO: needs_review - implement via C++ binding
    try:
        torch._C._opencl_setDevice(device)  # type: ignore[attr-defined]
    except AttributeError:
        pass


def synchronize(device: Optional[Union[int, torch.device]] = None) -> None:
    """Wait for all kernels on the OpenCL device to complete."""
    # SYCL TODO: needs_review - implement via sycl::queue::wait()
    try:
        torch._C._opencl_synchronize(device or current_device())  # type: ignore[attr-defined]
    except AttributeError:
        pass


def manual_seed(seed: int) -> None:
    """Set the random seed for the current OpenCL device."""
    # SYCL TODO: needs_review - implement RNG for SYCL
    pass


def manual_seed_all(seed: int) -> None:
    """Set the random seed for all OpenCL devices."""
    # SYCL TODO: needs_review - implement RNG for SYCL
    pass


def _lazy_init():
    """Lazy initialization of OpenCL/SYCL backend."""
    global _initialized
    if _initialized:
        return
    _initialized = True
    # SYCL TODO: needs_review - perform actual SYCL initialization
