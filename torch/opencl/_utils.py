# AI-TRANSLATED: OpenCL device utilities
from typing import Any


def _get_device_index(device: Any, optional: bool = False) -> int:
    """Get the device index from a device specification."""
    import torch
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "opencl":
            raise ValueError(f"Expected opencl device, got {device.type}")
        return device.index or 0
    if optional:
        return 0
    raise ValueError(f"Cannot determine device index from {device}")
