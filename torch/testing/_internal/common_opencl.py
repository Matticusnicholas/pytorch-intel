# AI-TRANSLATED: OpenCL test utilities for Intel GPU (Arc/Xe) backend
# Requires hardware validation on Intel Arc GPU.

import torch
from torch.testing._internal.common_utils import TEST_WITH_OPENCL  # type: ignore[attr-defined]

# SYCL TODO: needs_review - these will work once the C++ bindings are complete

TEST_OPENCL = False
try:
    TEST_OPENCL = torch.opencl.is_available()
except Exception:
    TEST_OPENCL = False

OPENCL_DEVICE_COUNT = 0
try:
    if TEST_OPENCL:
        OPENCL_DEVICE_COUNT = torch.opencl.device_count()
except Exception:
    pass


def skip_if_no_opencl(fn):
    """Decorator to skip test if OpenCL is not available."""
    import unittest
    return unittest.skipIf(not TEST_OPENCL, "OpenCL not available")(fn)


def opencl_device():
    """Return a torch.device for the first OpenCL device."""
    return torch.device("opencl", 0)


def get_opencl_device_count():
    """Return the number of OpenCL devices."""
    return OPENCL_DEVICE_COUNT
