# AI-TRANSLATED: OpenCL memory management utilities
# SYCL TODO: needs_review - implement actual memory tracking via SYCL USM


def memory_allocated(device=None) -> int:
    """Return the current GPU memory occupied by tensors in bytes."""
    # SYCL TODO: needs_review
    return 0


def max_memory_allocated(device=None) -> int:
    """Return the maximum GPU memory occupied by tensors in bytes."""
    # SYCL TODO: needs_review
    return 0


def memory_reserved(device=None) -> int:
    """Return the current GPU memory managed by the caching allocator in bytes."""
    # SYCL TODO: needs_review
    return 0


def empty_cache() -> None:
    """Release all unoccupied cached memory."""
    # SYCL TODO: needs_review - call OpenCLCachingAllocator::emptyCache()
    pass
