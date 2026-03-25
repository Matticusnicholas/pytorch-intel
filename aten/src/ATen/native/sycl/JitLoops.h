// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/CUDAJitLoops.cuh and ATen/native/cuda/JitLoops.cuh
//
// SYCL: JIT compilation is not used in the SYCL backend.
// SYCL kernels are compiled ahead-of-time (AOT) using the DPC++ compiler.
// This header provides stub declarations for compatibility.
#pragma once

#include <ATen/native/sycl/Loops.h>

namespace at::native {

// SYCL uses AOT compilation, so the Jiterator infrastructure is not needed.
// Kernels that relied on JIT in the CUDA backend are compiled at build time
// for the SYCL backend.
//
// If you need to add a new elementwise kernel, define a functor and use
// sycl_kernel() from Loops.h instead of the jiterator path.

} // namespace at::native
