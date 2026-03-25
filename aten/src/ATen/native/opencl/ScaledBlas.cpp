// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ScaledBlas.cpp
//
// SYCL TODO: needs_review - entire file requires oneMKL BLAS integration
// The CUDA version wraps cuBLAS scaled GEMM operations (_scaled_mm).
// For the SYCL/OpenCL backend, these must be replaced with oneMKL equivalents.
//
// Key changes needed:
// - Replace cuBLAS scaled MM with oneMKL scaled GEMM
// - Replace device capability checks (sm89/sm90/sm100) with Intel GPU arch checks
// - Replace MSLK/cublasLt paths with oneMKL extensions
// - fp8 support depends on Intel GPU hardware capabilities
