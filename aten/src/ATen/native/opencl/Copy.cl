// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Copy.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains cudaMemcpy, CUDAStream, CUDAEvent, CUDAGuard, peer-to-peer access,
// CUDACachingAllocator, Float8 intrinsics, and complex copy dispatch logic.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/SYCLEvent.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Loops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <c10/sycl/SYCLStream.h>

namespace at::native {

void neg_kernel_opencl(TensorIteratorBase &iter);
void conj_kernel_opencl(TensorIteratorBase &iter);

void float16_copy_kernel_opencl(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] SYCL_LAMBDA(float value) {
        return static_cast<at::Half>(value);
    });
}

void bfloat16_copy_kernel_opencl(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] SYCL_LAMBDA(float value) {
        return static_cast<at::BFloat16>(value);
    });
}

void direct_copy_kernel_opencl(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t x) { return x; });
    });
  } else if (iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
     if (dtype == kBFloat16) {
       bfloat16_copy_kernel_opencl(iter);
     } else {
       float16_copy_kernel_opencl(iter);
     }
  } else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] SYCL_LAMBDA(scalar_t x) { return x; });
    });
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t x) { return x; });
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_opencl(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_opencl", [&] {
    sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

// SYCL TODO: needs_review - device-to-device copy, host-device copy with
// SYCL queue memcpy, event synchronization, and peer-to-peer access
// need comprehensive SYCL porting.

static void copy_kernel_opencl(TensorIterator& iter, bool non_blocking) {
  TORCH_CHECK(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // For same-device copies, use kernel dispatch
  if (dst_device == src_device) {
    bool same_type = iter.dtype(0) == iter.dtype(1);
    bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
    bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();

    if (same_type && same_conj && same_neg && iter.is_contiguous()) {
      // Use memcpy for contiguous same-type tensors
      void *dst = iter.data_ptr(0);
      void *src = iter.data_ptr(1);
      size_t size = iter.numel() * iter.element_size(0);
      if (src != dst && size > 0) {
        auto& queue = at::sycl::getCurrentSYCLQueue();
        queue.memcpy(dst, src, size);
      }
    } else {
      if (same_neg) {
        if (!same_conj) {
          conj_kernel_opencl(iter);
        } else {
          direct_copy_kernel_opencl(iter);
        }
      } else {
        if (!same_conj) {
          neg_conj_kernel_opencl(iter);
        } else {
          neg_kernel_opencl(iter);
        }
      }
    }

    if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
       iter.tensor(0).conj_physical_();
    }
    if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
       iter.tensor(0).neg_();
    }
    return;
  }

  // Cross-device copies need SYCL event synchronization
  // SYCL TODO: needs_review - implement cross-device copy with SYCL queues
  TORCH_CHECK(false, "copy_kernel_opencl: cross-device copy pending full SYCL port");
}

REGISTER_DISPATCH(copy_stub, &copy_kernel_opencl)

} // namespace at::native
