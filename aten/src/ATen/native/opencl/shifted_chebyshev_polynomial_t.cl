// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/shifted_chebyshev_polynomial_t.cu

#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/sycl/Math.h>
// SYCL: JIT compilation not used, using direct kernel dispatch

namespace at::native {
        namespace {
            constexpr char shifted_chebyshev_polynomial_t_name[] = "shifted_chebyshev_polynomial_t_forward";

            void shifted_chebyshev_polynomial_t_kernel_opencl(TensorIteratorBase& iterator) {
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_opencl", [&]() {
                    sycl_kernel_with_scalars(iterator, []SYCL_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return shifted_chebyshev_polynomial_t_forward<scalar_t, true>(x, n);
                    });
                });
            } // shifted_chebyshev_polynomial_t_kernel_opencl
        } // namespace (anonymous)

        REGISTER_DISPATCH(shifted_chebyshev_polynomial_t_stub, &shifted_chebyshev_polynomial_t_kernel_opencl)
} // namespace at::native
