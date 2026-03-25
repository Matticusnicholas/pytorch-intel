// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/legendre_polynomial_p.cu

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
            constexpr char legendre_polynomial_p_name[] = "legendre_polynomial_p_forward";

            void legendre_polynomial_p_kernel_opencl(TensorIteratorBase& iterator) {
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_opencl", [&]() {
                    sycl_kernel_with_scalars(iterator, []SYCL_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return legendre_polynomial_p_forward<scalar_t, true>(x, n);
                    });
                });
            } // legendre_polynomial_p_kernel_opencl
        } // namespace (anonymous)

        REGISTER_DISPATCH(legendre_polynomial_p_stub, &legendre_polynomial_p_kernel_opencl)
} // namespace at::native
