// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/laguerre_polynomial_l.cu

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
            constexpr char laguerre_polynomial_l_name[] = "laguerre_polynomial_l_forward";

            void laguerre_polynomial_l_kernel_opencl(TensorIteratorBase& iterator) {
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_opencl", [&]() {
                    sycl_kernel_with_scalars(iterator, []SYCL_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return laguerre_polynomial_l_forward<scalar_t, true>(x, n);
                    });
                });
            } // laguerre_polynomial_l_kernel_opencl
        } // namespace (anonymous)

        REGISTER_DISPATCH(laguerre_polynomial_l_stub, &laguerre_polynomial_l_kernel_opencl)
} // namespace at::native
