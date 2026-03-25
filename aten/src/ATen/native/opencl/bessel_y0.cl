// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/bessel_y0.cu

#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/Math.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <c10/util/complex.h>

namespace at::native {
namespace {
constexpr char bessel_y0_name[] = "bessel_y0_forward";

void bessel_y0_kernel_opencl(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y0_opencl", [&]() {
        sycl_kernel(iterator, []SYCL_LAMBDA(scalar_t a) -> scalar_t {
            return bessel_y0_forward(a);
        });
    });
}

} // anonymous namespace

REGISTER_DISPATCH(special_bessel_y0_stub, &bessel_y0_kernel_opencl)
} // namespace at::native
