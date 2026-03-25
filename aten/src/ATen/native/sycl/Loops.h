// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/Loops.cuh and ATen/native/cuda/CUDALoops.cuh
#pragma once

#include <ATen/OpMathType.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/native/sycl/MemoryAccess.h>

#include <c10/util/C++17.h>
#include <tuple>

// SYCL equivalents of CUDA kernel launch macros
#define SYCL_LAMBDA [=]
#define SYCL_KERNEL_ASSERT(cond) assert(cond)

namespace at::native {

template<int N>
static OffsetCalculator<N> make_input_offset_calculator(const TensorIteratorBase& iter) {
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(const TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

// SYCL kernel helper: replaces CUDA's elementwise_kernel_helper
// In SYCL, __device__ is not needed; work-item index comes from sycl::nd_item
template <bool reverted_idx = false, typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy, sycl::nd_item<1> item) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;
  constexpr int elems_per_thread = policy_t::tws;

  int idx = item.get_group(0);
  if constexpr (reverted_idx)
    idx = item.get_group_range(0) - item.get_group(0) - 1;

  return_t results[elems_per_thread];
  args_t args[elems_per_thread];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = std::apply(f, args[i]);
    }
  }

  // store
  policy.store(results, idx);
}

// SYCL kernel implementation (replaces gpu_kernel_impl / gpu_kernel_impl_nocast)
// The actual SYCL kernel submission happens here using sycl::queue::parallel_for
template <typename func_t>
void sycl_kernel_impl(TensorIteratorBase& iter, const func_t& f);

template <typename func_t>
void sycl_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f);

// Primary entry point: sycl_kernel (replaces gpu_kernel)
template <typename func_t>
void sycl_kernel(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_xpu(),
      "argument ", arg, ": expected an XPU device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      sycl_kernel(sub_iter, f);
    }
    return;
  }

  sycl_kernel_impl(iter, f);
}

template <typename func_t>
void sycl_kernel_nocast(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_xpu(),
      "argument ", arg, ": expected an XPU device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      sycl_kernel_nocast(sub_iter, f);
    }
    return;
  }

  sycl_kernel_impl_nocast(iter, f);
}

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  AUnaryFunctor(func_t f_, opmath_arg1_t a_): f(f_), a(a_) {}
  private:
    func_t f;
    opmath_arg1_t a;
};

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  BUnaryFunctor(func_t f_, opmath_arg2_t b_): f(f_), b(b_) {}
  private:
    func_t f;
    opmath_arg2_t b;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_): f(f_) {}
  private:
    func_t f;
};

template <typename arg1_t, typename arg2_t = arg1_t, typename return_t = arg1_t, typename func_t>
void opmath_sycl_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(
      traits::arity == 2,
      "sycl_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    const OptionalDeviceGuard device_guard(iter.device(1));
    sycl_kernel(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    sycl_kernel(iter, bf);
  } else {
    sycl_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

// Alias for backward compatibility
template <typename arg1_t, typename arg2_t = arg1_t, typename return_t = arg1_t, typename func_t>
void opmath_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  opmath_sycl_kernel_with_scalars<arg1_t, arg2_t, return_t>(iter, f);
}

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_sycl_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "sycl_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same_v<opmath_arg_t, typename traits::template arg<1>::type>,
                "f is not symmetric");

  OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);
    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    sycl_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    sycl_kernel(iter, unary_f);
  }
}

// Alias for backward compatibility
template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  opmath_symmetric_sycl_kernel_with_scalars<scalar_t, return_t>(iter, f);
}

template <typename func_t>
void sycl_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "sycl_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_sycl_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t>(iter, f);
}

// Alias for backward compatibility
template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  sycl_kernel_with_scalars(iter, f);
}

// Alias: gpu_kernel -> sycl_kernel for backward compatibility
template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {
  sycl_kernel(iter, f);
}

namespace { // functions for `sycl_kernel_multiple_outputs`.

template <typename T> struct is_tuple: std::false_type {};
template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {};

} // namespace

template <typename func_t>
void sycl_kernel_multiple_outputs_impl(TensorIteratorBase& iter, const func_t& f);

template <typename func_t>
void sycl_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_xpu());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      sycl_kernel_multiple_outputs(sub_iter, f);
    }
    return;
  }

  sycl_kernel_multiple_outputs_impl(iter, f);
}

// Alias for backward compatibility
template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
  sycl_kernel_multiple_outputs(iter, f);
}

} // namespace at::native
