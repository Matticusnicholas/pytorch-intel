// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/MemoryAccess.cuh
#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <c10/core/DynamicCast.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/detail/FunctionTraits.h>

namespace at::native::memory {

namespace detail {

// static_unroll: compile-time loop unrolling via template metaprogramming
// (same as CUDA version, no device-specific changes needed)
template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... /*args*/) {}
};

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static void apply(policy_t &self, args_t *args, int idx, int block_work_size) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + block_work_size * idx;
    auto args_accessor = [&args](int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
    self.load_single_arg(args_accessor, ptr);
  }
};

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<typename data_t, typename offsets_t, typename ...Args>
  C10_HOST_DEVICE static void apply(
      const data_t& data,
      const offsets_t& offsets,
      std::tuple<Args...> ret) {
    using T = typename std::tuple_element<current, std::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = std::get<current>(ret);
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
  scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return c10::load(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

template <int N>
struct LoadWithCast {
  using array_t = std::array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  LoadWithCast(const TensorIteratorBase& iter) {
    TORCH_INTERNAL_ASSERT(iter.ninputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i + iter.noutputs());
      element_sizes[i] = c10::elementSize(iter.dtype(i + iter.noutputs()));
    }
  }

  template<typename scalar_t>
  scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    return c10::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
  void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

template <int N = 1>
struct StoreWithCast {
  using array_t = std::array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  StoreWithCast(const TensorIteratorBase& iter) {
    TORCH_INTERNAL_ASSERT(iter.noutputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i);
      element_sizes[i] = c10::elementSize(iter.dtype(i));
    }
  }

  template<typename scalar_t>
  void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    c10::cast_and_store<scalar_t>(dtypes[arg], ptr, value);
  }
};

// SYCL: aligned vector for vectorized load/store
// SYCL does not use __align__ but uses alignas
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <int vec_size, typename scalar_t>
aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}

template <int vec_size>
aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
  auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
  aligned_vector<bool, vec_size> ret;
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = bool(tmp.val[i]);
  }
  return ret;
}

namespace policies {

// SYCL: work-item local_id replaces threadIdx.x
// num_threads is the work-group size
template <
    int num_threads,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int elems_per_thread,
    int num_outputs = 1>
struct unroll_base {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  static constexpr int tws = elems_per_thread;
  static constexpr int block_work_size = elems_per_thread * num_threads;

  // SYCL: work-item local ID stored for access
  int local_id;

  unroll_base(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s,
      int local_id_ = 0)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        loader(l),
        storer(s),
        local_id(local_id_) {}

  inline bool check_inbounds(int thread_work_elem) {
    return ((int)(local_id + thread_work_elem * num_threads) < remaining);
  }

  template<typename args_t>
  inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = local_id;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      if (thread_idx < remaining) {
        int linear_idx = thread_idx + block_work_size * idx;
        auto offset = input_offset_calculator.get(linear_idx);
        detail::static_unroll<detail::unroll_load_helper, arity>::with_args(
            *this, args, offset, loader, i, num_outputs);
        thread_idx += num_threads;
      }
    }
  }

  template<typename scalar_t>
  inline void store(scalar_t *from, int idx) {
    int thread_idx = local_id;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      if (thread_idx < remaining) {
        int linear_idx = thread_idx + block_work_size * idx;
        int offset = output_offset_calculator.get(linear_idx)[0];
        storer.store(from[i], data[0], offset);
        thread_idx += num_threads;
      }
    }
  }
};

template <int vec_size, typename data_t, int elems_per_thread>
struct vectorized {
  static_assert(elems_per_thread % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = elems_per_thread / vec_size;
  static constexpr int tws = elems_per_thread;

  data_t data;
  int local_id;
  int num_threads_val;

  vectorized(data_t data, int local_id_ = 0, int num_threads_ = 1)
    : data(data), local_id(local_id_), num_threads_val(num_threads_) {}

  inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template<typename accessor_t, typename scalar_t>
  inline void load_single_arg(accessor_t to, scalar_t *from) {
    int thread_idx = local_id;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads_val;
      auto v = load_vector<vec_size>(from, index);
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

  template<typename args_t>
  inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, idx, elems_per_thread * num_threads_val);
  }

  template<typename scalar_t>
  inline void store(scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + elems_per_thread * num_threads_val * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = local_id;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads_val;
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;
  StoreWithoutCast storer;
  static constexpr int tws = 4; // thread_work_size default

  int local_id;
  int num_threads_val;

  multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc,
                       int local_id_ = 0, int num_threads_ = 1)
    : data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc),
      local_id(local_id_), num_threads_val(num_threads_) {}

  inline bool check_inbounds(int thread_work_elem) {
    return ((int)(local_id + thread_work_elem * num_threads_val) < remaining);
  }

  template<typename args_t>
  inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = local_id;
    #pragma unroll
    for (int i = 0; i < tws; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + (tws * num_threads_val) * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads_val;
    }
  }

  template <typename return_t>
  inline void store(return_t *from, int idx) {
    int thread_idx = local_id;
    #pragma unroll
    for (int i = 0; i < tws; i++) {
      if (thread_idx >= this->remaining) {
        return;
      }
      int linear_idx = thread_idx + (tws * num_threads_val) * idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      memory::detail::static_unroll<detail::multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      thread_idx += num_threads_val;
    }
  }
};

}  // namespace policies

template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(const char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of_v<aligned_vector<scalar_t, 2>>;
  constexpr int vec4_alignment = std::alignment_of_v<aligned_vector<scalar_t, 4>>;
  constexpr int vec8_alignment = std::alignment_of_v<aligned_vector<scalar_t, 8>>;
  if (address % vec8_alignment == 0) {
    return 8;
  } else if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(char *pointer) {
  return can_vectorize_up_to<scalar_t>(static_cast<const char*>(pointer));
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static C10_HOST_DEVICE void apply(int &result, array_t pointers, traits) {
    using arg_t = typename traits::template arg<i>::type;
    result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
  return result;
}

template <typename T>
inline size_t get_alignment(T ptr_or_size) {
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0) {
    return 16;
  } else if (val % 8 == 0) {
    return 8;
  } else if (val % 4 == 0) {
    return 4;
  } else if (val % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

template <>
inline size_t get_alignment<size_t>(size_t size) {
  return get_alignment(reinterpret_cast<void*>(size));
}

template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

// SYCL: vectorized load/store without inline assembly
// Uses standard memory operations which the SYCL compiler can optimize
template <int Size>
union Vec;

template <>
union Vec<4> {
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8> {
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16> {
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  float f32[4];
};

// SYCL: standard memory loads (no PTX asm needed)
template <int Alignment, typename T>
inline Vec<Alignment> ld_vec(const T* addr) {
  Vec<Alignment> vec;
  if constexpr (Alignment == 16) {
    auto* src = reinterpret_cast<const uint32_t*>(addr);
    vec.u32[0] = src[0];
    vec.u32[1] = src[1];
    vec.u32[2] = src[2];
    vec.u32[3] = src[3];
  } else if constexpr (Alignment == 8) {
    vec.u64 = *reinterpret_cast<const uint64_t*>(addr);
  } else if constexpr (Alignment == 4) {
    vec.u32 = *reinterpret_cast<const uint32_t*>(addr);
  } else {
    static_assert(dependent_false<T>);
  }
  return vec;
}

template <int Alignment, typename T>
inline void st_vec(T* addr, const Vec<Alignment>& vec) {
  if constexpr (Alignment == 16) {
    auto* dst = reinterpret_cast<uint32_t*>(addr);
    dst[0] = vec.u32[0];
    dst[1] = vec.u32[1];
    dst[2] = vec.u32[2];
    dst[3] = vec.u32[3];
  } else if constexpr (Alignment == 8) {
    *reinterpret_cast<uint64_t*>(addr) = vec.u64;
  } else if constexpr (Alignment == 4) {
    *reinterpret_cast<uint32_t*>(addr) = vec.u32;
  } else {
    static_assert(dependent_false<T>);
  }
}

} // namespace at::native::memory
