// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/WeightNorm.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/DeviceUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#endif

namespace at::native {
namespace {

#define BLOCK 256
#define TILE_W 16
#define TILE_H 64

template <typename T>
struct ReduceAdd {
  inline T operator()(const T a, const T b) const {
    return (a + b);
  }
};

template<typename T, typename ReduceOp>
inline void reduce_block_into_lanes(
  sycl::nd_item<1> item,
  T *x,
  T val,
  int lanes,
  ReduceOp reduceOp)
{
  int tid = item.get_local_id(0);
  int blockSize = item.get_local_range(0);

  if(blockSize >= 64) {
    x[tid] = val;
    item.barrier(sycl::access::fence_space::local_space);
  }

  for(int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if(tid < i)
      x[tid] = reduceOp(x[tid], x[tid+i]);
    item.barrier(sycl::access::fence_space::local_space);
  }

  if(tid < 32) {
    T final_val;
    if(blockSize >= 64)
      final_val = reduceOp(x[tid], x[tid+32]);
    else
      final_val = val;

    auto sg = item.get_sub_group();
    for(int i = 16; i >= lanes; i >>= 1)
      final_val = reduceOp(final_val, sycl::shift_group_left(sg, final_val, i));

    if(tid < lanes)
      x[tid] = final_val;
  }

  item.barrier(sycl::access::fence_space::local_space);
}

template<typename T, typename ReduceOp>
inline void reduce_block_into_lanes_2d(
  sycl::nd_item<2> item,
  T *x,
  T val,
  int lanes,
  ReduceOp reduceOp)
{
  int tid = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);
  int blockSize = item.get_local_range(0) * item.get_local_range(1);

  if(blockSize >= 64) {
    x[tid] = val;
    item.barrier(sycl::access::fence_space::local_space);
  }

  for(int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if(tid < i)
      x[tid] = reduceOp(x[tid], x[tid+i]);
    item.barrier(sycl::access::fence_space::local_space);
  }

  if(tid < 32) {
    T final_val;
    if(blockSize >= 64)
      final_val = reduceOp(x[tid], x[tid+32]);
    else
      final_val = val;

    auto sg = item.get_sub_group();
    for(int i = 16; i >= lanes; i >>= 1)
      final_val = reduceOp(final_val, sycl::shift_group_left(sg, final_val, i));

    if(tid < lanes)
      x[tid] = final_val;
  }

  item.barrier(sycl::access::fence_space::local_space);
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_fwd_first_dim_kernel_impl(
    sycl::nd_item<1> item,
    scalar_t* w,
    accscalar_t* norms,
    const scalar_t* v,
    const scalar_t* g,
    const int rowSize,
    accscalar_t* s) {
  const int tid = item.get_local_id(0);
  const int row = item.get_group(0);
  const int stride = item.get_local_range(0);
  const int rowStart = row * rowSize;

  accscalar_t thread_sum = 0.f;
  for(int i = tid; i < rowSize; i += stride) {
    accscalar_t val_f = static_cast<accscalar_t>(v[i+rowStart]);
    thread_sum += val_f * val_f;
  }

  reduce_block_into_lanes(item, s, thread_sum, 1, ReduceAdd<accscalar_t>());
  accscalar_t result = s[0];
  result = sycl::sqrt(result);

  if(tid == 0)
    norms[row] = result;

  accscalar_t g_this_row = static_cast<accscalar_t>(g[row]);
  accscalar_t rnorm = 1.f / result;

  for(int i = tid; i < rowSize; i += stride) {
    accscalar_t val_f = static_cast<accscalar_t>(v[i+rowStart]);
    w[i+rowStart] = static_cast<scalar_t>(g_this_row * val_f * rnorm);
  }
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_fwd_last_dim_kernel_impl(
    sycl::nd_item<2> item,
    scalar_t* w,
    accscalar_t* norms,
    const scalar_t* v,
    const scalar_t* g,
    const int fast_dim_size,
    const int slower_dims_size,
    accscalar_t* alloc) {
  const int fast_dim_location = item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);

  accscalar_t* s = &alloc[0];
  accscalar_t* rnorms_this_block = &alloc[item.get_local_range(1) * item.get_local_range(0)];

  accscalar_t thread_sum = 0.f;
  int slower_dims_location = item.get_local_id(0);
  int currentIdx = fast_dim_location + fast_dim_size * slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size) {
      accscalar_t val_f = static_cast<accscalar_t>(v[currentIdx]);
      thread_sum += val_f * val_f;
      currentIdx += item.get_local_range(0) * fast_dim_size;
      slower_dims_location += item.get_local_range(0);
    }

  reduce_block_into_lanes_2d(item, s, thread_sum, item.get_local_range(1), ReduceAdd<accscalar_t>());

  if(item.get_local_id(0) == 0) {
    accscalar_t result = s[item.get_local_id(1)];
    accscalar_t norm_this_col = sycl::sqrt(result);
    norms[fast_dim_location] = norm_this_col;
    rnorms_this_block[item.get_local_id(1)] = 1.f / norm_this_col;
  }

  item.barrier(sycl::access::fence_space::local_space);

  accscalar_t g_this_col = static_cast<accscalar_t>(g[fast_dim_location]);
  accscalar_t rnorm = rnorms_this_block[item.get_local_id(1)];

  slower_dims_location = item.get_local_id(0);
  currentIdx = fast_dim_location + fast_dim_size * slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size) {
      accscalar_t val_f = static_cast<accscalar_t>(v[currentIdx]);
      w[currentIdx] = static_cast<scalar_t>(g_this_col * val_f * rnorm);
      currentIdx += item.get_local_range(0) * fast_dim_size;
      slower_dims_location += item.get_local_range(0);
    }
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_bwd_first_dim_kernel_impl(
    sycl::nd_item<1> item,
    scalar_t* grad_v,
    scalar_t* grad_g,
    const scalar_t* grad_w,
    const scalar_t* saved_v,
    const scalar_t* saved_g,
    const accscalar_t* saved_norms,
    const int rowSize,
    accscalar_t* s) {
  const int tid = item.get_local_id(0);
  const int row = item.get_group(0);
  const int stride = item.get_local_range(0);
  const int rowStart = row * rowSize;

  accscalar_t thread_sum = 0.f;
  for(int i = tid; i < rowSize; i += stride) {
    accscalar_t grad_wi = static_cast<accscalar_t>(grad_w[i+rowStart]);
    accscalar_t saved_vi = static_cast<accscalar_t>(saved_v[i+rowStart]);
    thread_sum += grad_wi * saved_vi;
  }

  reduce_block_into_lanes(item, s, thread_sum, 1, ReduceAdd<accscalar_t>());
  accscalar_t result = s[0];

  accscalar_t rnorm = 1.f / saved_norms[row];
  accscalar_t rnorm3 = rnorm * rnorm * rnorm;

  if(tid == 0)
    grad_g[row] = static_cast<scalar_t>(result * rnorm);

  accscalar_t g_this_row = static_cast<accscalar_t>(saved_g[row]);

  for(int j = tid; j < rowSize; j += stride) {
    accscalar_t grad_wj = static_cast<accscalar_t>(grad_w[j+rowStart]);
    accscalar_t saved_vj = static_cast<accscalar_t>(saved_v[j+rowStart]);
    accscalar_t grad_vj = g_this_row * (rnorm * grad_wj - rnorm3 * saved_vj * result);
    grad_v[j+rowStart] = static_cast<scalar_t>(grad_vj);
  }
}

template <typename scalar_t, typename accscalar_t>
void weight_norm_bwd_last_dim_kernel_impl(
    sycl::nd_item<2> item,
    scalar_t* grad_v,
    scalar_t* grad_g,
    const scalar_t* grad_w,
    const scalar_t* saved_v,
    const scalar_t* saved_g,
    const accscalar_t* saved_norms,
    const int fast_dim_size,
    const int slower_dims_size,
    accscalar_t* s) {
  const int fast_dim_location = item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);

  accscalar_t thread_sum = 0.f;
  int slower_dims_location = item.get_local_id(0);
  int currentIdx = fast_dim_location + fast_dim_size * slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size) {
      accscalar_t grad_wi = static_cast<accscalar_t>(grad_w[currentIdx]);
      accscalar_t saved_vi = static_cast<accscalar_t>(saved_v[currentIdx]);
      thread_sum += grad_wi * saved_vi;
      currentIdx += item.get_local_range(0) * fast_dim_size;
      slower_dims_location += item.get_local_range(0);
    }

  reduce_block_into_lanes_2d(item, s, thread_sum, item.get_local_range(1), ReduceAdd<accscalar_t>());
  accscalar_t result = s[item.get_local_id(1)];

  accscalar_t rnorm = 1.f / saved_norms[fast_dim_location];
  accscalar_t rnorm3 = rnorm * rnorm * rnorm;

  if(item.get_local_id(0) == 0)
    grad_g[fast_dim_location] = static_cast<scalar_t>(result * rnorm);

  accscalar_t g_this_col = static_cast<accscalar_t>(saved_g[fast_dim_location]);

  slower_dims_location = item.get_local_id(0);
  currentIdx = fast_dim_location + fast_dim_size * slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size) {
      accscalar_t grad_wj = static_cast<accscalar_t>(grad_w[currentIdx]);
      accscalar_t saved_vj = static_cast<accscalar_t>(saved_v[currentIdx]);
      accscalar_t grad_vj = g_this_col * (rnorm * grad_wj - rnorm3 * saved_vj * result);
      grad_v[currentIdx] = static_cast<scalar_t>(grad_vj);
      currentIdx += item.get_local_range(0) * fast_dim_size;
      slower_dims_location += item.get_local_range(0);
    }
}

} // anonymous namespace

std::tuple<Tensor,Tensor> weight_norm_opencl(
    const Tensor & v,
    const Tensor & g,
    int64_t dim) {
  auto w = at::empty_like(v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  at::ScalarType AccType = g.scalar_type() == at::ScalarType::Half || g.scalar_type() == at::ScalarType::BFloat16 ?
                           at::ScalarType::Float : g.scalar_type();
  auto norms = at::empty_strided(g.sizes(), g.strides(), g.options().dtype(AccType));

  const int ndims = v.dim();

  if(dim == 0) {
    int rowSize = 1;
    for(int i = ndims - 1; i > 0; i--)
      rowSize *= v.size(i);

    auto stream = at::sycl::getCurrentSYCLStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, v.scalar_type(),
       "weight_norm_fwd_first_dim_kernel", [&] {
         using accscalar_t = acc_type<scalar_t, true>;
         stream.submit([&](sycl::handler& cgh) {
           sycl::local_accessor<accscalar_t, 1> smem(BLOCK, cgh);
           cgh.parallel_for(
               sycl::nd_range<1>(v.size(0) * BLOCK, BLOCK),
               [=, w_ptr=w.mutable_data_ptr<scalar_t>(),
                norms_ptr=norms.mutable_data_ptr<accscalar_t>(),
                v_ptr=v.const_data_ptr<scalar_t>(),
                g_ptr=g.const_data_ptr<scalar_t>()](sycl::nd_item<1> item) {
                 weight_norm_fwd_first_dim_kernel_impl<scalar_t, accscalar_t>(
                     item, w_ptr, norms_ptr, v_ptr, g_ptr, rowSize,
                     smem.get_pointer());
               });
         });
       });
  } else if(dim == ndims - 1) {
    int slower_dims_size = 1;
    for(int i = 0; i < ndims - 1; i++)
      slower_dims_size *= v.size(i);
    int fast_dim_size = v.size(ndims-1);

    auto stream = at::sycl::getCurrentSYCLStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, v.scalar_type(),
       "weight_norm_fwd_last_dim_kernel", [&] {
         using accscalar_t = acc_type<scalar_t, true>;
         int smem_size = TILE_W * TILE_H + TILE_W;
         stream.submit([&](sycl::handler& cgh) {
           sycl::local_accessor<accscalar_t, 1> smem(smem_size, cgh);
           cgh.parallel_for(
               sycl::nd_range<2>(
                   sycl::range<2>(TILE_H, ((fast_dim_size+TILE_W-1)/TILE_W) * TILE_W),
                   sycl::range<2>(TILE_H, TILE_W)),
               [=, w_ptr=w.mutable_data_ptr<scalar_t>(),
                norms_ptr=norms.mutable_data_ptr<accscalar_t>(),
                v_ptr=v.const_data_ptr<scalar_t>(),
                g_ptr=g.const_data_ptr<scalar_t>()](sycl::nd_item<2> item) {
                 weight_norm_fwd_last_dim_kernel_impl<scalar_t, accscalar_t>(
                     item, w_ptr, norms_ptr, v_ptr, g_ptr,
                     fast_dim_size, slower_dims_size, smem.get_pointer());
               });
         });
       });
  }

  return std::tuple<Tensor, Tensor>{w, norms};
}

std::tuple<Tensor, Tensor> weight_norm_backward_opencl(
    const Tensor & grad_w,
    const Tensor & saved_v,
    const Tensor & saved_g,
    const Tensor & saved_norms,
    int64_t dim) {
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(dim == 0 || dim == saved_v.dim() - 1, "fused kernels can only be applied for first or last dim")

  auto grad_v = at::empty_like(saved_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_g = at::empty_like(saved_g, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  const int ndims = saved_v.dim();

  if(dim == 0) {
    int rowSize = 1;
    for(int i = ndims - 1; i > 0; i--)
      rowSize *= saved_v.size(i);

    auto stream = at::sycl::getCurrentSYCLStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, saved_v.scalar_type(),
       "weight_norm_bwd_first_dim_kernel", [&] {
         using accscalar_t = acc_type<scalar_t, true>;
         stream.submit([&](sycl::handler& cgh) {
           sycl::local_accessor<accscalar_t, 1> smem(BLOCK, cgh);
           cgh.parallel_for(
               sycl::nd_range<1>(grad_w.size(0) * BLOCK, BLOCK),
               [=, gv_ptr=grad_v.mutable_data_ptr<scalar_t>(),
                gg_ptr=grad_g.mutable_data_ptr<scalar_t>(),
                gw_ptr=grad_w.const_data_ptr<scalar_t>(),
                sv_ptr=saved_v.const_data_ptr<scalar_t>(),
                sg_ptr=saved_g.const_data_ptr<scalar_t>(),
                sn_ptr=saved_norms.const_data_ptr<accscalar_t>()](sycl::nd_item<1> item) {
                 weight_norm_bwd_first_dim_kernel_impl<scalar_t, accscalar_t>(
                     item, gv_ptr, gg_ptr, gw_ptr, sv_ptr, sg_ptr, sn_ptr,
                     rowSize, smem.get_pointer());
               });
         });
       });
  } else if(dim == ndims - 1) {
    int slower_dims_size = 1;
    for(int i = 0; i < ndims - 1; i++)
      slower_dims_size *= saved_v.size(i);
    int fast_dim_size = saved_v.size(ndims-1);

    auto stream = at::sycl::getCurrentSYCLStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, saved_v.scalar_type(),
       "weight_norm_bwd_last_dim_kernel", [&] {
         using accscalar_t = acc_type<scalar_t, true>;
         int smem_size = TILE_W * TILE_H + TILE_W;
         stream.submit([&](sycl::handler& cgh) {
           sycl::local_accessor<accscalar_t, 1> smem(smem_size, cgh);
           cgh.parallel_for(
               sycl::nd_range<2>(
                   sycl::range<2>(TILE_H, ((fast_dim_size+TILE_W-1)/TILE_W) * TILE_W),
                   sycl::range<2>(TILE_H, TILE_W)),
               [=, gv_ptr=grad_v.mutable_data_ptr<scalar_t>(),
                gg_ptr=grad_g.mutable_data_ptr<scalar_t>(),
                gw_ptr=grad_w.const_data_ptr<scalar_t>(),
                sv_ptr=saved_v.const_data_ptr<scalar_t>(),
                sg_ptr=saved_g.const_data_ptr<scalar_t>(),
                sn_ptr=saved_norms.const_data_ptr<accscalar_t>()](sycl::nd_item<2> item) {
                 weight_norm_bwd_last_dim_kernel_impl<scalar_t, accscalar_t>(
                     item, gv_ptr, gg_ptr, gw_ptr, sv_ptr, sg_ptr, sn_ptr,
                     fast_dim_size, slower_dims_size, smem.get_pointer());
               });
         });
       });
  }

  return std::tuple<Tensor, Tensor>{grad_v, grad_g};
}

#undef BLOCK
#undef TILE_W
#undef TILE_H

} // namespace at::native
