// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/layer_norm_kernel.cu
// SYCL TODO: needs_review - Large complex kernel with vectorized loads,
// Welford online reduction, warp shuffle, shared memory, and multiple
// kernel paths. Sub-group size differences between CUDA (32) and Intel (16/32)
// may affect correctness of warp-level reductions.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/layer_norm.h>

#include <tuple>
#include <type_traits>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/detail/IndexUtils.h>
#include <ATen/native/sycl/block_reduce.h>
#include <ATen/native/sycl/thread_constants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <c10/sycl/SYCLMathCompat.h>
#include <c10/util/env.h>

namespace at::native {

namespace {

constexpr int kSYCLNumThreads = 256;
constexpr unsigned int kWarpSize = 32; // SYCL TODO: needs_review - Intel GPU sub-group may be 16
constexpr int vec_size = 4;

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template<typename T>
bool can_vectorize(const T * ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
};

struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;
  WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U, bool rms_norm>
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  if constexpr (!rms_norm){
    U delta = val - curr_sum.mean;
    U new_count = curr_sum.count + 1.f;
    U new_mean = curr_sum.mean + delta * (1.f/new_count);
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
  } else{
    return {0.f, curr_sum.sigma2 + val * val, 0};
  }
}

template<bool rms_norm>
WelfordDataLN cuWelfordCombine(
  const WelfordDataLN dataB,
  const WelfordDataLN dataA)
{
  if constexpr (!rms_norm){
    using U = decltype(dataB.count);
    U delta = dataB.mean - dataA.mean;
    U count = dataA.count + dataB.count;
    U mean, sigma2;
    if (count > decltype(dataB.count){0}) {
      auto coef = 1.f/count;
      auto nA = dataA.count * coef;
      auto nB = dataB.count * coef;
      mean = nA*dataA.mean + nB*dataB.mean;
      sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
      mean = U(0);
      sigma2 = U(0);
    }
    return {mean, sigma2, count};
  } else {
    return {0.f, dataB.sigma2 + dataA.sigma2, 0};
  }
}

template <typename T, typename T_ACC, bool rms_norm>
void RowwiseMomentsSYCLKernel(
    sycl::nd_item<1> item,
    int64_t N,
    T_ACC eps,
    const T* X,
    T_ACC* mean,
    T_ACC* rstd) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  const int64_t i = item.get_group(0);
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  // SYCL TODO: needs_review - BlockReduce via sub_group
  auto sg = item.get_sub_group();
  for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
    WelfordType other;
    other.mean = sycl::shift_group_left(sg, val.mean, offset);
    other.m2 = sycl::shift_group_left(sg, val.m2, offset);
    other.n = sycl::shift_group_left(sg, val.n, offset);
    other.nf = sycl::shift_group_left(sg, val.nf, offset);
    val = welford_op.combine(val, other);
  }

  if (item.get_local_id(0) == 0) {
    auto [m2, m1] = welford_op.project(val);
    if constexpr (!rms_norm){
      mean[i] = m1;
      rstd[i] = sycl::rsqrt(m2 + eps);
    } else {
      rstd[i] = sycl::rsqrt(m2 + m1 * m1 + eps);
    }
  }
}

template <typename T, typename T_ACC, bool rms_norm>
void LayerNormForwardSYCLKernel(
    sycl::nd_item<1> item,
    int64_t N,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
  const int64_t i = item.get_group(0);
  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    if constexpr (!rms_norm){
      const T_ACC beta_v =
          beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
      Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
              static_cast<T_ACC>(rstd[i]) * gamma_v + beta_v;
    } else {
      Y[index] = (static_cast<T_ACC>(X[index])) * static_cast<T_ACC>(rstd[i]) * gamma_v;
    }
  }
}

// SYCL TODO: needs_review - Vectorized kernel path needs adaptation for
// SYCL aligned loads. The original uses CUDA vectorized loads with aligned_vector.
template<typename T, bool rms_norm = false>
WelfordDataLN compute_stats_sycl(
  sycl::nd_item<2> item,
  const T* X,
  const int N,
  float * buf) {
    using vec_t = aligned_vector<T, vec_size>;
    using acc_t = acc_type<T, true>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(X);
    const int numx = item.get_local_range(0) * item.get_local_range(1);
    const int thrx = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);
    const int n_vec_to_read = N/vec_size;
    WelfordDataLN wd(0.f, 0.f, 0.f);
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      for (int ii=0; ii < vec_size; ii++){
        wd = cuWelfordOnlineSum<acc_t, rms_norm>(static_cast<acc_t>(data.val[ii]), wd);
      }
    }
    // intra-warp reduction via sub_group
    auto sg = item.get_sub_group();
    for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
        WelfordDataLN wdB{
          sycl::shift_group_left(sg, wd.mean, offset),
          sycl::shift_group_left(sg, wd.sigma2, offset),
          sycl::shift_group_left(sg, wd.count, offset)};
        wd = cuWelfordCombine<rms_norm>(wd, wdB);
    }
    // inter-warp reductions
    if (item.get_local_range(0) > 1) {
      float * meansigmabuf = buf;
      float * countbuf = buf + item.get_local_range(0);
      for (int offset = item.get_local_range(0)/2; offset > 0; offset /= 2) {
        if (item.get_local_id(1) == 0 && item.get_local_id(0) >= offset && item.get_local_id(0) < 2*offset) {
          const int wrt_y = item.get_local_id(0) - offset;
          meansigmabuf[2*wrt_y] = wd.mean;
          meansigmabuf[2*wrt_y+1] = wd.sigma2;
          countbuf[wrt_y] = wd.count;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (item.get_local_id(1) == 0 && item.get_local_id(0) < offset) {
          WelfordDataLN wdB{meansigmabuf[2*item.get_local_id(0)],
                          meansigmabuf[2*item.get_local_id(0)+1],
                          countbuf[item.get_local_id(0)]};
          wd = cuWelfordCombine<rms_norm>(wd, wdB);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (item.get_local_id(1) == 0 && item.get_local_id(0) == 0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2/float(N);
      }
      item.barrier(sycl::access::fence_space::local_space);
      return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.f};
    } else {
      return WelfordDataLN{
        sycl::group_broadcast(sg, wd.mean, 0),
        sycl::group_broadcast(sg, wd.sigma2, 0)/float(N),
        0.f};
    }
}

template <typename T, typename T_ACC, bool rms_norm = false,
typename std::enable_if_t<!std::is_same_v<T, double>, int> = 0>
inline void vectorized_layer_norm_kernel_impl(
  sycl::nd_item<2> item,
  const int N,
  T_ACC eps,
  const T* X,
  const T* gamma,
  const T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y,
  float* s_data) {
    auto i1 = item.get_group(0);
    const T * block_row = X + i1 * N;
    WelfordDataLN wd = compute_stats_sycl<T, rms_norm>(item, block_row, N, s_data);

    using vec_t = aligned_vector<T, vec_size>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(block_row);
    const vec_t * gamma_vec = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    const vec_t * beta_vec = (beta != nullptr) ? reinterpret_cast<const vec_t*>(beta) : nullptr;
    vec_t * Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

    const int numx = item.get_local_range(0) * item.get_local_range(1);
    const int thrx = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);
    const int n_vec_to_read = N/vec_size;

    T_ACC rstd_val = sycl::rsqrt(wd.sigma2 + eps);

    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;

      if (gamma_vec != nullptr && beta_vec != nullptr) {
        for (int ii=0; ii < vec_size; ii++){
          if constexpr (!rms_norm){
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) *
              (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
              + static_cast<T_ACC>(beta_vec[i].val[ii]);
          } else {
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) *
              (rstd_val * static_cast<T_ACC>(data.val[ii]));
          }
        }
      } else if (gamma_vec != nullptr) {
        for (int ii=0; ii < vec_size; ii++){
          if constexpr (!rms_norm){
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) *
              (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
          } else {
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) *
              (rstd_val * static_cast<T_ACC>(data.val[ii]));
          }
        }
      } else {
        for (int ii=0; ii < vec_size; ii++){
          if constexpr (!rms_norm){
            out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
          } else {
            out.val[ii] = rstd_val * static_cast<T_ACC>(data.val[ii]);
          }
        }
      }
      Y_vec[i] = out;
    }
    if (item.get_local_id(1) == 0 && item.get_local_id(0) == 0) {
      if constexpr (!rms_norm) {
        mean[i1] = wd.mean;
      }
      rstd[i1] = rstd_val;
    }
}

template <typename T, typename T_ACC, bool rms_norm = false>
void LayerNormForwardKernelSYCL(
    sycl::nd_item<1> item,
    int64_t N,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
  const int64_t i = item.get_group(0);
  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    if constexpr (!rms_norm){
      const T_ACC beta_v = beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
      Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
              static_cast<T_ACC>(rstd[i]) * gamma_v + beta_v;
    } else {
      Y[index] = (static_cast<T_ACC>(X[index])) * static_cast<T_ACC>(rstd[i]) * gamma_v;
    }
  }
}

// SYCL TODO: needs_review - Backward kernels for layer norm
template <typename T, typename T_ACC>
void ComputeInternalGradientsSYCLKernel(
    sycl::nd_item<1> item,
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    acc_type<T, true>* ds,
    acc_type<T, true>* db) {
  const int64_t i = item.get_group(0);
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;
  }
  auto sg = item.get_sub_group();
  for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
    sum1 += sycl::shift_group_left(sg, sum1, offset);
    sum2 += sycl::shift_group_left(sg, sum2, offset);
  }
  if (item.get_local_id(0) == 0) {
    ds[i] = sum1;
    db[i] = sum2;
  }
}

template <typename T, typename T_ACC>
void ComputeGradInputSYCLKernel(
    sycl::nd_item<1> item,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T_ACC* ds,
    const T_ACC* db,
    T* dX) {
  const int64_t i = item.get_group(0);
  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    const T_ACC dy_v = static_cast<T_ACC>(dY[index]);
    const T_ACC x_v = static_cast<T_ACC>(X[index]);
    const T_ACC mean_v = mean[i];
    const T_ACC rstd_v = rstd[i];
    const T_ACC ds_v = ds[i];
    const T_ACC db_v = db[i];
    const T_ACC inv_N = T_ACC(1) / static_cast<T_ACC>(N);

    T_ACC f_grad_input = gamma_v * rstd_v * dy_v;
    T_ACC b_grad_input = (db_v * mean_v - ds_v) * rstd_v * rstd_v * rstd_v * inv_N;
    T_ACC c_grad_input = -b_grad_input * mean_v - db_v * rstd_v * inv_N;

    dX[index] = f_grad_input + b_grad_input * x_v + c_grad_input;
  }
}

template <typename T, typename T_ACC>
void GammaBetaBackwardSYCLKernel(
    sycl::nd_item<1> item,
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dgamma,
    T* dbeta) {
  const int64_t j = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += (dgamma != nullptr)
          ? static_cast<T_ACC>(dY[index]) *
                (static_cast<T_ACC>(X[index]) - mean[i]) * rstd[i]
          : T_ACC(0);
      sum2 += (dbeta != nullptr) ? static_cast<T_ACC>(dY[index]) : T_ACC(0);
    }
    if (dgamma != nullptr) dgamma[j] = sum1;
    if (dbeta != nullptr) dbeta[j] = sum2;
  }
}

template <typename T, bool rms_norm>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  using T_ACC = acc_type<T, true>;
  TORCH_INTERNAL_ASSERT(googly_is_empty(X) || X.numel() == M * N);
  TORCH_INTERNAL_ASSERT(!gamma.defined() || gamma.numel() == N);
  TORCH_INTERNAL_ASSERT(!beta.defined() || beta.numel() == N);

  const T* X_data = X.template const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.template const_data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.template const_data_ptr<T>() : nullptr;
  T* Y_data = Y.template mutable_data_ptr<T>();
  T_ACC* mean_data = rms_norm ? nullptr : mean.template mutable_data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.template mutable_data_ptr<T_ACC>();

  auto sycl_stream = at::sycl::getCurrentSYCLStream();

  bool can_use_vectorized = (N % vec_size == 0) &&
      can_vectorize(X_data, sizeof(T) * vec_size) &&
      can_vectorize(Y_data, sizeof(T) * vec_size);
  if (gamma_data != nullptr) can_use_vectorized = can_use_vectorized && can_vectorize(gamma_data, sizeof(T) * vec_size);
  if (beta_data != nullptr) can_use_vectorized = can_use_vectorized && can_vectorize(beta_data, sizeof(T) * vec_size);

  if (can_use_vectorized && !std::is_same_v<T, double>) {
    int warp_size = 32; // SYCL TODO: needs_review - use actual sub_group size
    int nwarp = std::min<int>(N / (vec_size * warp_size), 4);
    nwarp = std::max(nwarp, 1);
    int block_x = warp_size;
    int block_y = nwarp;

    sycl_stream.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> smem(3 * nwarp, cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(
              sycl::range<2>(block_y, M * block_x),
              sycl::range<2>(block_y, block_x)),
          [=](sycl::nd_item<2> item) {
            vectorized_layer_norm_kernel_impl<T, T_ACC, rms_norm>(
                item, N, static_cast<T_ACC>(eps), X_data, gamma_data, beta_data,
                mean_data, rstd_data, Y_data, smem.get_pointer());
          });
    });
  } else {
    int num_threads = std::min<int>(kSYCLNumThreads, std::max<int>(1, (int)(N + 31) / 32 * 32));
    sycl_stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(M * num_threads, num_threads),
          [=](sycl::nd_item<1> item) {
            RowwiseMomentsSYCLKernel<T, T_ACC, rms_norm>(
                item, N, static_cast<T_ACC>(eps), X_data, mean_data, rstd_data);
          });
    });
    sycl_stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(M * num_threads, num_threads),
          [=](sycl::nd_item<1> item) {
            LayerNormForwardSYCLKernel<T, T_ACC, rms_norm>(
                item, N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
          });
    });
  }
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      X.scalar_type(), "LayerNormKernelImpl", [&]() {
        LayerNormKernelImplInternal<scalar_t, false>(
            X, gamma, beta, M, N, static_cast<scalar_t>(eps), Y, mean, rstd);
      });
}

void RMSNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& rstd) {
  Tensor mean_dummy;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      X.scalar_type(), "RMSNormKernelImpl", [&]() {
        Tensor beta_dummy;
        LayerNormKernelImplInternal<scalar_t, true>(
            X, gamma, beta_dummy, M, N, static_cast<scalar_t>(eps), Y, mean_dummy, rstd);
      });
}

void LayerNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const scalar_t* dY_data = dY.const_data_ptr<scalar_t>();
        const scalar_t* X_data = X.const_data_ptr<scalar_t>();
        const T_ACC* mean_data = mean.const_data_ptr<T_ACC>();
        const T_ACC* rstd_data = rstd.const_data_ptr<T_ACC>();
        const scalar_t* gamma_data = gamma.defined() ? gamma.const_data_ptr<scalar_t>() : nullptr;

        auto sycl_stream = at::sycl::getCurrentSYCLStream();

        if (dX != nullptr) {
          const auto kAccType =
              (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16) ? kFloat : X.scalar_type();
          Tensor ds = at::empty({M}, X.options().dtype(kAccType));
          Tensor db = at::empty({M}, X.options().dtype(kAccType));
          T_ACC* ds_data = ds.mutable_data_ptr<T_ACC>();
          T_ACC* db_data = db.mutable_data_ptr<T_ACC>();
          scalar_t* dX_data = dX->mutable_data_ptr<scalar_t>();

          int num_threads = std::min<int>(kSYCLNumThreads, std::max<int>(1, (int)(N + 31) / 32 * 32));
          sycl_stream.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(M * num_threads, num_threads),
                [=](sycl::nd_item<1> item) {
                  ComputeInternalGradientsSYCLKernel<scalar_t, T_ACC>(
                      item, N, dY_data, X_data, gamma_data, ds_data, db_data);
                });
          });
          sycl_stream.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(M * num_threads, num_threads),
                [=](sycl::nd_item<1> item) {
                  ComputeGradInputSYCLKernel<scalar_t, T_ACC>(
                      item, N, dY_data, X_data, mean_data, rstd_data,
                      gamma_data, ds_data, db_data, dX_data);
                });
          });
        }
        if (dgamma != nullptr || dbeta != nullptr) {
          scalar_t* dgamma_data = dgamma != nullptr ? dgamma->mutable_data_ptr<scalar_t>() : nullptr;
          scalar_t* dbeta_data = dbeta != nullptr ? dbeta->mutable_data_ptr<scalar_t>() : nullptr;
          const int64_t B = (N + kSYCLNumThreads - 1) / kSYCLNumThreads;
          sycl_stream.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(B * kSYCLNumThreads, kSYCLNumThreads),
                [=](sycl::nd_item<1> item) {
                  GammaBetaBackwardSYCLKernel<scalar_t, T_ACC>(
                      item, M, N, dY_data, X_data, mean_data, rstd_data,
                      dgamma_data, dbeta_data);
                });
          });
        }
      });
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl)
REGISTER_DISPATCH(RMSNormKernel, &RMSNormKernelImpl)
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl)

} // namespace at::native
