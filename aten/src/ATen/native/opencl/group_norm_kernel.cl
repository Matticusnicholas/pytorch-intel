// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/group_norm_kernel.cu
// SYCL TODO: needs_review - Complex kernel with shared memory reductions,
// warp-level primitives, and multiple kernel dispatch patterns.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>

#include <type_traits>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <ATen/sycl/detail/IndexUtils.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/block_reduce.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

constexpr int kSYCLNumThreads = 256;
constexpr int kReduceTileSize = 32;

template <typename T>
void RowwiseMomentsSYCLKernel(
    sycl::nd_item<1> item,
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd) {
  using T_ACC = acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  const int64_t i = item.get_group(0);
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  // SYCL TODO: needs_review - BlockReduce needs SYCL work-group reduction
  if (item.get_local_range(0) <= 32) {
    // Warp-level reduction via sub_group
    auto sg = item.get_sub_group();
    for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
      WelfordType other;
      other.mean = sycl::shift_group_left(sg, val.mean, offset);
      other.m2 = sycl::shift_group_left(sg, val.m2, offset);
      other.n = sycl::shift_group_left(sg, val.n, offset);
      other.nf = sycl::shift_group_left(sg, val.nf, offset);
      val = welford_op.combine(val, other);
    }
  } else {
    // SYCL TODO: needs_review - Full block reduction via shared memory
    // Using sub_group cascaded reduction as approximation
    auto sg = item.get_sub_group();
    for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
      WelfordType other;
      other.mean = sycl::shift_group_left(sg, val.mean, offset);
      other.m2 = sycl::shift_group_left(sg, val.m2, offset);
      other.n = sycl::shift_group_left(sg, val.n, offset);
      other.nf = sycl::shift_group_left(sg, val.nf, offset);
      val = welford_op.combine(val, other);
    }
  }
  if (item.get_local_id(0) == 0) {
    auto [m2, m1] = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = sycl::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
void ComputeFusedParamsSYCLKernel(
    sycl::nd_item<1> item,
    int64_t N, int64_t C, int64_t group,
    const T* mean, const T* rstd,
    const T* gamma, const T* beta,
    acc_type<T, true>* a,
    acc_type<T, true>* b) {
  using T_ACC = acc_type<T, true>;
  const int64_t index = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);
  if (index < N * C) {
    const int64_t ng = index / (C / group);
    const int64_t c = index % C;
    const T_ACC scale = (gamma == nullptr)
        ? static_cast<T_ACC>(rstd[ng])
        : static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
    a[index] = scale;
    b[index] = -scale * static_cast<T_ACC>(mean[ng]) +
        ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[c]));
  }
}

template <typename T>
void Compute1dBackwardFusedParamsSYCLKernel(
    sycl::nd_item<2> item,
    int64_t C, int64_t group,
    const T* dY, const T* X,
    const T* mean, const T* rstd, const T* gamma,
    acc_type<T, true>* c2, acc_type<T, true>* c3) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  const int64_t n = item.get_group(0);
  const int64_t g = item.get_group(1);
  const int64_t ng = n * G + g;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t i = item.get_local_id(0); i < D; i += item.get_local_range(0)) {
    const int64_t index = ng * D + i;
    const int64_t c = g * D + i;
    const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
    sum1 += dY[index] * X[index] * gamma_v;
    sum2 += dY[index] * gamma_v;
  }
  // SYCL TODO: needs_review - sub_group reduction
  auto sg = item.get_sub_group();
  for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
    sum1 += sycl::shift_group_left(sg, sum1, offset);
    sum2 += sycl::shift_group_left(sg, sum2, offset);
  }
  if (item.get_local_id(0) == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D);
    const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
        static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
        static_cast<T_ACC>(rstd[ng]) * s;
    c2[ng] = x;
    c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
        sum2 * static_cast<T_ACC>(rstd[ng]) * s;
  }
}

template <typename T>
void GammaBeta1dBackwardSYCLKernel1(
    sycl::nd_item<1> item,
    int64_t N, int64_t C, int64_t group,
    const T* dY, const T* X, const T* mean, const T* rstd,
    T* dgamma, T* dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t c = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      const T_ACC dy_acc = static_cast<T_ACC>(dY[nc]);
      const T_ACC x_acc = static_cast<T_ACC>(X[nc]);
      sum1 += (dgamma == nullptr) ? T_ACC(0)
          : ((dy_acc * x_acc - dy_acc * static_cast<T_ACC>(mean[ng])) *
             static_cast<T_ACC>(rstd[ng]));
      sum2 += (dbeta == nullptr) ? T_ACC(0) : dy_acc;
    }
    if (dgamma != nullptr) dgamma[c] = sum1;
    if (dbeta != nullptr) dbeta[c] = sum2;
  }
}

template <typename T>
void ComputeInternalGradientsSYCLKernel(
    sycl::nd_item<1> item,
    int64_t HxW, const T* dY, const T* X,
    acc_type<T, true>* ds, acc_type<T, true>* db) {
  using T_ACC = acc_type<T, true>;
  const int64_t nc = item.get_group(0);
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t hw = item.get_local_id(0); hw < HxW; hw += item.get_local_range(0)) {
    const int64_t index = nc * HxW + hw;
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]);
    sum2 += static_cast<T_ACC>(dY[index]);
  }
  auto sg = item.get_sub_group();
  for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
    sum1 += sycl::shift_group_left(sg, sum1, offset);
    sum2 += sycl::shift_group_left(sg, sum2, offset);
  }
  if (item.get_local_id(0) == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

template <typename T>
void ComputeBackwardFusedParamsSYCLKernel(
    sycl::nd_item<2> item,
    int64_t C, int64_t HxW, int64_t group,
    const T* mean, const T* rstd, const T* gamma,
    const acc_type<T, true>* ds, const acc_type<T, true>* db,
    acc_type<T, true>* c2, acc_type<T, true>* c3) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  const int64_t n = item.get_group(0);
  const int64_t g = item.get_group(1);
  const int64_t ng = n * G + g;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t i = item.get_local_id(0); i < D; i += item.get_local_range(0)) {
    const int64_t index = ng * D + i;
    const int64_t c = g * D + i;
    const T_ACC gamma_v = gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
    sum1 += ds[index] * gamma_v;
    sum2 += db[index] * gamma_v;
  }
  auto sg = item.get_sub_group();
  for (int offset = sg.get_local_range()[0] / 2; offset > 0; offset /= 2) {
    sum1 += sycl::shift_group_left(sg, sum1, offset);
    sum2 += sycl::shift_group_left(sg, sum2, offset);
  }
  if (item.get_local_id(0) == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
    const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
        static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
        static_cast<T_ACC>(rstd[ng]) * s;
    c2[ng] = x;
    c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
        sum2 * static_cast<T_ACC>(rstd[ng]) * s;
  }
}

template <typename T>
void GammaBetaBackwardSYCLKernel1(
    sycl::nd_item<1> item,
    int64_t N, int64_t C, int64_t group,
    const T* mean, const T* rstd,
    const acc_type<T, true>* ds, const acc_type<T, true>* db,
    T* dgamma, T* dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t c = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      sum1 += (dgamma == nullptr) ? T_ACC(0)
          : ((ds[nc] - db[nc] * static_cast<T_ACC>(mean[ng])) *
             static_cast<T_ACC>(rstd[ng]));
      sum2 += (dbeta == nullptr) ? T_ACC(0) : db[nc];
    }
    if (dgamma != nullptr) dgamma[c] = sum1;
    if (dbeta != nullptr) dbeta[c] = sum2;
  }
}

template <typename T>
void GroupNorm1dForward(
    const Tensor& X, const Tensor& mean, const Tensor& rstd,
    const Tensor& gamma, const Tensor& beta,
    int64_t N, int64_t C, int64_t group, Tensor& Y) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  if (gamma.defined() && beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    sycl_kernel(iter, [] SYCL_LAMBDA(T x, T mean, T rstd, T gamma, T beta) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma) +
          static_cast<T_ACC>(beta);
    });
  } else if (gamma.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .build();
    sycl_kernel(iter, [] SYCL_LAMBDA(T x, T mean, T rstd, T gamma) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
    });
  } else if (beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    sycl_kernel(iter, [] SYCL_LAMBDA(T x, T mean, T rstd, T beta) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd) + static_cast<T_ACC>(beta);
    });
  } else {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D}))
                    .add_owned_const_input(X.view({N * G, D}))
                    .add_owned_input(mean.view({N * G, 1}))
                    .add_owned_input(rstd.view({N * G, 1}))
                    .build();
    sycl_kernel(iter, [] SYCL_LAMBDA(T x, T mean, T rstd) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd);
    });
  }
}

template <typename T>
void GroupNormKernelImplInternal(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    int64_t N, int64_t C, int64_t HxW, int64_t group,
    T eps, Tensor& Y, Tensor& mean, Tensor& rstd) {
  using T_ACC = acc_type<T, true>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) return;

  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = mean.mutable_data_ptr<T>();
  T* rstd_data = rstd.mutable_data_ptr<T>();

  auto sycl_stream = at::sycl::getCurrentSYCLStream();
  const int64_t num_threads = D * HxW < 128 ? 32 : 128;

  sycl_stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(N * G * num_threads, num_threads),
        [=](sycl::nd_item<1> item) {
          RowwiseMomentsSYCLKernel<T>(item, D * HxW, eps, X_data, mean_data, rstd_data);
        });
  });

  if (HxW == 1) {
    GroupNorm1dForward<T>(X, mean, rstd, gamma, beta, N, C, G, Y);
  } else if (!gamma.defined() && !beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D * HxW}))
                    .add_owned_const_input(X.view({N * G, D * HxW}))
                    .add_owned_input(mean.view({N * G, 1}))
                    .add_owned_input(rstd.view({N * G, 1}))
                    .build();
    sycl_kernel(iter, [] SYCL_LAMBDA(T x, T mean, T rstd) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd);
    });
  } else {
    const auto kAccType =
        (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16) ? kFloat : X.scalar_type();
    Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
    Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
    T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
    T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

    const int64_t B = (N * C + kSYCLNumThreads - 1) / kSYCLNumThreads;
    sycl_stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(B * kSYCLNumThreads, kSYCLNumThreads),
          [=](sycl::nd_item<1> item) {
            ComputeFusedParamsSYCLKernel<T>(
                item, N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
          });
    });

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * C, HxW}))
                    .add_owned_const_input(X.view({N * C, HxW}))
                    .add_owned_input(a.view({N * C, 1}))
                    .add_owned_input(b.view({N * C, 1}))
                    .build();
    sycl_kernel(iter, [] SYCL_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
      return a * static_cast<T_ACC>(x) + b;
    });
  }
}

void GroupNormKernelImpl(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    int64_t N, int64_t C, int64_t HxW, int64_t group,
    double eps, Tensor& Y, Tensor& mean, Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      X.scalar_type(), "GroupNormKernelImpl", [&]() {
        GroupNormKernelImplInternal<scalar_t>(
            X, gamma, beta, N, C, HxW, group,
            static_cast<scalar_t>(eps), Y, mean, rstd);
      });
}

// SYCL TODO: needs_review - Backward kernels use shared memory tile reductions
// that need validation on Intel GPU sub-group sizes
template <typename T>
void GroupNormBackwardKernelImplInternal(
    const Tensor& dY, const Tensor& X, const Tensor& mean,
    const Tensor& rstd, const Tensor& gamma,
    int64_t N, int64_t C, int64_t HxW, int64_t group,
    Tensor& dX, Tensor& dgamma, Tensor& dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * G);
  TORCH_CHECK(rstd.numel() == N * G);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  auto sycl_stream = at::sycl::getCurrentSYCLStream();

  if (N == 0) {
    if (dgamma.defined()) dgamma.fill_(T(0));
    if (dbeta.defined()) dbeta.fill_(T(0));
    return;
  }

  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16) ? kFloat : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* ds_data = ds.mutable_data_ptr<T_ACC>();
  T_ACC* db_data = db.mutable_data_ptr<T_ACC>();

  if (HxW == 1) {
    // 1d path - simplified
    // SYCL TODO: needs_review - 1d backward path needs full implementation
    if (dX.defined()) {
      const T* gamma_data_local = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
      Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
      Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
      T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
      T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();
      const int64_t num_threads = (C / G) < 128 ? 32 : 128;
      sycl_stream.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(num_threads, N * G), sycl::range<2>(num_threads, 1)),
            [=](sycl::nd_item<2> item) {
              Compute1dBackwardFusedParamsSYCLKernel<T>(
                  item, C, G, dY_data, X_data, mean_data, rstd_data,
                  gamma_data_local, c2_data, c3_data);
            });
      });

      if (gamma.defined()) {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(dX.view({N, G, D}))
                        .add_owned_const_input(dY.view({N, G, D}))
                        .add_owned_const_input(X.view({N, G, D}))
                        .add_owned_const_input(rstd.view({N, G, 1}))
                        .add_owned_const_input(gamma.view({1, G, D}))
                        .add_owned_const_input(c2.view({N, G, 1}))
                        .add_owned_const_input(c3.view({N, G, 1}))
                        .build();
        sycl_kernel(iter,
            [] SYCL_LAMBDA(T dy, T x, T rstd, T gamma, T_ACC c2, T_ACC c3) -> T {
              const T_ACC c1 = static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
              return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
            });
      } else {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(dX.view({N * G, D}))
                        .add_owned_const_input(dY.view({N * G, D}))
                        .add_owned_const_input(X.view({N * G, D}))
                        .add_owned_const_input(rstd.view({N * G, 1}))
                        .add_owned_const_input(c2.view({N * G, 1}))
                        .add_owned_const_input(c3.view({N * G, 1}))
                        .build();
        sycl_kernel(iter,
            [] SYCL_LAMBDA(T dy, T x, T rstd, T_ACC c2, T_ACC c3) -> T {
              const T_ACC c1 = static_cast<T_ACC>(rstd);
              return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
            });
      }
    }
    if (dgamma.defined() || dbeta.defined()) {
      T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
      T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
      const int64_t B = (C + kSYCLNumThreads - 1) / kSYCLNumThreads;
      sycl_stream.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(B * kSYCLNumThreads, kSYCLNumThreads),
            [=](sycl::nd_item<1> item) {
              GammaBeta1dBackwardSYCLKernel1<T>(
                  item, N, C, G, dY_data, X_data, mean_data, rstd_data,
                  dgamma_data, dbeta_data);
            });
      });
    }
    return;
  }

  int64_t num_threads = HxW < 128 ? 32 : 128;
  sycl_stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(N * C * num_threads, num_threads),
        [=](sycl::nd_item<1> item) {
          ComputeInternalGradientsSYCLKernel<T>(
              item, HxW, dY_data, X_data, ds_data, db_data);
        });
  });

  if (dX.defined()) {
    Tensor c1 = at::empty({0}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
    T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .add_output(c1)
                      .add_owned_const_input(rstd.view({N, G, 1}))
                      .add_owned_const_input(gamma.view({1, G, D}))
                      .build();
      sycl_kernel(iter, [] SYCL_LAMBDA(T rstd, T gamma) -> T_ACC {
        return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
      });
    }

    num_threads = (C / G) < 128 ? 32 : 128;
    sycl_stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(num_threads, N * G), sycl::range<2>(num_threads, 1)),
          [=](sycl::nd_item<2> item) {
            ComputeBackwardFusedParamsSYCLKernel<T>(
                item, C, HxW, G, mean_data, rstd_data, gamma_data,
                ds_data, db_data, c2_data, c3_data);
          });
    });

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D, HxW}))
                      .add_owned_const_input(dY.view({N * G, D, HxW}))
                      .add_owned_const_input(X.view({N * G, D, HxW}))
                      .add_owned_const_input(c1.view({N * G, D, 1}))
                      .add_owned_const_input(c2.view({N * G, 1, 1}))
                      .add_owned_const_input(c3.view({N * G, 1, 1}))
                      .build();
      sycl_kernel(iter,
          [] SYCL_LAMBDA(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) -> T {
            return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
          });
    } else {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D * HxW}))
                      .add_owned_const_input(dY.view({N * G, D * HxW}))
                      .add_owned_const_input(X.view({N * G, D * HxW}))
                      .add_owned_const_input(rstd.view({N * G, 1}))
                      .add_owned_const_input(c2.view({N * G, 1}))
                      .add_owned_const_input(c3.view({N * G, 1}))
                      .build();
      sycl_kernel(iter,
          [] SYCL_LAMBDA(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) -> T {
            return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
          });
    }
  }
  if (dgamma.defined() || dbeta.defined()) {
    T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
    const int64_t B = (C + kSYCLNumThreads - 1) / kSYCLNumThreads;
    sycl_stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(B * kSYCLNumThreads, kSYCLNumThreads),
          [=](sycl::nd_item<1> item) {
            GammaBetaBackwardSYCLKernel1<T>(
                item, N, C, G, mean_data, rstd_data, ds_data, db_data,
                dgamma_data, dbeta_data);
          });
    });
  }
}

void GroupNormBackwardKernelImpl(
    const Tensor& dY, const Tensor& X, const Tensor& mean,
    const Tensor& rstd, const Tensor& gamma,
    int64_t N, int64_t C, int64_t HxW, int64_t group,
    Tensor& dX, Tensor& dgamma, Tensor& dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        GroupNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
      });
}

} // namespace

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl)
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl)

} // namespace at::native
