// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/int8mm.cu

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/sycl/SYCLContext.h>
#include <c10/sycl/SYCLGuard.h>

namespace at::native {

void weight_int8pack_mm_kernel_impl(
    sycl::nd_item<2> item,
    const float* x,
    const int8_t* w,
    const float* scale,
    float* out,
    int B,
    int K,
    int N) {
  int b = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
  int n = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  if (b >= B || n >= N)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += x[b * K + k] * static_cast<float>(w[n * K + k]);
  }

  out[b * N + n] = acc * scale[n];
}

void launch_weight_int8pack_mm_opencl_kernel(
    const Tensor& x,
    const Tensor& w_int8,
    const Tensor& scale,
    Tensor& out) {
  const int B = x.size(0);
  const int K = x.size(1);
  const int N = w_int8.size(0);

  const int block_x = 16;
  const int block_y = 16;
  const int grid_x = (N + block_x - 1) / block_x;
  const int grid_y = (B + block_y - 1) / block_y;

  auto stream = at::sycl::getCurrentSYCLStream();

  stream.submit([&](sycl::handler& cgh) {
    auto x_ptr = x.data_ptr<float>();
    auto w_ptr = w_int8.data_ptr<int8_t>();
    auto s_ptr = scale.data_ptr<float>();
    auto o_ptr = out.data_ptr<float>();
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(grid_x * block_x, grid_y * block_y),
            sycl::range<2>(block_x, block_y)),
        [=](sycl::nd_item<2> item) {
          weight_int8pack_mm_kernel_impl(item, x_ptr, w_ptr, s_ptr, o_ptr, B, K, N);
        });
  });
}

at::Tensor _weight_int8pack_mm_opencl(
    const at::Tensor& x,
    const at::Tensor& w_int8,
    const at::Tensor& scale) {
  TORCH_CHECK(x.is_cuda() || x.is_xpu(), "x must be a GPU tensor");
  TORCH_CHECK(w_int8.is_cuda() || w_int8.is_xpu(), "w must be a GPU tensor");
  TORCH_CHECK(scale.is_cuda() || scale.is_xpu(), "scale must be a GPU tensor");

  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w_int8.dim() == 2, "w must be 2D");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D");

  TORCH_CHECK(
      x.size(1) == w_int8.size(1),
      "K dimension mismatch: x.size(1) != w.size(1)");
  TORCH_CHECK(
      w_int8.size(0) == scale.size(0),
      "Output dim mismatch: w.size(0) != scale.size(0)");

  auto B = x.size(0);
  auto N = w_int8.size(0);

  auto x_f32 = x.to(at::kFloat);
  auto w_int8_contiguous = w_int8.contiguous();
  auto scale_f32 = scale.to(at::kFloat);

  auto out = at::empty({B, N}, x_f32.options());

  launch_weight_int8pack_mm_opencl_kernel(
      x_f32, w_int8_contiguous, scale_f32, out);

  return out.to(x.dtype());
}

} // namespace at::native
