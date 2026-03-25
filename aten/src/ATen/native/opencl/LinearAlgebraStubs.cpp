// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/LinearAlgebraStubs.cpp
//
// SYCL TODO: needs_review - replace with oneMKL equivalent
// The CUDA version uses cuSOLVER/MAGMA for linear algebra operations via lazy loading.
// For the SYCL/OpenCL backend, these should use oneMKL LAPACK equivalents.

#include <ATen/Context.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/native/opencl/MiscUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/TransposeType.h>

#if defined(BUILD_LAZY_OPENCL_LINALG)
#include <ATen/native/opencl/linalg/BatchLinearAlgebraLib.h>
#endif

namespace at::native {
#if defined(BUILD_LAZY_OPENCL_LINALG)
namespace {
// SYCL TODO: needs_review - replace with oneMKL equivalent
opencl::detail::LinalgDispatch disp = {_cholesky_solve_helper_opencl};

at::DynamicLibrary& getTorchLinalgLibrary() {
  static at::DynamicLibrary lib("libtorch_opencl_linalg.so", nullptr, true);
  return lib;
}

void loadLazyTorchLinalgLibrary() {
  static int invoke_count = 0;
  getTorchLinalgLibrary();
  TORCH_CHECK(invoke_count++ == 0, "lazy wrapper should be called at most once");
}

void lazy_cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
  loadLazyTorchLinalgLibrary();
  cholesky_stub(DeviceType::OpenCL, input, info, upper);
}

Tensor& lazy_cholesky_inverse_kernel(Tensor &result, Tensor& infos, bool upper) {
  loadLazyTorchLinalgLibrary();
  return cholesky_inverse_stub(DeviceType::OpenCL, result, infos, upper);
}

void lazy_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  loadLazyTorchLinalgLibrary();
  lu_factor_stub(DeviceType::OpenCL, input, pivots, infos, compute_pivots);
}

void lazy_triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  loadLazyTorchLinalgLibrary();
  triangular_solve_stub(DeviceType::OpenCL, A, B, left, upper, transpose, unitriangular);
}

Tensor& lazy_orgqr_kernel(Tensor& result, const Tensor& tau) {
  loadLazyTorchLinalgLibrary();
  return orgqr_stub(DeviceType::OpenCL, result, tau);
}

void lazy_ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  loadLazyTorchLinalgLibrary();
  ormqr_stub(DeviceType::OpenCL, input, tau, other, left, transpose);
}

void lazy_geqrf_kernel(const Tensor& input, const Tensor& tau) {
  loadLazyTorchLinalgLibrary();
  geqrf_stub(DeviceType::OpenCL, input, tau);
}

void lazy_linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  loadLazyTorchLinalgLibrary();
  linalg_eigh_stub(DeviceType::OpenCL, eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
}

void lazy_linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  getTorchLinalgLibrary();
  linalg_eig_stub(DeviceType::OpenCL, eigenvalues, eigenvectors, infos, input, compute_eigenvectors);
}

void lazy_svd_kernel(const Tensor& A,
                     const bool full_matrices,
                     const bool compute_uv,
                     const std::optional<std::string_view>& driver,
                     const Tensor& U,
                     const Tensor& S,
                     const Tensor& Vh,
                     const Tensor& info) {
  getTorchLinalgLibrary();
  svd_stub(DeviceType::OpenCL, A, full_matrices, compute_uv, driver, U, S, Vh, info);
}

void lazy_lu_solve(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  getTorchLinalgLibrary();
  lu_solve_stub(DeviceType::OpenCL, LU, pivots, B, trans);
}

void lazy_lstsq_kernel(const Tensor& a, Tensor& b, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, std::string driver_name)  {
  getTorchLinalgLibrary();
  lstsq_stub(DeviceType::OpenCL, a, b, rank, singular_values, infos, rcond, driver_name);
}

void lazy_ldl_factor(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  loadLazyTorchLinalgLibrary();
  ldl_factor_stub(DeviceType::OpenCL, LD, pivots, info, upper, hermitian);
}

void lazy_ldl_solve(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  loadLazyTorchLinalgLibrary();
  ldl_solve_stub(DeviceType::OpenCL, LD, pivots, B, upper, hermitian);
}

REGISTER_OPENCL_DISPATCH(cholesky_stub, &lazy_cholesky_kernel)
REGISTER_OPENCL_DISPATCH(cholesky_inverse_stub, &lazy_cholesky_inverse_kernel)
REGISTER_OPENCL_DISPATCH(lu_factor_stub, &lazy_lu_factor)
REGISTER_OPENCL_DISPATCH(ldl_factor_stub, &lazy_ldl_factor)
REGISTER_OPENCL_DISPATCH(ldl_solve_stub, &lazy_ldl_solve)
REGISTER_OPENCL_DISPATCH(triangular_solve_stub, &lazy_triangular_solve_kernel)
REGISTER_OPENCL_DISPATCH(orgqr_stub, &lazy_orgqr_kernel)
REGISTER_OPENCL_DISPATCH(ormqr_stub, &lazy_ormqr_kernel)
REGISTER_OPENCL_DISPATCH(geqrf_stub, &lazy_geqrf_kernel)
REGISTER_OPENCL_DISPATCH(linalg_eigh_stub, &lazy_linalg_eigh_kernel)
REGISTER_OPENCL_DISPATCH(linalg_eig_stub, &lazy_linalg_eig_kernel)
REGISTER_OPENCL_DISPATCH(svd_stub, &lazy_svd_kernel)
REGISTER_OPENCL_DISPATCH(lu_solve_stub, &lazy_lu_solve)
REGISTER_OPENCL_DISPATCH(lstsq_stub, &lazy_lstsq_kernel)
} // anonymous namespace

namespace opencl::detail {
void registerLinalgDispatch(const LinalgDispatch& disp_) {
  disp = disp_;
}
} //namespace opencl::detail

Tensor _cholesky_solve_helper_opencl(const Tensor& self, const Tensor& A, bool upper) {
    getTorchLinalgLibrary();
    TORCH_CHECK(disp.cholesky_solve_helper != _cholesky_solve_helper_opencl, "Can't find _cholesky_solve_helper_opencl");
    return disp.cholesky_solve_helper(self, A, upper);
}

#endif /*defined(BUILD_LAZY_OPENCL_LINALG)*/

} // namespace at::native
