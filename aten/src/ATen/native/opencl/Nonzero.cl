// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Nonzero.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/EmptyTensor.h>
#include <ATen/sycl/detail/KernelUtils.h>
#include <c10/sycl/SYCLCachingAllocator.h>
#include <ATen/sycl/cub.h>
#include <ATen/native/sycl/OffsetCalculator.h> //for MAX_DIMS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/nonzero_native.h>
#endif

namespace at::native {

namespace {
template <typename T>
struct NonZeroOp {
  inline bool operator()(const T& a) const {
    return (a != T(0));
  }
};

// TODO: actually support int64_t index_t
template <typename index_t>
struct TensorDims {
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
/* SYCL kernel */ void write_indices(
    int64_t* inp,
    TensorDims<index_t> dims,
    int ndim,
    index_t n,
    int64_t * total = nullptr,
    int64_t fill_value = -1) {
  auto index = item.get_local_id(0) + (int64_t)item.get_group(0) * item.get_local_range(0);
  bool cond = (total == nullptr || index < *total);
  if (index < n && cond) {
    index_t div = 1;
    int64_t idx_flat = inp[index];
#pragma unroll
    for (int dim = MAX_DIMS; dim >= 0; dim--) {
      if (dim > ndim - 1)
        continue;
      auto dim_size = dims.sizes[dim];
      inp[index + dim * n] = (idx_flat / div) % dim_size;
      div *= dim_size;
    }
  } else if (index < n) {
    // 0th dim has correct values already
    for (int dim = ndim - 1; dim > 0; dim--) {
      inp[index + dim * n] = fill_value;
    }
  }
}

/* SYCL kernel */ void write_fill_value(int64_t * inp, int64_t * total, int64_t fill_value, int64_t n){
  int64_t total_val = *total;
  // not aiming for vectorized stores

  for (int64_t idx = total_val + (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); idx < n; idx += item.get_local_range(0) * item.get_group_range(0)) {
      inp[idx] = fill_value;
  }
}

template <int BLOCK_THREADS>
/* SYCL kernel */ void compute_agg(int32_t * agg, int64_t * agg_cum, uint32_t n_blocks) {

  using BlockScanT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<int64_t, BLOCK_THREADS, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_SCAN_WARP_SCANS>;
  /* SYCL: use sycl::local_accessor */ typename BlockScanT::TempStorage temp_storage;
  int agg_data;
  int64_t agg_cum_data;
  agg_data = item.get_local_id(0) < n_blocks ? agg[item.get_local_id(0)] : 0;
  BlockScanT(temp_storage).InclusiveSum(agg_data, agg_cum_data);
  if (item.get_local_id(0) < n_blocks) {
    agg_cum[item.get_local_id(0)] = agg_cum_data;
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T>
/* SYCL kernel */ void flag_kernel(const T* d_in, int64_t * d_out, const int64_t * agg, int64_t input_nelem, int64_t output_nelem, int iters_per_cta) {
  int64_t start_idx = BLOCK_THREADS * ITEMS_PER_THREAD * iters_per_cta * (int64_t)item.get_group(0);
  if (start_idx >= input_nelem) return;
  d_in += start_idx;

  using BlockLoadT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_LOAD_WARP_TRANSPOSE>;

  // Specialize BlockScan type for our thread block
  using BlockScanT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<int, BLOCK_THREADS, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_SCAN_WARP_SCANS>;
  using TransformInputIteratorT = ATEN_CUB_TRANSFORM_ITERATOR(int, NonZeroOp<T>, const T*);
  using BlockExchangeT =  ROCM_HIPCUB(at_cuda_detail::cub)::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD>;

  // Shared memory
  /* SYCL: use sycl::local_accessor */ union TempStorage
  {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage scan;
    typename BlockExchangeT::TempStorage exchange;
  } temp_storage;

  int64_t aggregate = item.get_group(0) == 0 ? 0 : agg[item.get_group(0) - 1];
  d_out += aggregate;

  TransformInputIteratorT t_input_itr(d_in, NonZeroOp<T>());

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];
  int out_indices[ITEMS_PER_THREAD];

  int64_t remaining =  input_nelem - start_idx;
  int64_t out_remaining = output_nelem - aggregate;
  for (int i=0; i<iters_per_cta; i++){

  // Load items into a blocked arrangement
    if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
      BlockLoadT(temp_storage.load).Load(t_input_itr, data);
    } else {
      BlockLoadT(temp_storage.load).Load(t_input_itr, data, remaining, int(0));
    }

    // Barrier for smem reuse
    item.barrier(sycl::access::fence_space::local_space);

    // Compute inclusive prefix sum
    int aggregate;
    /* SYCL: use sycl::local_accessor */ int aggregate_sh;
    BlockScanT(temp_storage.scan).ExclusiveSum(data, out_indices, aggregate);

    if (item.get_local_id(0) == 0){
      aggregate_sh = aggregate;
    }

    // Barrier for smem reuse
    item.barrier(sycl::access::fence_space::local_space);
    // striped arrangement will provide a slightly better
    // coalescing for writes (although it's still bad because it's indirect indexing)
    BlockExchangeT(temp_storage.exchange).BlockedToStriped(data);
    item.barrier(sycl::access::fence_space::local_space);
    BlockExchangeT(temp_storage.exchange).BlockedToStriped(out_indices);
    for (int ii=0; ii<ITEMS_PER_THREAD; ii++){
      if (data[ii] != 0 && out_indices[ii] < out_remaining) {
        int64_t inp_idx = start_idx + item.get_local_id(0) + item.get_local_range(0) * ii;
        d_out[out_indices[ii]] = inp_idx;
      }
    }

    out_remaining -= aggregate_sh;
    remaining -= BLOCK_THREADS * ITEMS_PER_THREAD;
    if (remaining <= 0 || out_remaining <= 0) return;
    d_out += aggregate_sh;
    t_input_itr += BLOCK_THREADS * ITEMS_PER_THREAD;
    start_idx += BLOCK_THREADS * ITEMS_PER_THREAD;
    item.barrier(sycl::access::fence_space::local_space);
  }

}



} // anonymous namespace

template <typename scalar_t>
void nonzero_cuda_out_impl(const Tensor& self, Tensor& out) {
  Tensor self_ = self.contiguous();
  const sycl::queue* stream = at::sycl::getCurrentSYCLStream();
  int64_t chunk_size, num_chunks;
  if (self.numel() < std::numeric_limits<int>::max()) {
    chunk_size = self.numel();
    num_chunks = 1;
  } else {
    chunk_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
    num_chunks = (self.numel() + chunk_size - 1) / chunk_size;
  }
  // compute number of nonzero elements
  size_t temp_storage_bytes = 0;
  auto& allocator = *c10::sycl::SYCLCachingAllocator::get();
  auto num_nonzeros = allocator.allocate(sizeof(int) * num_chunks);
  for (int64_t idx = 0; idx < num_chunks; idx++) {
    int64_t remaining = std::min<int64_t>(chunk_size, self.numel() - idx * chunk_size);
    ATEN_CUB_TRANSFORM_ITERATOR(bool, NonZeroOp<scalar_t>, const scalar_t*) itr(
        self_.const_data_ptr<scalar_t>() + idx * chunk_size,
        NonZeroOp<scalar_t>());
    AT_CUDA_CHECK(cub::DeviceReduce::Sum(
        nullptr,
        temp_storage_bytes,
        itr,
        ((int*)num_nonzeros.get()) + idx,
        remaining,
        stream));
    auto temp_storage = allocator.allocate(temp_storage_bytes);
    AT_CUDA_CHECK(cub::DeviceReduce::Sum(
        temp_storage.get(),
        temp_storage_bytes,
        itr,
        ((int*)num_nonzeros.get()) + idx,
        remaining,
        stream));
  }
  auto pinned_num_nonzeros_h = at::detail::empty_cpu(
      {num_chunks}, /* size */
      c10::CppTypeToScalarType<int>(), /* dtype */
      std::nullopt, /* layout */
      std::nullopt, /* device */
      true, /* pin_memory */
      std::nullopt /* memory format */
  );
  at::cuda::memcpy_and_sync(
      pinned_num_nonzeros_h.template data_ptr<int>(),
      num_nonzeros.get(),
      sizeof(int) * num_chunks,
      cudaMemcpyDeviceToHost,
      stream);
  int64_t num_nonzeros_h = 0;

  for (int64_t idx = 0; idx < num_chunks; idx++) {
    num_nonzeros_h += pinned_num_nonzeros_h.template const_data_ptr<int>()[idx];
  }
  // num_nonzeros_h = (int)*(pinned_num_nonzeros_h.const_data_ptr<int>());
  // expected output size is num_nonzeros x ndim
  // we are producing output with size {num_nonzeros, ndim} and strides {1,
  // num_nonzeros} (that is, transposed ndim x num_nonzeros output) we are able
  // to directly use passed output with this size and strides, and we can also
  // (per contract) resize passed output with incorrect sizes anyway we want.
  // However, out with correct sizes and incorrect strides will have to be
  // copied to from the intermediate we've produced.
  bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros_h &&
      out.sizes()[1] == self.dim() && !out.t().is_contiguous();
  at::Tensor out_temp = need_to_copy
      ? Tensor(
            at::detail::empty_sycl({self.dim(), num_nonzeros_h}, out.options()))
      : out.resize_({self.dim(), num_nonzeros_h});
  // Scalars are expected to produce output of size (1,0), so we can't write to
  // it
  int64_t curr_nonzeros = 0;
  if (self.dim() > 0) {
    for (int64_t idx = 0; idx < num_chunks; idx++) {
      int remaining = std::min<int64_t>(chunk_size, self.numel() - idx * chunk_size);

      ATEN_CUB_COUNTING_ITERATOR(int64_t) counting_itr(idx * chunk_size);
      ATEN_CUB_TRANSFORM_ITERATOR(bool, NonZeroOp<scalar_t>, const scalar_t*)
          itr(self_.const_data_ptr<scalar_t>() + idx * chunk_size,
              NonZeroOp<scalar_t>());
      temp_storage_bytes = 0;
      AT_CUDA_CHECK(cub::DeviceSelect::Flagged(
          nullptr,
          temp_storage_bytes,
          counting_itr,
          itr,
          out_temp.mutable_data_ptr<int64_t>(),
          ((int*)num_nonzeros.get()) + idx,
          remaining,
          stream));
      auto temp_storage = allocator.allocate(temp_storage_bytes);
      AT_CUDA_CHECK(cub::DeviceSelect::Flagged(
          temp_storage.get(),
          temp_storage_bytes,
          counting_itr,
          itr,
          out_temp.mutable_data_ptr<int64_t>() + curr_nonzeros,
          ((int*)num_nonzeros.get()) + idx,
          remaining,
          stream));
      curr_nonzeros += pinned_num_nonzeros_h.template const_data_ptr<int>()[idx];
    }
    if (num_nonzeros_h > 0 && self.dim() > 1) {
      TensorDims<int64_t> dims;
      for (int i = 0; i < self.dim(); i++) {
        dims.sizes[i] = self.sizes()[i];
      }
      const int nthreads = 256;
      const int nblocks = (num_nonzeros_h + nthreads - 1) / nthreads;
      write_indices/* SYCL: kernel launch with nd_range(nblocks, nthreads, 0, stream) */(
          out_temp.mutable_data_ptr<int64_t>(),
          dims,
          self.dim(),
          num_nonzeros_h);
      // SYCL: kernel launch check handled by SYCL runtime;
    }
  }
  if (need_to_copy) {
    out.copy_(out_temp.t());
  } else {
    // transpose out so it is correct size
    Tensor out_ = out_temp.t();
    out.set_(out_);
  }
}

template <typename scalar_t>
void nonzero_static_cuda_out_impl(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& out) {
  Tensor self_contiguous_ = self.contiguous();
  // see comment in nonzero_cuda_out_impl on reqs for out
  bool out_correct_size =
      out.dim() == 2 && out.sizes()[0] == size && out.sizes()[1] == self.dim();
  bool need_to_copy = out_correct_size && !out.t().is_contiguous();
  if (!out_correct_size) {
    out.resize_({self.dim(), size}).t();
  }
  if (out.numel() == 0) return;
  // we need to allocate temporary out to then copy to user provided out
  at::Tensor out_temp;
  if (need_to_copy) {
    out_temp =
        Tensor(at::detail::empty_sycl({self.dim(), size}, out.options())).t();
  }
  // If input has zero elements, avoid kernel grid calculations (which can
  // produce zero divisors) and just fill the output with fill_value.
  if (self.numel() == 0) {
    if (need_to_copy) {
      out_temp.fill_(fill_value);
      out.copy_(out_temp);
    } else {
      out.fill_(fill_value);
    }
    return;
  }
  int64_t* out_data_ptr = need_to_copy ? out_temp.mutable_data_ptr<int64_t>()
                                       : out.mutable_data_ptr<int64_t>();

  const scalar_t * in_data_ptr = self_contiguous_.const_data_ptr<scalar_t>();
  constexpr int BLOCK_THREADS = 512; //block_threads<sizeof(scalar_t)>();
  constexpr int ITEMS_PER_THREAD = 16;
  auto grid_size = (self.numel() + BLOCK_THREADS * ITEMS_PER_THREAD - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);
  const int64_t num_sms = at::sycl::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t target_blocks = sizeof(scalar_t) == 1 ? 2 * num_sms : num_sms;
  const int iters_per_cta = (grid_size + target_blocks - 1)/target_blocks;
  grid_size = (self.numel() + iters_per_cta * BLOCK_THREADS * ITEMS_PER_THREAD - 1) / (iters_per_cta * BLOCK_THREADS * ITEMS_PER_THREAD);
  auto& allocator = *c10::sycl::SYCLCachingAllocator::get();
  auto agg = allocator.allocate(grid_size * sizeof(int));
  at::sycl::onedpl::calc_block_sums<BLOCK_THREADS, ITEMS_PER_THREAD, true>
  /* SYCL: kernel launch with nd_range(grid_size, BLOCK_THREADS, 0, at::sycl::getCurrentSYCLStream()) */(
    in_data_ptr, (int*)agg.get(), self.numel(), iters_per_cta);
  // SYCL: kernel launch check handled by SYCL runtime;
  auto agg_cum = allocator.allocate(grid_size * sizeof(int64_t));
  // computing partial sums in int64 in the flag kernel
  // leads to 20-30% slowdown, so compute them in a separate 2 us kernel
  compute_agg<BLOCK_THREADS>/* SYCL: kernel launch with nd_range(1, BLOCK_THREADS, 0, at::sycl::getCurrentSYCLStream()) */(
   (int*)agg.get(), (int64_t*)agg_cum.get(), grid_size
  );
  // SYCL: kernel launch check handled by SYCL runtime;
  flag_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
  /* SYCL: kernel launch with nd_range(grid_size, BLOCK_THREADS, 0, at::sycl::getCurrentSYCLStream()) */(
    in_data_ptr, out_data_ptr, (int64_t*)agg_cum.get(), self.numel(), size, iters_per_cta);
  // SYCL: kernel launch check handled by SYCL runtime;
  int64_t out_grid = std::min<int64_t>(num_sms, (size + BLOCK_THREADS - 1)/BLOCK_THREADS);
  write_fill_value/* SYCL: kernel launch with nd_range(out_grid, BLOCK_THREADS, 0, at::sycl::getCurrentSYCLStream()) */(out_data_ptr, (int64_t *)agg_cum.get() + grid_size - 1, fill_value, size);
  if (self.dim() > 1) {
    TensorDims<int64_t> dims;
    for (int i = 0; i < self.dim(); i++) {
      dims.sizes[i] = self.sizes()[i];
    }
    const int nthreads = 256;
    const int nblocks = (size + nthreads - 1) / nthreads;
    write_indices/* SYCL: kernel launch with nd_range(nblocks, nthreads, 0, at::sycl::getCurrentSYCLStream()) */(
        out_data_ptr,
        dims,
        self.dim(),
        size,
        (int64_t *)agg_cum.get() + grid_size - 1,
        fill_value);
    // SYCL: kernel launch check handled by SYCL runtime;
  }
  if (need_to_copy) {
    out.copy_(out_temp);
  }
}

Tensor& nonzero_out_opencl(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= MAX_DIMS,
      "nonzero is not supported for tensor with more than ",
      MAX_DIMS,
      " dimensions");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "nonzero_opencl",
      [&] { nonzero_cuda_out_impl<scalar_t>(self, out); });
  return out;
}

Tensor nonzero_cuda(const Tensor& self) {
  Tensor out = at::detail::empty_sycl({0}, self.options().dtype(kLong));
  return at::native::nonzero_out_opencl(self, out);
}

Tensor& nonzero_static_out_opencl(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& out) {
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "nonzero_static: Expected out tensor to have scalar type ",
      at::kLong,
      " but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= MAX_DIMS,
      "nonzero_static is not supported for tensor with more than ",
      MAX_DIMS,
      " dimensions");
  TORCH_CHECK(
      size >= 0, "nonzero_static: 'size' must be an non-negative integer"
  )
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "nonzero_opencl",
      [&] {
        nonzero_static_cuda_out_impl<scalar_t>(self, size, fill_value, out);
      });
  return out;
}

Tensor nonzero_static_cuda(
    const Tensor& self,
    int64_t size,
    int64_t fill_value) {
  TORCH_CHECK(
      size >= 0, "nonzero_static: 'size' must be an non-negative integer"
  )
  Tensor out = Tensor(at::detail::empty_sycl(
                          {self.dim(), size}, self.options().dtype(kLong)))
                   .t();
  return at::native::nonzero_static_out_opencl(self, size, fill_value, out);
}

} // namespace at::native
