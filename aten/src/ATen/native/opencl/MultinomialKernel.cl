// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/MultinomialKernel.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/EmptyTensor.h>
#include <ATen/sycl/detail/KernelUtils.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/sycl/CUDAGraphsUtils.h>
#include <ATen/native/sycl/BlockReduce.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/SYCLFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/cumsum_cuda_dispatch.h>
#include <ATen/ops/uniform_native.h>
#endif

// SYCL: curand not available, using oneMKL RNG
// #include <curand.h>
#include <oneapi/mkl/rng.hpp>
// SYCL: curand not available, using oneMKL RNG
// #include <curand_kernel.h>
// SYCL: curand not available
// #include <curand_philox4x32_x.h>
#include <type_traits>

namespace at::native {

namespace {

template <
    typename T,
    typename = std::enable_if_t<
        std::is_floating_point_v<T> || std::is_convertible_v<T, float>>>
inline bool _isinf(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return ::isinf(x);
  } else {
    return ::isinf(static_cast<float>(x));
  }
}

#define MAX_NUM_BLOCKS 200

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename scalar_t>
// SYCL: launch bounds
C10_LAUNCH_BOUNDS_1(sycl::detail::SYCL_NUM_THREADS)
/* SYCL kernel */ void renormRowsL1(scalar_t* dist, long rows, long cols) {
  /* SYCL: use sycl::local_accessor for dynamic local memory */ unsigned char my_smem[];
  scalar_t *smem = reinterpret_cast<scalar_t *>(my_smem);
  scalar_t zero = static_cast<scalar_t>(0);
  scalar_t val;
  for (int64_t row = item.get_group(0); row < rows; row += item.get_group_range(0)) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t col = item.get_local_id(0); col < cols; col += item.get_local_range(0)) {
      val = dist[row * cols + col];
      SYCL_KERNEL_ASSERT(!(val < zero)); // ! < 0 for NaN handling
      sum = sum + val;
    }

    sum = sycl_utils::BlockReduceSum(sum, smem);
    if (item.get_local_id(0) == 0) {
      SYCL_KERNEL_ASSERT(!(val < zero)); // ! < 0 for NaN handling
      smem[0] = sum;
    }
    item.barrier(sycl::access::fence_space::local_space);

    sum = smem[0];
    if (sum > zero) {
      for (int64_t col = item.get_local_id(0); col < cols; col += item.get_local_range(0)) {
        dist[row * cols + col] = dist[row * cols + col] / sum;
      }
    }
  }
}

void renormRows(Tensor& t) {
  TORCH_CHECK(t.dim() == 2);
  int64_t rows = t.size(0);
  int64_t cols = t.size(1);

  auto props = at::sycl::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr);
  int numSM = props->multiProcessorCount;
  const int64_t maxThreads = std::min(
      props->maxThreadsPerBlock, sycl_utils::kCUDABlockReduceMaxThreads());

  int warp_size = at::sycl::sub_group_size();
  sycl::range<3> grid(rows < numSM * 4 ? rows : numSM * 4);
  sycl::range<3> block(std::min(maxThreads, warp_size * ceil_div(cols, int64_t{warp_size})));

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, t.scalar_type(), "renormRows_opencl", [&] {
    renormRowsL1<scalar_t>
        /* SYCL: kernel launch with nd_range(grid, block, (block.x / warp_size) * sizeof(scalar_t),
        at::sycl::getCurrentSYCLStream()) */(t.mutable_data_ptr<scalar_t>(),
            rows, cols);
    // SYCL: kernel launch check handled by SYCL runtime;
  });
}

template <typename scalar_t>
int binarySearchForMultinomial(const scalar_t* cumdist,
                                          const scalar_t* dist,
                                          int size,
                                          scalar_t val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  SYCL_KERNEL_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    scalar_t midVal = cumdist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  while(start >= 1 && dist[start] == 0) start--;

  return start;
}

template <typename scalar_t>
/* SYCL kernel */ void
sampleMultinomialWithReplacement(PhiloxSYCLState philox_args,
                                 int totalSamples,
                                 int64_t* dest,
                                 int64_t distributions,
                                 int categories,
                                 const scalar_t* normDistPrefixSum,
                                 const scalar_t* normDist) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  auto seeds = at::sycl::philox::unpack(philox_args);

  // global index formula for 2D grid of 1D blocks
  int idx = item.get_group(1) * item.get_group_range(0) * item.get_local_range(0) + item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  // SYCL: use oneMKL RNG engine
    sycl_philox_state_t state;
  // SYCL: initialize RNG
    sycl_rng_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = item.get_group(1);
       curDist < distributions;
       curDist += item.get_group_range(1)) {
    for (int sample = item.get_group(0)*item.get_local_range(0) + item.get_local_id(0);
         sample < totalSamples; sample += item.get_local_range(0)*item.get_group_range(0)) {

      //we are losing 3 out of 4 generated numbers but it's ok
      //this kernel is not very efficient anyway
      auto rand = sycl_uniform4(&state);
      scalar_t r = static_cast<scalar_t>(rand.x);

      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          normDist + curDist * categories,
          categories,
          r);

      dest[curDist * totalSamples + sample] = choice;

    }
  }
}

template <typename scalar_t, typename accscalar_t>
// SYCL: launch bounds
C10_LAUNCH_BOUNDS_1(sycl::detail::SYCL_NUM_THREADS)
/* SYCL kernel */ void sampleMultinomialOnce(
    int64_t* dest,
    int64_t distributions,
    int categories,
    const scalar_t* sampled,
    const scalar_t* dist,
    int stride_dist, // dist->stride(0)
    int stride_categories // dist->stride(1)
) {
  /* SYCL: use sycl::local_accessor for dynamic local memory */ unsigned char my_smem[];
  /* SYCL: use sycl::local_accessor */ bool found;
  /* SYCL: use sycl::local_accessor */ unsigned foundPos;

  accscalar_t *smem = reinterpret_cast<accscalar_t *>(my_smem);

  accscalar_t accZero = static_cast<accscalar_t>(0);
  scalar_t zero = static_cast<scalar_t>(0);

  for (int64_t curDist = item.get_group(0);
       curDist < distributions; curDist += item.get_group_range(0)) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    accscalar_t sum = accZero;
    scalar_t val;
    for (int cat = item.get_local_id(0); cat < categories; cat += item.get_local_range(0)) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      SYCL_KERNEL_ASSERT(!at::_isnan(val));
      SYCL_KERNEL_ASSERT(!_isinf(val));
      SYCL_KERNEL_ASSERT(!(val < zero));
      sum = sum + static_cast<accscalar_t>(val);
    }

    // item.get_local_id(0) == 0 has the sum value from this
    sum = sycl_utils::BlockReduceSum(sum, smem);

    // Broadcast sum and sample value
    if (item.get_local_id(0) == 0) {
      // Make sure the sum of our distribution didn't overflow
      SYCL_KERNEL_ASSERT(!_isinf(val));
      SYCL_KERNEL_ASSERT(sum > accZero);

      foundPos = 0;
      smem[0] = sum;
      smem[1] = sampled[curDist];
    }
    item.barrier(sycl::access::fence_space::local_space);

    sum = smem[0];
    scalar_t sample = static_cast<scalar_t>(smem[1]);
    item.barrier(sycl::access::fence_space::local_space);

    if (sum == accZero) {
      // Choose the first element
      if (item.get_local_id(0) == 0) {
        dest[curDist] = 0;
      }

      continue;
    }

    int chunks = (categories + (int)item.get_local_range(0) - 1) / item.get_local_range(0);
    accscalar_t prevHighProb = accZero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * item.get_local_range(0) + item.get_local_id(0);

      accscalar_t dist_val = cat < categories ?
                             static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
                             accZero;

      smem[item.get_local_id(0)] = dist_val;
      item.barrier(sycl::access::fence_space::local_space);

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < item.get_local_range(0); offset *= 2) {
        accscalar_t val = accZero;

        if (item.get_local_id(0) >= offset) {
          val = smem[item.get_local_id(0) - offset] + smem[item.get_local_id(0)];
        }

        item.barrier(sycl::access::fence_space::local_space);
        if (item.get_local_id(0) >= offset) {
          smem[item.get_local_id(0)] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      scalar_t curBucket =
          static_cast<scalar_t>(smem[item.get_local_id(0)] + prevHighProb);
      scalar_t prevBucket = static_cast<scalar_t>(
          item.get_local_id(0) == 0 ? prevHighProb
                          : smem[item.get_local_id(0) - 1] + prevHighProb);
      bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
          (sample >= prevBucket) &&
          (dist_val > zero));

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        sycl_atomic_max(&foundPos, cat);
        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = prevHighProb + smem[item.get_local_range(0) - 1];

      item.barrier(sycl::access::fence_space::local_space);
    }

    if (item.get_local_id(0) == 0) {
      if (found) {
          dest[curDist] = foundPos;
      } else {
        // This should address a rare bug where we don't select a valid index. This likely occurs when
        // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
        // and our uniform sample is greater than this value. In this case we likely have uninitialized memory
        // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
        // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
        // rarity in which this occurs, this should not be an issue.
        for (int cat = categories - 1; cat >= 0; --cat) {
          if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
            dest[curDist] = cat;
            break;
          }
        }
      }
    }
  }
}

void multinomial_with_replacement_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<SYCLGeneratorImpl>(generator, cuda::detail::getDefaultSYCLGenerator());

  int inputSize = self.dim();
  int64_t numDist =
      inputSize == 1 ? 1 : self.size(0);
  int numCategories =
      inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;

  result.resize_({numDist, n_sample});

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self_v.scalar_type(), "multinomial_kernel_opencl", [&] {
    using accscalar_t = at::acc_type<scalar_t, /*is_sycl*/true>;
    auto props = at::sycl::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr);
    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;
    int maxShared = props->sharedMemPerBlock;

    int warp_size = at::sycl::sub_group_size();
    int requiredWarps = at::ceil_div(numCategories, warp_size);
    int requiredThreads = std::min(maxThreads, requiredWarps * warp_size);
    int requiredShared = requiredThreads * sizeof(accscalar_t);

    if (n_sample == 1 && maxShared >= requiredShared) {
      // Optimized allocation-free implementation
      // To exploit greater parallelism for the sampling, generate the
      // Uniform random samples in a separate kernel launch, into
      // temporarily allocated memory. The device RNG is thread-limited
      Tensor sampled = at::detail::empty_sycl({numDist, n_sample}, self_v.options());
      at::native::uniform_(sampled, 0.0, 1.0, generator);

      sycl::range<3> block(requiredThreads);
      sycl::range<3> grid(std::min(static_cast<int>(numDist), numSM * 4));

      sampleMultinomialOnce<scalar_t, accscalar_t>
          /* SYCL: kernel launch with nd_range(grid, block,
          requiredShared,
          at::sycl::getCurrentSYCLStream()) */(
              result.mutable_data_ptr<int64_t>(),
                  numDist,
                  numCategories,
                  sampled.const_data_ptr<scalar_t>(),
                  self_v.const_data_ptr<scalar_t>(),
                  self_v.stride(0),
                  self_v.stride(1)
          );
      // SYCL: kernel launch check handled by SYCL runtime;
    } else {
      // Generic, slow implementation with memory allocations

      // For sampling without replacement, we modify the distribution
      // for subsequent samples in this space
      Tensor origDist = native::empty_like(
          self_v,
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      origDist.copy_(self_v);

      Tensor normDist = native::empty_like(
          self_v,
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      Tensor prefixSum = native::empty_like(
          self_v,
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // Renorm along rows
      normDist.copy_(origDist);
      renormRows(normDist);

      // Prefix sum along rows
      at::sycl::cumsum_out(prefixSum, normDist, 1);

      PhiloxSYCLState rng_engine_inputs;

        // Binary search is warp divergent (so effectively we're running
        // with just a single thread), but for better utilization,
        // we need each block to have at least 4 warps.
        sycl::range<3> block(128);

        // Each block will generate a sample from one
        // distribution concurrently.
        int grid_y=std::min<int>(numDist, at::sycl::getCurrentDeviceProperties()->maxGridSize[1]);
        sycl::range<3> grid((n_sample-1)/block.x+1, grid_y);
        {
          // See Note [Acquire lock when using random generators]
          std::lock_guard<std::mutex> lock(gen->mutex_);

          // each thread generates a single sample for (numdist/numblocks.y) distributions, however, since we have to use
          // sycl_uniform4 (See Note [Register spilling in curand call for CUDA < 10]),
          // offset is 4 times that.
          auto offset = ((numDist-1)/grid.y+1)*4;
          rng_engine_inputs = gen->philox_cuda_state(offset);
        }
        // Sample with replacement

        sampleMultinomialWithReplacement
            /* SYCL: kernel launch with nd_range(grid, block, 0, at::sycl::getCurrentSYCLStream()) */(
                rng_engine_inputs,
                n_sample,
                result.mutable_data_ptr<int64_t>(),
                numDist, numCategories,
                prefixSum.const_data_ptr<scalar_t>(),
                normDist.const_data_ptr<scalar_t>());
        // SYCL: kernel launch check handled by SYCL runtime;
    }
  });

  if (inputSize == 1) {
    result.resize_({n_sample});
  }
}
}

REGISTER_DISPATCH(
    multinomial_with_replacement_stub,
    &multinomial_with_replacement_kernel_impl);
} // namespace at::native
