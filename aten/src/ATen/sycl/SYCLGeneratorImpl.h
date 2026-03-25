// AI-TRANSLATED: SYCL generator for Intel GPU random number generation
#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/DeviceType.h>

// SYCL TODO: needs_review - implement actual SYCL RNG using oneMKL
// #include <oneapi/mkl/rng.hpp>

namespace at {

struct TORCH_API SYCLGeneratorImpl : public c10::GeneratorImpl {
  SYCLGeneratorImpl(DeviceIndex device_index = -1);
  ~SYCLGeneratorImpl() override = default;

  SYCLGeneratorImpl* clone() const {
    // SYCL TODO: needs_review
    return new SYCLGeneratorImpl(device().index());
  }

  void set_current_seed(uint64_t seed) override {
    current_seed_ = seed;
  }

  void set_offset(uint64_t offset) override {
    offset_ = offset;
  }

  uint64_t current_seed() const override {
    return current_seed_;
  }

  uint64_t seed() override {
    // SYCL TODO: needs_review - use proper seed generation
    current_seed_ = c10::detail::getNonDeterministicRandom();
    return current_seed_;
  }

  DeviceType device_type() const override {
    return DeviceType::OPENCL;
  }

private:
  uint64_t current_seed_ = c10::default_rng_seed_val;
  uint64_t offset_ = 0;
};

namespace sycl::detail {

TORCH_API const Generator& getDefaultSYCLGenerator(
    DeviceIndex device_index = -1);

} // namespace sycl::detail

} // namespace at
