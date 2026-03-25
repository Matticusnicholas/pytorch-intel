// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/SortingCommon.cuh
#pragma once
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>

namespace at::native {

constexpr int SYCL_SORT_MAX_BLOCK_SIZE = 1024;
constexpr int64_t SYCL_SORT_MAX_GRID_SIZE = 65535LL;

inline bool getGridFromTiles(int64_t gridTiles, sycl::range<3>& grid) {
  if (gridTiles > SYCL_SORT_MAX_GRID_SIZE * SYCL_SORT_MAX_GRID_SIZE * SYCL_SORT_MAX_GRID_SIZE) {
    return false;
  }

  int64_t gridX = gridTiles > SYCL_SORT_MAX_GRID_SIZE ? SYCL_SORT_MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (gridTiles > SYCL_SORT_MAX_GRID_SIZE) {
    gridTiles = ceil_div(gridTiles, SYCL_SORT_MAX_GRID_SIZE);
    gridY = gridTiles > SYCL_SORT_MAX_GRID_SIZE ? SYCL_SORT_MAX_GRID_SIZE : gridTiles;

    if (gridTiles > SYCL_SORT_MAX_GRID_SIZE) {
      gridTiles = ceil_div(gridTiles, SYCL_SORT_MAX_GRID_SIZE);
      gridZ = gridTiles > SYCL_SORT_MAX_GRID_SIZE ? SYCL_SORT_MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = sycl::range<3>(gridZ, gridY, gridX);
  return true;
}

template <typename scalar_t, bool handleNaN = false>
struct GTOp {
  bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && at::_isnan(lhs) && !at::_isnan(rhs)) ||
        (static_cast<scalar_t>(lhs) > static_cast<scalar_t>(rhs));
  }
};

template <typename scalar_t, bool handleNaN = false>
struct LTOp {
  bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && at::_isnan(rhs) && !at::_isnan(lhs)) ||
        (static_cast<scalar_t>(lhs) < static_cast<scalar_t>(rhs));
  }
};

} // namespace at::native
