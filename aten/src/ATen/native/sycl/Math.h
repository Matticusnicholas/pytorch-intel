// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/Math.cuh
//
// This file contains string-ified math functions originally used by the CUDA
// Jiterator (JIT compilation). In the SYCL backend these are compiled
// ahead-of-time (AOT), so the jiterator_stringify wrappers are not needed.
// The functions themselves are kept for use in AOT-compiled SYCL kernels.
#pragma once

#include <ATen/AccumulateType.h>
#include <c10/macros/Macros.h>
#include <cmath>

namespace at::native {

// SYCL: These functions are compiled AOT rather than JIT-compiled.
// The CUDA version wraps these in jiterator_stringify() for runtime
// compilation; SYCL compiles them at build time.

namespace sycl_math {

template <typename T>
T polevl(const T x, const T A[], const int len) {
  T result = 0;
  for (int i = 0; i < len; ++i) {
    result = result * x + A[i];
  }
  return result;
}

template <typename T>
T ndtri(T y0) {
  constexpr T zero = 0;
  constexpr T one = 1;

  if (y0 == zero) {
    return -std::numeric_limits<T>::infinity();
  }
  if (y0 == one) {
    return std::numeric_limits<T>::infinity();
  }
  if (y0 < zero || y0 > one) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  bool code = true;
  T y = y0;
  if (y > one - T{0.13533528323661269189}) {
    y = one - y;
    code = false;
  }

  if (y > T{0.13533528323661269189}) {
    static const T P0[5] = {
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    };
    static const T Q0[9] = {
      1.00000000000000000000E0,
      1.95448858338141759834E0,
      4.67627912898881538453E0,
      8.63602421390890590575E1,
      -2.25462687854119370527E2,
      2.00260212380060660359E2,
      -8.20372256168333339912E1,
      1.59056225126211695515E1,
      -1.18331621121330003142E0,
    };
    constexpr T s2pi = 2.50662827463100050242E0;

    y = y - T{0.5};
    const T y2 = y * y;
    T x = y + y * (y2 * polevl(y2, P0, int{5}) / polevl(y2, Q0, int{9}));
    return x * s2pi;
  }

  T x = std::sqrt(T{-2.} * std::log(y));
  const T x0 = x - (std::log(x) / x);
  const T z = one / x;
  T x1;

  if (x < T{8.0}) {
    static const T P1[9] = {
      4.05544892305962419923E0, 3.15251094599893866154E1,
      5.71628192246421288162E1, 4.40805073893200834700E1,
      1.46849561928858024014E1, 2.18663306850790267539E0,
      -1.40256079171354495875E-1, -3.50424626827848203418E-2,
      -8.57456785154685413611E-4,
    };
    static const T Q1[9] = {
      1.00000000000000000000E0, 1.57799883256466749731E1,
      4.53907635128879210584E1, 4.13172038254672030440E1,
      1.50425385692907503408E1, 2.50464946208309415979E0,
      -1.42182922854787788574E-1, -3.80806407691578277194E-2,
      -9.33259480895457427372E-4,
    };
    x1 = z * polevl(z, P1, int{9}) / polevl(z, Q1, int{9});
  } else {
    static const T P2[9] = {
      3.23774891776946035970E0, 6.91522889068984211695E0,
      3.93881025292474443415E0, 1.33303460815807542389E0,
      2.01485389549179081538E-1, 1.23716634817820021358E-2,
      3.01581553508235416007E-4, 2.65806974686737550832E-6,
      6.23974539184983293730E-9,
    };
    static const T Q2[9] = {
      1.00000000000000000000E0, 6.02427039364742014255E0,
      3.67983563856160859403E0, 1.37702099489081330271E0,
      2.16236993594496635890E-1, 1.34204006088543189037E-2,
      3.28014464682127739104E-4, 2.89247864745380683936E-6,
      6.79019408009981274425E-9,
    };
    x1 = z * polevl(z, P2, int{9}) / polevl(z, Q2, int{9});
  }

  x = x0 - x1;
  return (!code) ? x : -x;
}

template <typename T>
T log_ndtr(T x) {
  constexpr T SQRT1_2{0.707106781186547524400844362104849039};
  T t = x * SQRT1_2;
  if (x < T{-1.0}) {
    return std::log(std::erfc(-t) / 2) - t * t;
  } else {
    return std::log1p(-std::erfc(t) / 2);
  }
}

template <typename T>
T gcd(const T a_in, const T b_in) {
  T a = std::abs(a_in);
  T b = std::abs(b_in);
  while (a != T{0}) {
    T c = a;
    a = b % a;
    b = c;
  }
  return b;
}

} // namespace sycl_math

} // namespace at::native
