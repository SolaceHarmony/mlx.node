// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

// Float64 uses double-double representation on GPU (two float32 values)
// This matches the binary layout of complex64_t for copy compatibility
struct float64_t;

template <typename T>
static constexpr constant bool can_convert_to_float64 =
    !is_same_v<T, float64_t> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_float64 =
    !is_same_v<T, float64_t> && is_convertible_v<float, T>;

struct float64_t {
  float hi;  // High-order component
  float lo;  // Low-order component (error term)

  // Constructors
  constexpr float64_t(float h, float l) : hi(h), lo(l) {};
  constexpr float64_t() : hi(0), lo(0) {};
  constexpr float64_t() threadgroup : hi(0), lo(0) {};

  // Conversions to float64_t (from scalar types)
  template <
      typename T,
      typename = typename enable_if<can_convert_to_float64<T>>::type>
  constexpr float64_t(T x) thread : hi(static_cast<float>(x)), lo(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_float64<T>>::type>
  constexpr float64_t(T x) threadgroup : hi(static_cast<float>(x)), lo(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_float64<T>>::type>
  constexpr float64_t(T x) device : hi(static_cast<float>(x)), lo(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_float64<T>>::type>
  constexpr float64_t(T x) constant : hi(static_cast<float>(x)), lo(0) {}

  // Conversions from float64_t (to scalar types - takes high component)
  template <
      typename T,
      typename = typename enable_if<can_convert_from_float64<T>>::type>
  constexpr operator T() const thread {
    return static_cast<T>(hi);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_float64<T>>::type>
  constexpr operator T() const threadgroup {
    return static_cast<T>(hi);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_float64<T>>::type>
  constexpr operator T() const device {
    return static_cast<T>(hi);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_float64<T>>::type>
  constexpr operator T() const constant {
    return static_cast<T>(hi);
  }
};

// Comparison operators (compare hi first, then lo for equality)
constexpr bool operator==(float64_t a, float64_t b) {
  return a.hi == b.hi && a.lo == b.lo;
}

constexpr bool operator!=(float64_t a, float64_t b) {
  return !(a == b);
}

constexpr bool operator<(float64_t a, float64_t b) {
  return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

constexpr bool operator<=(float64_t a, float64_t b) {
  return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo);
}

constexpr bool operator>(float64_t a, float64_t b) {
  return operator<(b, a);
}

constexpr bool operator>=(float64_t a, float64_t b) {
  return operator<=(b, a);
}

// Unary negation
constexpr float64_t operator-(float64_t x) {
  return {-x.hi, -x.lo};
}

// Forward declarations for double-double arithmetic
#include "mlx/backend/metal/kernels/double_double.h"

// Arithmetic operators using double-double arithmetic
inline float64_t operator+(float64_t x, float64_t y) {
  double_double dx = unpack_dd(float2(x.hi, x.lo));
  double_double dy = unpack_dd(float2(y.hi, y.lo));
  double_double result = dd_add(dx, dy);
  float2 packed = pack_dd(result);
  return float64_t(packed.x, packed.y);
}

inline float64_t operator-(float64_t x, float64_t y) {
  double_double dx = unpack_dd(float2(x.hi, x.lo));
  double_double dy = unpack_dd(float2(y.hi, y.lo));
  double_double result = dd_sub(dx, dy);
  float2 packed = pack_dd(result);
  return float64_t(packed.x, packed.y);
}

inline float64_t operator*(float64_t x, float64_t y) {
  double_double dx = unpack_dd(float2(x.hi, x.lo));
  double_double dy = unpack_dd(float2(y.hi, y.lo));
  double_double result = dd_mul(dx, dy);
  float2 packed = pack_dd(result);
  return float64_t(packed.x, packed.y);
}

inline float64_t operator/(float64_t x, float64_t y) {
  double_double dx = unpack_dd(float2(x.hi, x.lo));
  double_double dy = unpack_dd(float2(y.hi, y.lo));
  // Simple division: q = x.hi / y.hi, then refine
  float q = dx.hi / dy.hi;
  auto qy = dd_mul(double_double(q), dy);
  auto r = dd_sub(dx, qy);
  float correction = dd_to_float(r) / dy.hi;
  double_double result = double_double(q + correction);
  float2 packed = pack_dd(result);
  return float64_t(packed.x, packed.y);
}

// isnan for float64_t
inline bool isnan(float64_t x) {
  return metal::isnan(x.hi) || metal::isnan(x.lo);
}
