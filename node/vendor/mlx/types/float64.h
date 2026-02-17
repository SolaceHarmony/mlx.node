// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cmath>

namespace mlx::core {

// Float64 CPU representation using double-double arithmetic
// Binary layout matches Metal's float2 for seamless GPU transfers
struct float64_t {
  float hi;  // High-order component
  float lo;  // Low-order component (error term)

  // Default constructor
  constexpr float64_t() : hi(0.0f), lo(0.0f) {}

  // Construct from components
  constexpr float64_t(float h, float l) : hi(h), lo(l) {}

  // Construct from native double (conversion to double-double)
  float64_t(double d) {
    hi = static_cast<float>(d);
    double err = d - static_cast<double>(hi);
    lo = static_cast<float>(err);
  }

  // Convert to native double (for Python export and calculations)
  operator double() const {
    return static_cast<double>(hi) + static_cast<double>(lo);
  }

  // Construct from single float
  constexpr float64_t(float f) : hi(f), lo(0.0f) {}

  // Comparison operators
  bool operator==(const float64_t& other) const {
    return hi == other.hi && lo == other.lo;
  }

  bool operator!=(const float64_t& other) const {
    return !(*this == other);
  }

  bool operator<(const float64_t& other) const {
    return (hi < other.hi) || (hi == other.hi && lo < other.lo);
  }

  bool operator<=(const float64_t& other) const {
    return (hi < other.hi) || (hi == other.hi && lo <= other.lo);
  }

  bool operator>(const float64_t& other) const {
    return other < *this;
  }

  bool operator>=(const float64_t& other) const {
    return other <= *this;
  }
};

// Unary negation
inline float64_t operator-(const float64_t& x) {
  return float64_t(-x.hi, -x.lo);
}

} // namespace mlx::core
