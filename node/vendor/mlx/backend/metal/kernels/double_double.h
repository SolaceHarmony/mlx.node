// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.
//
// Extended Precision Arithmetic for MLX Metal Kernels
//
// Implements double-double (DD) arithmetic using two float32 values
// to achieve ~30-32 digits of precision (vs 7-8 for float32).
//
// Based on:
// - Dekker (1971): A floating-point technique for extending the available precision
// - Knuth TAOCP Vol 2: Seminumerical Algorithms
// - Shewchuk (1997): Adaptive Precision Floating-Point Arithmetic

#pragma once

#include <metal_stdlib>

// Note: This header is used in Metal shaders, so types are in global namespace

// ============================================================================
// Double-Double Representation
// ============================================================================

struct double_double {
    float hi;  // High-order term (standard float32)
    float lo;  // Low-order correction term (residual error)

    // Constructors
    constexpr double_double() : hi(0.0f), lo(0.0f) {}
    constexpr double_double(float h, float l) : hi(h), lo(l) {}
    explicit constexpr double_double(float x) : hi(x), lo(0.0f) {}
};

struct complex_dd {
    double_double re;  // Real part (2 floats)
    double_double im;  // Imaginary part (2 floats)

    constexpr complex_dd() : re(), im() {}
    constexpr complex_dd(double_double r, double_double i) : re(r), im(i) {}
    explicit constexpr complex_dd(metal::float2 z)
        : re(double_double(z.x)), im(double_double(z.y)) {}
};

// ============================================================================
// Error-Free Transformations
// ============================================================================

// Two-Sum: Exact sum with error term (Knuth)
// Returns (s, e) where s = round(a + b) and e = (a + b) - s
inline double_double quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return double_double(s, e);
}

// Two-Sum (general case, no ordering assumption)
inline double_double two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return double_double(s, e);
}

// Two-Product: Exact product with error term
// Uses FMA for maximum accuracy
inline double_double two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);  // Error: a*b - round(a*b)
    return double_double(p, e);
}

// ============================================================================
// Double-Double Arithmetic
// ============================================================================

// Addition: dd + dd
inline double_double dd_add(double_double a, double_double b) {
    double_double s = two_sum(a.hi, b.hi);
    double_double t = two_sum(a.lo, b.lo);

    // Normalize: collect all error terms
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    s = quick_two_sum(s.hi, s.lo);

    return s;
}

// Addition: dd + float
inline double_double dd_add(double_double a, float b) {
    double_double s = two_sum(a.hi, b);
    s.lo += a.lo;
    s = quick_two_sum(s.hi, s.lo);
    return s;
}

// Subtraction: dd - dd
inline double_double dd_sub(double_double a, double_double b) {
    return dd_add(a, double_double(-b.hi, -b.lo));
}

// Multiplication: dd * dd
inline double_double dd_mul(double_double a, double_double b) {
    double_double p = two_prod(a.hi, b.hi);

    // Add cross terms: a.hi*b.lo + a.lo*b.hi
    p.lo += a.hi * b.lo + a.lo * b.hi;
    p = quick_two_sum(p.hi, p.lo);

    return p;
}

// Multiplication: dd * float
inline double_double dd_mul(double_double a, float b) {
    double_double p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    p = quick_two_sum(p.hi, p.lo);
    return p;
}

// Division: dd / float (common case for normalization)
inline double_double dd_div(double_double a, float b) {
    float q = a.hi / b;
    double_double p = two_prod(q, b);
    float e = (a.hi - p.hi - p.lo + a.lo) / b;
    return quick_two_sum(q, e);
}

// Negation
inline double_double dd_neg(double_double a) {
    return double_double(-a.hi, -a.lo);
}

// ============================================================================
// Conversion and Rounding
// ============================================================================

// Round double-double to single float (THIS IS WHERE PRECISION IS LOST)
inline float dd_to_float(double_double a) {
    return a.hi + a.lo;  // Final rounding happens here
}

// Lift float to DD
inline double_double float_to_dd(float x) {
    return double_double(x, 0.0f);
}

// ============================================================================
// Memory Layout Helpers (for MLX integration)
// ============================================================================

// Pack DD into float2 for storage
inline float2 pack_dd(double_double a) {
    return float2(a.hi, a.lo);
}

// Unpack float2 into DD
inline double_double unpack_dd(float2 v) {
    return double_double(v.x, v.y);
}
