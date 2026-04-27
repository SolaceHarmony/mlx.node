#include <algorithm>
#include <iostream>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <napi.h>

#include "addon_data.h"
#include "mlx/array.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/dtype_utils.h"
#include "mlx/export.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/linalg.h"
#include "mlx/random.h"
#include "mlx/fast.h"
#include "mlx/fft.h"
#include "mlx/memory.h"
#include "mlx/device.h"
#include "mlx/einsum.h"
#include "mlx/graph_utils.h"
#include "mlx_bridge.h"

#include "dtype.h"
#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"
#include "runtime.h"
#include "stream.h"

namespace {

Napi::Object WrapArray(Napi::Env env, std::shared_ptr<mlx::core::array> tensor);

// Forward declarations for helpers used by factories
bool IsDtypeArg(
    Napi::Env env,
    const Napi::Value& value,
    mlx::node::AddonData& data);
mlx::core::Dtype MaybeParseDtype(
    Napi::Env env,
    const Napi::Value& value,
    mlx::core::Dtype fallback,
    mlx::node::AddonData& data);

// Forward declaration: parse nested JS numeric arrays into flat data + dims
static bool ParseNestedNumberArray(
    Napi::Env env,
    const Napi::Value& v,
    std::vector<double>& data,
    std::vector<int>& dims);

const std::unordered_map<std::string, mlx::core::Dtype>& DtypeLookup() {
  static const std::unordered_map<std::string, mlx::core::Dtype> mapping = {
      {"bool", mlx::core::bool_},
      {"int8", mlx::core::int8},
      {"int16", mlx::core::int16},
      {"int32", mlx::core::int32},
      {"int64", mlx::core::int64},
      {"uint8", mlx::core::uint8},
      {"uint16", mlx::core::uint16},
      {"uint32", mlx::core::uint32},
      {"uint64", mlx::core::uint64},
      {"float16", mlx::core::float16},
      {"bfloat16", mlx::core::bfloat16},
      {"float32", mlx::core::float32},
      {"float64", mlx::core::float64},
      {"complex64", mlx::core::complex64},
  };
  return mapping;
}

std::string DtypeToString(mlx::core::Dtype dtype) {
  return std::string(mlx::core::dtype_to_string(dtype));
}

mlx::core::Dtype ParseDtypeKey(Napi::Env env, const std::string& key) {
  const auto& mapping = DtypeLookup();
  auto it = mapping.find(key);
  if (it == mapping.end()) {
    Napi::TypeError::New(env, "Unsupported dtype: " + key)
        .ThrowAsJavaScriptException();
    return mlx::core::float32;
  }
  return it->second;
}

class ArrayWrapper : public Napi::ObjectWrap<ArrayWrapper> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    auto& data = mlx::node::GetAddonData(env);
    if (!data.array_constructor.IsEmpty()) {
      exports.Set("Array", data.array_constructor.Value());
      return;
    }

    Napi::Function func = DefineClass(
        env,
        "Array",
        {
            InstanceMethod("shape", &ArrayWrapper::Shape),
            InstanceMethod("dtype", &ArrayWrapper::Dtype),
            InstanceMethod("eval", &ArrayWrapper::Eval),
            InstanceMethod("toFloat32Array", &ArrayWrapper::ToFloat32Array),
            InstanceMethod("toTypedArray", &ArrayWrapper::ToTypedArray),
            StaticMethod(
                "fromFloat32Array",
                &ArrayWrapper::FromFloat32Array,
                napi_default,
                nullptr),
            StaticMethod(
                "fromTypedArray",
                &ArrayWrapper::FromTypedArray,
                napi_default,
                nullptr),
        });

    data.array_constructor = Napi::Persistent(func);
    exports.Set("Array", func);
  }

  explicit ArrayWrapper(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<ArrayWrapper>(info) {
    auto env = info.Env();
    if (info.Length() != 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "mlx.core.Array cannot be constructed directly")
          .ThrowAsJavaScriptException();
      return;
    }

    auto external =
        info[0].As<Napi::External<std::shared_ptr<mlx::bridge::ArrayHolder>>>();
    holder_ = *external.Data();
  }

  [[nodiscard]] const mlx::core::array& tensor() const {
    return holder_->array;
  }

 private:
  using TypedArray = Napi::TypedArray;

 public:
  // Public wrapper so factory code can reuse the conversion logic.
  static std::optional<mlx::core::array> BuildFromTyped(
      Napi::Env env,
      const Napi::TypedArray& typed,
      const Napi::Array& shapeArray,
      std::optional<mlx::core::Dtype> requestedDtype);

  static mlx::core::Dtype ParseDtype(Napi::Env env, const std::string& key) {
    return ParseDtypeKey(env, key);
  }

  static std::string DtypeToString(mlx::core::Dtype dtype) {
    return ::DtypeToString(dtype);
  }

  static std::optional<mlx::core::Dtype> InferDtypeFromTypedArray(
      Napi::Env env,
      const TypedArray& typed) {
    switch (typed.TypedArrayType()) {
      case napi_int8_array:
        return mlx::core::int8;
      case napi_uint8_array:
      case napi_uint8_clamped_array:
        return mlx::core::uint8;
      case napi_int16_array:
        return mlx::core::int16;
      case napi_uint16_array:
        return mlx::core::uint16;
      case napi_int32_array:
        return mlx::core::int32;
      case napi_uint32_array:
        return mlx::core::uint32;
      case napi_float32_array:
        return mlx::core::float32;
      case napi_float64_array:
        return mlx::core::float64;
      case napi_bigint64_array:
        return mlx::core::int64;
      case napi_biguint64_array:
        return mlx::core::uint64;
      default:
        Napi::TypeError::New(env, "Unsupported typed array input")
            .ThrowAsJavaScriptException();
        return std::nullopt;
    }
  }

  static std::vector<int32_t> ExtractShape(
      Napi::Env env,
      const Napi::Array& shapeArray,
      size_t& elementCount) {
    std::vector<int32_t> shapeValues;
    shapeValues.reserve(shapeArray.Length());
    elementCount = 1;
    for (uint32_t i = 0; i < shapeArray.Length(); ++i) {
      auto value = shapeArray.Get(i);
      if (!value.IsNumber()) {
        Napi::TypeError::New(env, "Shape dimensions must be numbers")
            .ThrowAsJavaScriptException();
        return {};
      }
      auto dim = static_cast<int32_t>(value.As<Napi::Number>().Int64Value());
      if (dim < 0) {
        Napi::TypeError::New(env, "Shape dimensions must be non-negative")
            .ThrowAsJavaScriptException();
        return {};
      }
      shapeValues.push_back(dim);
      elementCount *= dim;
    }
    return shapeValues;
  }

  // NOTE: CopyTypedArray is intentionally not provided.  All typed-array paths
  // now go through MakeArrayFromTyped which writes directly from the V8
  // ArrayBuffer backing-store into mlx::core::allocator::malloc memory,
  // avoiding the intermediate std::vector<T>.

  static std::optional<mlx::core::array> MakeArrayFromTyped(
      Napi::Env env,
      const TypedArray& typed,
      const Napi::Array& shapeArray,
      std::optional<mlx::core::Dtype> requestedDtype) {
    size_t elementCount = 1;
    auto shapeValues = ExtractShape(env, shapeArray, elementCount);
    if (env.IsExceptionPending()) {
      return {};
    }

    const size_t typedLength = typed.ElementLength();

    auto dtype = requestedDtype.value_or(
        InferDtypeFromTypedArray(env, typed).value_or(mlx::core::float32));
    if (env.IsExceptionPending()) {
      return {};
    }

    switch (dtype) {
      case mlx::core::float32:
        if (typed.TypedArrayType() != napi_float32_array) {
          Napi::TypeError::New(env, "float32 dtype requires Float32Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::float64:
        if (typed.TypedArrayType() != napi_float64_array) {
          Napi::TypeError::New(env, "float64 dtype requires Float64Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::int8:
        if (typed.TypedArrayType() != napi_int8_array) {
          Napi::TypeError::New(env, "int8 dtype requires Int8Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::uint8:
      case mlx::core::bool_: {
        auto type = typed.TypedArrayType();
        if (type != napi_uint8_array && type != napi_uint8_clamped_array &&
            !(dtype == mlx::core::bool_ && type == napi_int8_array)) {
          Napi::TypeError::New(
              env,
              std::string(DtypeToString(dtype)) +
                  " dtype requires Uint8Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      }
      case mlx::core::int16:
        if (typed.TypedArrayType() != napi_int16_array) {
          Napi::TypeError::New(env, "int16 dtype requires Int16Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::uint16:
      case mlx::core::float16:
      case mlx::core::bfloat16:
        if (typed.TypedArrayType() != napi_uint16_array) {
          Napi::TypeError::New(env, "uint16-based dtype requires Uint16Array")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::int32:
        if (typed.TypedArrayType() != napi_int32_array) {
          Napi::TypeError::New(env, "int32 dtype requires Int32Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::uint32:
        if (typed.TypedArrayType() != napi_uint32_array) {
          Napi::TypeError::New(env, "uint32 dtype requires Uint32Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::int64:
        if (typed.TypedArrayType() != napi_bigint64_array) {
          Napi::TypeError::New(env, "int64 dtype requires BigInt64Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::uint64:
        if (typed.TypedArrayType() != napi_biguint64_array) {
          Napi::TypeError::New(
              env, "uint64 dtype requires BigUint64Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::complex64:
        if (typed.TypedArrayType() != napi_float32_array) {
          Napi::TypeError::New(
              env, "complex64 dtype requires Float32Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        if (elementCount * 2 != typedLength) {
          Napi::RangeError::New(
              env, "complex64 expects data length equal to element count * 2")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      default:
        break;
    }

    if (dtype != mlx::core::complex64 && elementCount != typedLength) {
      Napi::RangeError::New(env, "Shape does not match data length")
          .ThrowAsJavaScriptException();
      return {};
    }

    const uint8_t* rawData =
        static_cast<const uint8_t*>(typed.ArrayBuffer().Data()) +
        typed.ByteOffset();

    mlx::core::Shape shape;
    shape.reserve(shapeValues.size());
    for (auto dim : shapeValues) {
      shape.push_back(dim);
    }

    // Direct path: one mlx::core::allocator::malloc + one memcpy from the V8
    // ArrayBuffer backing-store.  No intermediate std::vector<T>.
    switch (dtype) {
      case mlx::core::float32: {
        const size_t nbytes = elementCount * sizeof(float);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::float64: {
        // float64 uses a Dekker-split internal representation (float64_t != double).
        // A conversion loop is unavoidable; we still malloc once.
        const size_t n = elementCount;
        const size_t nbytes = n * sizeof(mlx::core::float64_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        auto* dst = static_cast<mlx::core::float64_t*>(buf.raw_ptr());
        const double* src = reinterpret_cast<const double*>(rawData);
        for (size_t i = 0; i < n; ++i) {
          dst[i] = mlx::core::float64_t(src[i]);
        }
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::int8: {
        const size_t nbytes = elementCount * sizeof(int8_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::uint8: {
        const size_t nbytes = elementCount * sizeof(uint8_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::bool_: {
        // bool in MLX is stored as uint8 (0/1).  The incoming Uint8Array already
        // contains 0 or non-zero — normalise to strict 0/1.
        const size_t nbytes = elementCount * sizeof(uint8_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        auto* dst = static_cast<uint8_t*>(buf.raw_ptr());
        const uint8_t* src = reinterpret_cast<const uint8_t*>(rawData);
        for (size_t i = 0; i < elementCount; ++i) {
          dst[i] = src[i] ? 1u : 0u;
        }
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::int16: {
        const size_t nbytes = elementCount * sizeof(int16_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::uint16: {
        const size_t nbytes = elementCount * sizeof(uint16_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::float16: {
        // float16_t has the same wire format as uint16; straight memcpy.
        const size_t nbytes = elementCount * sizeof(uint16_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::bfloat16: {
        const size_t nbytes = elementCount * sizeof(uint16_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::int32: {
        const size_t nbytes = elementCount * sizeof(int32_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::uint32: {
        const size_t nbytes = elementCount * sizeof(uint32_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::int64: {
        const size_t nbytes = elementCount * sizeof(int64_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::uint64: {
        const size_t nbytes = elementCount * sizeof(uint64_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      case mlx::core::complex64: {
        // complex64 is stored as interleaved (re, im) float pairs; the
        // incoming Float32Array has the same layout — straight memcpy.
        const size_t nbytes = elementCount * 2 * sizeof(float);
        auto buf = mlx::core::allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), rawData, nbytes);
        return mlx::core::array(buf, shape, dtype);
      }
      default:
        Napi::TypeError::New(env, "Unsupported dtype conversion")
            .ThrowAsJavaScriptException();
        return {};
    }
  }

  static Napi::Value FromFloat32Array(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env, "Expected Float32Array and shape array")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto typed = info[0].As<Napi::TypedArray>();
    if (typed.TypedArrayType() != napi_float32_array) {
      Napi::TypeError::New(env, "Only Float32Array inputs are supported")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto tensor = MakeArrayFromTyped(
        env,
        typed.As<Napi::Float32Array>(),
        info[1].As<Napi::Array>(),
        mlx::core::float32);
    if (!tensor) {
      return env.Null();
    }
    return WrapArray(
        env, std::make_shared<mlx::core::array>(std::move(*tensor)));
  }

  static Napi::Value FromTypedArray(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env, "Expected TypedArray and shape array")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    std::optional<mlx::core::Dtype> dtype;
    if (info.Length() >= 3) {
      auto& addon = mlx::node::GetAddonData(env);
      if (IsDtypeArg(env, info[2], addon)) {
        dtype = MaybeParseDtype(env, info[2], mlx::core::float32, addon);
      }
      if (env.IsExceptionPending()) {
        return env.Null();
      }
    }

    auto tensor = MakeArrayFromTyped(
        env, info[0].As<Napi::TypedArray>(), info[1].As<Napi::Array>(), dtype);
    if (!tensor) {
      return env.Null();
    }
    return WrapArray(
        env, std::make_shared<mlx::core::array>(std::move(*tensor)));
  }

  Napi::Value Shape(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    const auto& shape = tensor().shape();
    auto jsShape = Napi::Array::New(env, shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      jsShape.Set(i, Napi::Number::New(env, shape[i]));
    }
    return jsShape;
  }

  Napi::Value Dtype(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    // Return a Dtype object (e.g., mlx.float32) to match Python parity
    auto& addon = mlx::node::GetAddonData(env);
    auto dt = tensor().dtype();
    // Construct via the recorded dtype constructor, mirroring
    // DtypeWrapper::Create
    auto ext = Napi::External<mlx::core::Dtype>::New(
        env,
        new mlx::core::Dtype(dt),
        [](Napi::Env /*env*/, mlx::core::Dtype* ptr) { delete ptr; });
    return addon.dtype_constructor.New({ext});
  }

  Napi::Value Eval(const Napi::CallbackInfo& info) {
    holder_->array.eval();
    return info.Env().Undefined();
  }

  Napi::Value ToFloat32Array(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    // Flatten to ensure contiguous row-major layout.  eval() flushes the lazy
    // graph so data<float>() is valid.
    auto flat_arr = std::make_shared<mlx::core::array>(
        mlx::core::flatten(tensor()));
    flat_arr->eval();

    const size_t length = flat_arr->size();
    const size_t nbytes = length * sizeof(float);

    // Zero-copy: point the V8 ArrayBuffer directly at the MLX malloc buffer.
    // The shared_ptr keeps the mlx::core::array alive until V8 GCs the buffer.
    float* raw = flat_arr->data<float>();
    auto* hint = new std::shared_ptr<mlx::core::array>(flat_arr);
    Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
        env,
        raw,
        nbytes,
        [](Napi::Env /*env*/, void* /*data*/, void* h) {
          delete static_cast<std::shared_ptr<mlx::core::array>*>(h);
        },
        hint);
    return Napi::Float32Array::New(env, length, buffer, 0);
  }

  Napi::Value ToTypedArray(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    // Flatten to ensure contiguous row-major layout.
    auto arr = std::make_shared<mlx::core::array>(
        mlx::core::flatten(tensor()));
    arr->eval();

    const auto dtype = arr->dtype();
    const size_t length = arr->size();

    // Helper: create an external zero-copy ArrayBuffer backed by the MLX
    // malloc buffer and keep arr alive via shared_ptr until V8 GCs it.
    // For dtypes where the wire format matches the JS TypedArray element type
    // exactly, no copy is needed.
    auto make_external = [&](void* raw,
                             size_t nbytes) -> Napi::ArrayBuffer {
      auto* hint = new std::shared_ptr<mlx::core::array>(arr);
      return Napi::ArrayBuffer::New(
          env,
          raw,
          nbytes,
          [](Napi::Env /*env*/, void* /*data*/, void* h) {
            delete static_cast<std::shared_ptr<mlx::core::array>*>(h);
          },
          hint);
    };

    switch (dtype) {
      case mlx::core::float32: {
        auto buf = make_external(arr->data<float>(), length * sizeof(float));
        return Napi::Float32Array::New(env, length, buf, 0);
      }
      case mlx::core::float64: {
        // float64_t (Dekker split) is not a native JS type; we must convert
        // to double — one copy is unavoidable here.
        const size_t nbytes = length * sizeof(double);
        Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, nbytes);
        auto* dst = static_cast<double*>(buffer.Data());
        const auto* src = arr->data<mlx::core::float64_t>();
        for (size_t i = 0; i < length; ++i) {
          dst[i] = static_cast<double>(src[i]);
        }
        return Napi::Float64Array::New(env, length, buffer, 0);
      }
      case mlx::core::int8: {
        auto buf = make_external(arr->data<int8_t>(), length * sizeof(int8_t));
        return Napi::Int8Array::New(env, length, buf, 0);
      }
      case mlx::core::uint8: {
        auto buf = make_external(arr->data<uint8_t>(), length * sizeof(uint8_t));
        return Napi::Uint8Array::New(env, length, buf, 0);
      }
      case mlx::core::bool_: {
        // bool in MLX is stored as uint8 with values 0 or 1; the external
        // Uint8Array view is valid directly.
        auto buf = make_external(
            arr->data<uint8_t>(), length * sizeof(uint8_t));
        return Napi::Uint8Array::New(env, length, buf, 0);
      }
      case mlx::core::int16: {
        auto buf = make_external(arr->data<int16_t>(), length * sizeof(int16_t));
        return Napi::Int16Array::New(env, length, buf, 0);
      }
      case mlx::core::uint16: {
        auto buf = make_external(arr->data<uint16_t>(), length * sizeof(uint16_t));
        return Napi::Uint16Array::New(env, length, buf, 0);
      }
      case mlx::core::float16: {
        // float16_t has the same bit layout as uint16; expose as Uint16Array.
        auto buf = make_external(
            arr->data<mlx::core::float16_t>(), length * sizeof(uint16_t));
        return Napi::Uint16Array::New(env, length, buf, 0);
      }
      case mlx::core::bfloat16: {
        auto buf = make_external(
            arr->data<mlx::core::bfloat16_t>(), length * sizeof(uint16_t));
        return Napi::Uint16Array::New(env, length, buf, 0);
      }
      case mlx::core::int32: {
        auto buf = make_external(arr->data<int32_t>(), length * sizeof(int32_t));
        return Napi::Int32Array::New(env, length, buf, 0);
      }
      case mlx::core::uint32: {
        auto buf = make_external(arr->data<uint32_t>(), length * sizeof(uint32_t));
        return Napi::Uint32Array::New(env, length, buf, 0);
      }
      case mlx::core::int64: {
        auto buf = make_external(arr->data<int64_t>(), length * sizeof(int64_t));
        return Napi::BigInt64Array::New(env, length, buf, 0);
      }
      case mlx::core::uint64: {
        auto buf = make_external(arr->data<uint64_t>(), length * sizeof(uint64_t));
        return Napi::BigUint64Array::New(env, length, buf, 0);
      }
      case mlx::core::complex64: {
        // complex64 is stored as interleaved (re, im) float pairs.  Expose as
        // Float32Array of length*2 — same zero-copy view.
        auto buf = make_external(
            arr->data<mlx::core::complex64_t>(), length * 2 * sizeof(float));
        return Napi::Float32Array::New(env, length * 2, buf, 0);
      }
      default:
        Napi::TypeError::New(env, "Unsupported dtype for toTypedArray")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
  }

  std::shared_ptr<mlx::bridge::ArrayHolder> holder_;
};

// Static member definition placed after the class so it can call the private
// helper.
std::optional<mlx::core::array> ArrayWrapper::BuildFromTyped(
    Napi::Env env,
    const Napi::TypedArray& typed,
    const Napi::Array& shapeArray,
    std::optional<mlx::core::Dtype> requestedDtype) {
  return MakeArrayFromTyped(env, typed, shapeArray, requestedDtype);
}
mlx::core::Shape ParseShapeArgument(Napi::Env env, const Napi::Value& value) {
  // Accept a single integer => 1D shape [n], or an array of integers => shape
  if (value.IsNumber()) {
    auto n = value.As<Napi::Number>().Int64Value();
    if (n < 0) {
      Napi::RangeError::New(env, "Shape dimension must be non-negative")
          .ThrowAsJavaScriptException();
      return {};
    }
    return mlx::core::Shape{static_cast<mlx::core::ShapeElem>(n)};
  }
  if (!value.IsArray()) {
    Napi::TypeError::New(
        env, "Shape must be an integer or an array of integers")
        .ThrowAsJavaScriptException();
    return {};
  }
  auto array = value.As<Napi::Array>();
  mlx::core::Shape shape;
  shape.reserve(array.Length());
  for (uint32_t i = 0; i < array.Length(); ++i) {
    auto dimVal = array.Get(i);
    if (!dimVal.IsNumber()) {
      Napi::TypeError::New(env, "Shape entries must be numbers")
          .ThrowAsJavaScriptException();
      return {};
    }
    auto dim = dimVal.As<Napi::Number>().Int64Value();
    if (dim < 0) {
      Napi::RangeError::New(env, "Shape dimensions must be non-negative")
          .ThrowAsJavaScriptException();
      return {};
    }
    shape.push_back(static_cast<mlx::core::ShapeElem>(dim));
  }
  return shape;
}

bool IsStreamObject(const Napi::Value& value) {
  if (!value.IsObject()) {
    return false;
  }
  auto obj = value.As<Napi::Object>();
  if (!obj.Has("index") || !obj.Has("device")) {
    return false;
  }
  return obj.Get("index").IsNumber();
}

mlx::core::Device ParseDeviceValue(Napi::Env env, const Napi::Value& value) {
  if (value.IsUndefined() || value.IsNull()) {
    return mlx::core::default_device();
  }

  std::string type;
  int index = 0;

  if (value.IsString()) {
    type = value.As<Napi::String>().Utf8Value();
  } else if (value.IsObject()) {
    auto obj = value.As<Napi::Object>();
    if (obj.Has("type")) {
      type = obj.Get("type").As<Napi::String>().Utf8Value();
    }
    if (obj.Has("index")) {
      index = obj.Get("index").As<Napi::Number>().Int32Value();
    }
  } else {
    Napi::TypeError::New(
        env, "Device must be a string or object with type/index")
        .ThrowAsJavaScriptException();
    return mlx::core::default_device();
  }

  if (type.empty() || type == "cpu") {
    return mlx::core::Device(mlx::core::Device::cpu, index);
  }
  if (type == "gpu") {
    return mlx::core::Device(mlx::core::Device::gpu, index);
  }

  Napi::TypeError::New(env, "Unsupported device type: " + type)
      .ThrowAsJavaScriptException();
  return mlx::core::default_device();
}

mlx::core::Stream ParseStreamValue(Napi::Env env, const Napi::Value& value) {
  if (!IsStreamObject(value)) {
    Napi::TypeError::New(env, "Expected a stream or device object")
        .ThrowAsJavaScriptException();
    return mlx::core::Stream(0, mlx::core::default_device());
  }
  auto obj = value.As<Napi::Object>();
  int index = obj.Get("index").As<Napi::Number>().Int32Value();
  auto deviceValue = obj.Get("device");
  auto device = ParseDeviceValue(env, deviceValue);
  return mlx::core::Stream(index, device);
}

mlx::core::StreamOrDevice ParseStreamOrDeviceValue(
    Napi::Env env,
    const Napi::Value& value) {
  if (value.IsUndefined() || value.IsNull()) {
    return {};
  }
  if (IsStreamObject(value)) {
    return ParseStreamValue(env, value);
  }
  return ParseDeviceValue(env, value);
}

// Helper: infer dtype for JS numbers/booleans/bigints
mlx::core::Dtype InferScalarDtype(const Napi::Value& v) {
  if (v.IsBoolean())
    return mlx::core::bool_;
  if (v.IsBigInt())
    return mlx::core::int64;
  if (v.IsNumber()) {
    double d = v.As<Napi::Number>().DoubleValue();
    double r = std::floor(d);
    // If it's an integer within int32 range, default to int32.
    // This matches MLX Python's behavior for '2' vs '2.0' (as best as JS can).
    if (std::fabs(d - r) < 1e-12 && d >= std::numeric_limits<int32_t>::min() &&
        d <= std::numeric_limits<int32_t>::max()) {
      return mlx::core::int32;
    }
    return mlx::core::float32;
  }
  return mlx::core::float32;
}

mlx::core::StreamOrDevice GetStreamArgument(
    const Napi::CallbackInfo& info,
    size_t startIdx) {
  for (size_t i = startIdx; i < info.Length(); ++i) {
    if (info[i].IsObject() && !info[i].IsArray() && !info[i].IsNull()) {
      auto obj = info[i].As<Napi::Object>();
      // Check for Stream (stream_id/device) OR Device (type/index)
      if (obj.Has("stream_id") || obj.Has("device") || obj.Has("type")) {
        return ParseStreamOrDeviceValue(info.Env(), info[i]);
      }
    }
  }
  return {};
}

std::vector<int>
ParseAxisVector(Napi::Env env, const Napi::Value& value, const char* name) {
  std::vector<int> axes;
  if (value.IsNumber()) {
    axes.push_back(value.As<Napi::Number>().Int32Value());
    return axes;
  }
  if (!value.IsArray()) {
    Napi::TypeError::New(
        env, std::string(name) + " must be an array of integers")
        .ThrowAsJavaScriptException();
    return axes;
  }
  auto array = value.As<Napi::Array>();
  axes.reserve(array.Length());
  for (uint32_t i = 0; i < array.Length(); ++i) {
    auto entry = array.Get(i);
    if (!entry.IsNumber()) {
      Napi::TypeError::New(env, std::string(name) + " entries must be numbers")
          .ThrowAsJavaScriptException();
      return {};
    }
    axes.push_back(entry.As<Napi::Number>().Int32Value());
  }
  return axes;
}

Napi::Object WrapArray(
    Napi::Env env,
    std::shared_ptr<mlx::core::array> tensor) {
  auto& data = mlx::node::GetAddonData(env);
  auto holder = std::make_shared<mlx::bridge::ArrayHolder>(std::move(*tensor));
  auto external = Napi::External<std::shared_ptr<mlx::bridge::ArrayHolder>>::New(
      env,
      new std::shared_ptr<mlx::bridge::ArrayHolder>(std::move(holder)),
      [](Napi::Env, std::shared_ptr<mlx::bridge::ArrayHolder>* data) {
        delete data;
      });
  return data.array_constructor.New({external});
}

Napi::Value AsArray(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for asarray")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "asarray expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse optional dtype and stream (kw-only last)
  std::optional<mlx::core::Dtype> maybeDtype;
  size_t streamIndex = 1;
  if (info.Length() >= 2 && IsDtypeArg(env, info[1], *addon)) {
    maybeDtype = MaybeParseDtype(env, info[1], mlx::core::float32, *addon);
    if (env.IsExceptionPending())
      return env.Null();
    streamIndex = 2;
  }
  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending())
    return env.Null();

  // Case A: already an mlx.core.Array
  if (info[0].IsObject()) {
    auto obj = info[0].As<Napi::Object>();
    auto ctor = addon->array_constructor.Value();
    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
      const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
      if (!wrapper) {
        Napi::TypeError::New(env, "Invalid mlx.core.Array")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
      auto a = wrapper->tensor();
      bool changed = false;
      if (maybeDtype.has_value() && maybeDtype.value() != a.dtype()) {
        a = mlx::core::astype(a, maybeDtype.value(), streamArg);
        changed = true;
      }
      if (!std::holds_alternative<std::monostate>(streamArg)) {
        // If only stream differs, copy; if dtype cast above already copied,
        // skip
        if (!changed) {
          a = mlx::core::copy(a, streamArg);
          changed = true;
        }
      }
      // If nothing changed, return original object
      if (!changed)
        return obj;
      return WrapArray(env, std::make_shared<mlx::core::array>(std::move(a)));
    }
  }

  // Case B: TypedArray / ArrayBuffer / Buffer → wrap as array (1D by default)
  if (info[0].IsTypedArray()) {
    auto typed = info[0].As<Napi::TypedArray>();
    Napi::Array shapeArray = Napi::Array::New(env, 1);
    shapeArray[(uint32_t)0u] =
        Napi::Number::New(env, (double)typed.ElementLength());
    auto arrOpt =
        ArrayWrapper::BuildFromTyped(env, typed, shapeArray, maybeDtype);
    if (!arrOpt.has_value())
      return env.Null();
    auto a = std::make_shared<mlx::core::array>(std::move(arrOpt.value()));
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      *a = mlx::core::copy(*a, streamArg);
    }
    return WrapArray(env, a);
  }

  // Case C: nested JS arrays / scalars → reuse ArrayFactory pathways
  if (info[0].IsArray()) {
    std::vector<double> flat;
    std::vector<int> dims;
    if (!ParseNestedNumberArray(env, info[0], flat, dims))
      return env.Null();
    mlx::core::Shape shape;
    for (int d : dims)
      shape.push_back(d);
    auto dtype = maybeDtype.value_or(mlx::core::float32);
    auto a = std::make_shared<mlx::core::array>(
        flat.begin(),
        shape.empty() ? mlx::core::Shape{(int)flat.size()} : shape,
        dtype);
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      *a = mlx::core::copy(*a, streamArg);
    }
    return WrapArray(env, a);
  }

  if (info[0].IsNumber() || info[0].IsBoolean() || info[0].IsBigInt()) {
    auto infer = [&](const Napi::Value& v) {
      if (v.IsBoolean())
        return mlx::core::bool_;
      if (v.IsBigInt())
        return mlx::core::int64;
      double d = v.As<Napi::Number>().DoubleValue();
      double r = std::floor(d);
      if (std::fabs(d - r) < 1e-12 &&
          d >= std::numeric_limits<int32_t>::min() &&
          d <= std::numeric_limits<int32_t>::max())
        return mlx::core::int32;
      return mlx::core::float32;
    };
    auto dtype = maybeDtype.value_or(infer(info[0]));
    double scalar = info[0].IsBoolean()
        ? (info[0].As<Napi::Boolean>().Value() ? 1.0 : 0.0)
        : info[0].IsBigInt()
        ? info[0].As<Napi::BigInt>().ToNumber().DoubleValue()
        : info[0].As<Napi::Number>().DoubleValue();
    auto a = std::make_shared<mlx::core::array>(scalar, dtype);
    if (!std::holds_alternative<std::monostate>(streamArg))
      *a = mlx::core::copy(*a, streamArg);
    return WrapArray(env, a);
  }

  Napi::TypeError::New(env, "Unsupported input to mx.core.asarray")
      .ThrowAsJavaScriptException();
  return env.Null();
}
// Helper: parse nested JS numeric arrays into flat vector + shape
// Parse nested numeric arrays; dims accumulates dimension sizes
// (outermost-first).
static bool ParseNestedNumberArray(
    Napi::Env env,
    const Napi::Value& v,
    std::vector<double>& data,
    std::vector<int>& dims) {
  if (v.IsArray()) {
    auto arr = v.As<Napi::Array>();
    size_t len = arr.Length();
    if (dims.empty())
      dims.push_back((int)len);
    else {
      // verify consistent size at this depth
      if (dims.back() != (int)len)
        dims.back() = (int)len; // accept ragged by overwriting
    }
    for (size_t i = 0; i < len; ++i) {
      if (!ParseNestedNumberArray(env, arr[i], data, dims))
        return false;
    }
    return true;
  }
  if (!v.IsNumber()) {
    Napi::TypeError::New(
        env, "Only numeric JS arrays are supported in mx.core.array")
        .ThrowAsJavaScriptException();
    return false;
  }
  data.push_back(v.As<Napi::Number>().DoubleValue());
  return true;
}

Napi::Value FromJSArray(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "from_js_array expects an array").ThrowAsJavaScriptException();
    return env.Null();
  }

  auto jsArray = info[0].As<Napi::Array>();
  auto dtype = mlx::core::float32;
  if (info.Length() > 1 && IsDtypeArg(env, info[1], *addon)) {
    dtype = MaybeParseDtype(env, info[1], dtype, *addon);
  }

  // Simplified version: flatten and create
  // For production this needs to be more robust for all dtypes and nested shapes
  struct ScalarVal {
    enum { DOUBLE, BOOL, BIGINT } type;
    union {
      double d;
      bool b;
      int64_t i;
    } val;
  };

  auto flatten = [](Napi::Array arr, auto& rec, std::vector<ScalarVal>& out) -> void {
    for (uint32_t i = 0; i < arr.Length(); ++i) {
      Napi::Value v = arr[i];
      if (v.IsArray()) {
        rec(v.As<Napi::Array>(), rec, out);
      } else if (v.IsBoolean()) {
        ScalarVal s; s.type = ScalarVal::BOOL; s.val.b = v.As<Napi::Boolean>().Value();
        out.push_back(s);
      } else if (v.IsBigInt()) {
        ScalarVal s; s.type = ScalarVal::BIGINT; s.val.i = v.As<Napi::BigInt>().Int64Value(nullptr);
        out.push_back(s);
      } else {
        ScalarVal s; s.type = ScalarVal::DOUBLE; s.val.d = v.As<Napi::Number>().DoubleValue();
        out.push_back(s);
      }
    }
  };

  std::vector<ScalarVal> flatData;
  flatten(jsArray, flatten, flatData);

  mlx::core::Shape shape;
  if (info.Length() > 2 && info[2].IsArray()) {
    shape = ParseShapeArgument(env, info[2]);
  } else {
    shape = {static_cast<int>(flatData.size())};
  }

  try {
    // If no explicit dtype, infer from all elements
    if (info.Length() <= 1 || !IsDtypeArg(env, info[1], *addon)) {
      bool has_float = false;
      bool has_bool = false;
      bool has_bigint = false;
      for (const auto& s : flatData) {
        if (s.type == ScalarVal::BOOL) has_bool = true;
        else if (s.type == ScalarVal::BIGINT) has_bigint = true;
        else if (std::fabs(s.val.d - std::round(s.val.d)) > 1e-12) has_float = true;
      }
      if (has_float) dtype = mlx::core::float32;
      else if (has_bigint) dtype = mlx::core::int64;
      else if (has_bool && flatData.size() > 0) {
        // If all are bools or ints that could be bools, but we saw a bool.
        // Actually, if there are ANY bools and no floats/bigints, we might want bool.
        // But MLX usually prefers int32 for [0, 1]. 
        // We'll follow the first element's lead if it's bool.
        if (flatData[0].type == ScalarVal::BOOL) dtype = mlx::core::bool_;
        else dtype = mlx::core::int32;
      } else {
        dtype = mlx::core::int32;
      }
    }

    if (dtype == mlx::core::bool_) {
      auto data = std::unique_ptr<bool[]>(new bool[flatData.size()]);
      for (size_t i = 0; i < flatData.size(); ++i) {
        const auto& s = flatData[i];
        if (s.type == ScalarVal::BOOL) data[i] = s.val.b;
        else if (s.type == ScalarVal::DOUBLE) data[i] = s.val.d != 0;
        else data[i] = s.val.i != 0;
      }
      return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::array(data.get(), shape)));
    } else if (dtype == mlx::core::int32) {
      std::vector<int32_t> data;
      for (const auto& s : flatData) {
        if (s.type == ScalarVal::DOUBLE) data.push_back((int32_t)s.val.d);
        else if (s.type == ScalarVal::BOOL) data.push_back(s.val.b ? 1 : 0);
        else data.push_back((int32_t)s.val.i);
      }
      return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::array(data.data(), shape)));
    } else if (dtype == mlx::core::int64) {
      std::vector<int64_t> data;
      for (const auto& s : flatData) {
        if (s.type == ScalarVal::DOUBLE) data.push_back((int64_t)s.val.d);
        else if (s.type == ScalarVal::BOOL) data.push_back(s.val.b ? 1 : 0);
        else data.push_back(s.val.i);
      }
      return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::array(data.data(), shape)));
    } else {
      std::vector<float> data;
      for (const auto& s : flatData) {
        if (s.type == ScalarVal::DOUBLE) data.push_back((float)s.val.d);
        else if (s.type == ScalarVal::BOOL) data.push_back(s.val.b ? 1.0f : 0.0f);
        else data.push_back((float)s.val.i);
      }
      return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::array(data.data(), shape, dtype)));
    }
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value ArrayFactory(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Case A: first arg is TypedArray (optional explicit shape)
  if (info.Length() >= 1 && info[0].IsTypedArray()) {
    auto typed = info[0].As<Napi::TypedArray>();
    // Optional explicit shape; if omitted, assume 1D [len]
    Napi::Array shapeArray;
    size_t argIndex = 1;
    if (info.Length() >= 2 && info[1].IsArray()) {
      shapeArray = info[1].As<Napi::Array>();
      argIndex = 2;
    } else {
      shapeArray = Napi::Array::New(env, 1);
      shapeArray[(uint32_t)0u] =
          Napi::Number::New(env, (double)typed.ElementLength());
    }

    std::optional<mlx::core::Dtype> maybeDtype;
    size_t streamIndex = argIndex;
    if (info.Length() > argIndex && IsDtypeArg(env, info[argIndex], *addon)) {
      maybeDtype =
          MaybeParseDtype(env, info[argIndex], mlx::core::float32, *addon);
      if (env.IsExceptionPending())
        return env.Null();
      streamIndex = argIndex + 1;
    }
    auto streamArg = GetStreamArgument(info, streamIndex);
    if (env.IsExceptionPending())
      return env.Null();

    auto arrOpt =
        ArrayWrapper::BuildFromTyped(env, typed, shapeArray, maybeDtype);
    if (!arrOpt.has_value())
      return env.Null();
    std::shared_ptr<mlx::core::array> tensor;
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      tensor = std::make_shared<mlx::core::array>(
          mlx::core::copy(arrOpt.value(), streamArg));
    } else {
      tensor = std::make_shared<mlx::core::array>(std::move(arrOpt.value()));
    }
    return WrapArray(env, tensor);
  }

  // Case B: first arg is an existing mlx.core.Array; behave like asarray/astype
  if (info.Length() >= 1 && info[0].IsObject()) {
    auto obj = info[0].As<Napi::Object>();
    auto ctor = addon->array_constructor.Value();
    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
      const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
      if (!wrapper) {
        Napi::TypeError::New(env, "Invalid mlx.core.Array")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
      std::optional<mlx::core::Dtype> maybeDtype;
      size_t streamIndex = 1;
      if (info.Length() >= 2 && IsDtypeArg(env, info[1], *addon)) {
        maybeDtype =
            MaybeParseDtype(env, info[1], wrapper->tensor().dtype(), *addon);
        if (env.IsExceptionPending())
          return env.Null();
        streamIndex = 2;
      }
      auto streamArg = GetStreamArgument(info, streamIndex);
      if (env.IsExceptionPending())
        return env.Null();
      auto a = wrapper->tensor();
      if (maybeDtype.has_value() && maybeDtype.value() != a.dtype()) {
        a = mlx::core::astype(a, maybeDtype.value(), streamArg);
      } else if (!std::holds_alternative<std::monostate>(streamArg)) {
        a = mlx::core::copy(a, streamArg);
      }
      return WrapArray(env, std::make_shared<mlx::core::array>(std::move(a)));
    }
  }

  // Case C: nested JS numeric arrays
  if (info.Length() >= 1 && info[0].IsArray()) {
    std::vector<double> flat;
    std::vector<int> dims;
    if (!ParseNestedNumberArray(env, info[0], flat, dims))
      return env.Null();
    // Compute shape from dims; when ragged, dims may have been overwritten;
    // treat as 1D
    mlx::core::Shape shape;
    if (!dims.empty()) {
      for (int d : dims)
        shape.push_back(d);
    } else {
      shape.push_back((int)flat.size());
    }
    // Optional dtype and stream
    std::optional<mlx::core::Dtype> maybeDtype;
    size_t streamIndex = 1;
    if (info.Length() >= 2 && IsDtypeArg(env, info[1], *addon)) {
      maybeDtype = MaybeParseDtype(env, info[1], mlx::core::float32, *addon);
      if (env.IsExceptionPending())
        return env.Null();
      streamIndex = 2;
    }
    auto streamArg = GetStreamArgument(info, streamIndex);
    if (env.IsExceptionPending())
      return env.Null();

    // Build array from flattened doubles; use float32 default unless dtype
    // provided
    auto dtype = maybeDtype.value_or(mlx::core::float32);
    std::shared_ptr<mlx::core::array> tensor;
    if (dtype == mlx::core::float64) {
      // float64 needs Dekker split: convert double → float64_t in buffer
      size_t nbytes = flat.size() * sizeof(mlx::core::float64_t);
      auto buf = mlx::core::allocator::malloc(nbytes);
      auto* ptr = static_cast<mlx::core::float64_t*>(buf.raw_ptr());
      for (size_t i = 0; i < flat.size(); i++) {
        ptr[i] = mlx::core::float64_t(flat[i]);
      }
      tensor = std::make_shared<mlx::core::array>(buf, shape, dtype);
    } else {
      tensor =
          std::make_shared<mlx::core::array>(flat.begin(), shape, dtype);
    }
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      *tensor = mlx::core::copy(*tensor, streamArg);
    }
    return WrapArray(env, tensor);
  }

  // Case D: scalar → rank-0
  if (info.Length() >= 1 &&
      (info[0].IsNumber() || info[0].IsBoolean() || info[0].IsBigInt())) {
    std::optional<mlx::core::Dtype> maybeDtype;
    size_t streamIndex = 1;
    if (info.Length() >= 2 && IsDtypeArg(env, info[1], *addon)) {
      maybeDtype = MaybeParseDtype(env, info[1], mlx::core::float32, *addon);
      if (env.IsExceptionPending())
        return env.Null();
      streamIndex = 2;
    }
    auto streamArg = GetStreamArgument(info, streamIndex);
    if (env.IsExceptionPending())
      return env.Null();

    // Infer dtype if omitted (bool->bool, integer Number->int32 else float32,
    // BigInt->int64)
    auto infer = [&](const Napi::Value& v) {
      if (v.IsBoolean())
        return mlx::core::bool_;
      if (v.IsBigInt())
        return mlx::core::int64;
      double d = v.As<Napi::Number>().DoubleValue();
      double r = std::floor(d);
      // Only treat as int32 if it's exactly equal to its floor AND within range.
      if (d == r &&
          d >= (double)std::numeric_limits<int32_t>::min() &&
          d <= (double)std::numeric_limits<int32_t>::max())
        return mlx::core::int32;
      return mlx::core::float32;
    };
    auto dtype = maybeDtype.value_or(infer(info[0]));
    double scalar = info[0].IsBoolean()
        ? (info[0].As<Napi::Boolean>().Value() ? 1.0 : 0.0)
        : info[0].IsBigInt()
        ? info[0].As<Napi::BigInt>().ToNumber().DoubleValue()
        : info[0].As<Napi::Number>().DoubleValue();
    std::shared_ptr<mlx::core::array> tensor;
    if (dtype == mlx::core::float64) {
      // float64 scalar needs Dekker split
      auto buf = mlx::core::allocator::malloc(sizeof(mlx::core::float64_t));
      auto* ptr = static_cast<mlx::core::float64_t*>(buf.raw_ptr());
      *ptr = mlx::core::float64_t(scalar);
      tensor = std::make_shared<mlx::core::array>(
          buf, mlx::core::Shape{}, dtype);
    } else {
      tensor = std::make_shared<mlx::core::array>(scalar, dtype);
    }
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      *tensor = mlx::core::copy(*tensor, streamArg);
    }
    return WrapArray(env, tensor);
  }

  Napi::TypeError::New(env, "Unsupported input to mx.core.array")
      .ThrowAsJavaScriptException();
  return env.Null();
}

const ArrayWrapper* UnwrapArray(Napi::Env env, const Napi::Value& value) {
  if (!value.IsObject()) {
    Napi::TypeError::New(env, "Expected mlx.core.Array instance")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
  auto obj = value.As<Napi::Object>();
  auto ctor = mlx::node::GetAddonData(env).array_constructor.Value();
  if (ctor.IsEmpty() || !obj.InstanceOf(ctor)) {
    Napi::TypeError::New(env, "Expected mlx.core.Array instance")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
  return Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
}

mlx::core::Dtype MaybeParseDtype(
    Napi::Env env,
    const Napi::Value& value,
    mlx::core::Dtype fallback,
    mlx::node::AddonData& addon_data) {
  if (value.IsUndefined() || value.IsNull()) {
    return fallback;
  }
  // Accept string dtype keys directly (e.g. 'float32', 'bool')
  if (value.IsString()) {
    return ParseDtypeKey(env, value.As<Napi::String>().Utf8Value());
  }
  // Use Dtype.key property (exported via InstanceAccessor) to avoid unwrap
  // issues
  if (value.IsObject()) {
    auto obj = value.As<Napi::Object>();
    auto key = obj.Get("key");
    if (key.IsString()) {
      return ParseDtypeKey(env, key.As<Napi::String>().Utf8Value());
    }
  }
  Napi::TypeError::New(
      env, "dtype must be a mlx.core.Dtype object or string (e.g., mlx.float32 or 'float32')")
      .ThrowAsJavaScriptException();
  return fallback;
}

bool IsDtypeArg(
    Napi::Env env,
    const Napi::Value& value,
    mlx::node::AddonData&) {
  if (value.IsUndefined() || value.IsNull())
    return false;
  // Accept string dtype keys (e.g. 'float32', 'bool')
  if (value.IsString()) {
    auto key = value.As<Napi::String>().Utf8Value();
    return DtypeLookup().count(key) > 0;
  }
  if (!value.IsObject())
    return false;
  auto obj = value.As<Napi::Object>();
  auto key = obj.Get("key");
  return key.IsString();
}

template <typename T>
T ParseScalarValue(Napi::Env env, const Napi::Value& value) {
  if constexpr (std::is_same_v<T, double>) {
    if (!value.IsNumber()) {
      Napi::TypeError::New(env, "fill value must be a number")
          .ThrowAsJavaScriptException();
      return 0.0;
    }
    return value.As<Napi::Number>().DoubleValue();
  } else {
    static_assert(sizeof(T) == 0, "Unsupported scalar type");
  }
}

// Accumulates element-kind flags while flattening JS nested arrays
struct TypeScan {
  bool has_float = false;
  bool has_bigint = false;
  bool has_int = false;
  bool has_bool = false;
};

static bool FlattenNestedJS(
    Napi::Env env,
    const Napi::Value& v,
    std::vector<double>& data,
    std::vector<int>& dims,
    size_t depth,
    TypeScan& scan) {
  if (v.IsArray()) {
    auto arr = v.As<Napi::Array>();
    const size_t len = arr.Length();
    if (dims.size() <= depth)
      dims.push_back((int)len);
    else if (dims[depth] != (int)len) {
      Napi::TypeError::New(env, "Ragged nested lists are not supported")
          .ThrowAsJavaScriptException();
      return false;
    }
    for (size_t i = 0; i < len; ++i) {
      if (!FlattenNestedJS(env, arr[i], data, dims, depth + 1, scan))
        return false;
    }
    return true;
  }
  if (v.IsBoolean()) {
    scan.has_bool = true;
    data.push_back(v.As<Napi::Boolean>().Value() ? 1.0 : 0.0);
    return true;
  }
  if (v.IsBigInt()) {
    scan.has_bigint = true;
    bool lossless = false;
    auto val = v.As<Napi::BigInt>().Int64Value(&lossless);
    data.push_back(static_cast<double>(val));
    return true;
  }
  if (v.IsNumber()) {
    double d = v.As<Napi::Number>().DoubleValue();
    double r = std::floor(d);
    if (std::fabs(d - r) > 1e-12)
      scan.has_float = true;
    else
      scan.has_int = true;
    data.push_back(d);
    return true;
  }
  Napi::TypeError::New(
      env, "Only numeric JS arrays are supported in mx.core.array")
      .ThrowAsJavaScriptException();
  return false;
}

static mlx::core::Dtype ChooseDtypeFromScan(const TypeScan& s) {
  if (s.has_float)
    return mlx::core::float32;
  if (s.has_bigint)
    return mlx::core::int64;
  if (s.has_int)
    return mlx::core::int32;
  if (s.has_bool)
    return mlx::core::bool_;
  return mlx::core::float32;
}

// Unified conversion helper (internal to binding, not exported)
// - overrideShape: only used by array(typed, shape, ...)
static std::optional<mlx::core::array> ToArrayValue(
    Napi::Env env,
    const Napi::Value& x,
    std::optional<mlx::core::Dtype> requestedDtype,
    const mlx::core::StreamOrDevice& streamArg,
    std::optional<Napi::Array> overrideShape) {
  auto& addon = mlx::node::GetAddonData(env);

  // Case: existing mlx Array
  if (x.IsObject()) {
    auto obj = x.As<Napi::Object>();
    auto ctor = addon.array_constructor.Value();
    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
      const auto* w = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
      if (w == nullptr) {
        Napi::TypeError::New(env, "Invalid mlx.core.Array")
            .ThrowAsJavaScriptException();
        return {};
      }
      auto a = w->tensor();
      if (requestedDtype.has_value() && requestedDtype.value() != a.dtype()) {
        a = mlx::core::astype(a, requestedDtype.value(), streamArg);
      } else if (!std::holds_alternative<std::monostate>(streamArg)) {
        a = mlx::core::copy(a, streamArg);
      }
      return a;
    }
  }

  // Case: TypedArray
  if (x.IsTypedArray()) {
    auto typed = x.As<Napi::TypedArray>();
    Napi::Array shapeArray;
    if (overrideShape.has_value()) {
      shapeArray = *overrideShape;
    } else {
      shapeArray = Napi::Array::New(env, 1);
      shapeArray[(uint32_t)0u] =
          Napi::Number::New(env, (double)typed.ElementLength());
    }
    auto arrOpt =
        ArrayWrapper::BuildFromTyped(env, typed, shapeArray, requestedDtype);
    if (!arrOpt.has_value())
      return {};
    auto a = std::move(arrOpt.value());
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      a = mlx::core::copy(a, streamArg);
    }
    return a;
  }

  // Case: nested JS arrays
  if (x.IsArray()) {
    std::vector<double> flat;
    std::vector<int> dims;
    TypeScan scan;
    if (!FlattenNestedJS(env, x, flat, dims, 0, scan))
      return {};
    mlx::core::Shape shape;
    for (int d : dims)
      shape.push_back(d);
    auto dtype = requestedDtype.value_or(mlx::core::float32);
    std::optional<mlx::core::array> out;
    switch (dtype) {
      case mlx::core::int32: {
        std::vector<int32_t> host(flat.size());
        for (size_t i = 0; i < flat.size(); ++i)
          host[i] = static_cast<int32_t>(std::llround(flat[i]));
        out = mlx::core::array(
            host.begin(),
            shape.empty() ? mlx::core::Shape{(int)host.size()} : shape,
            dtype);
        break;
      }
      case mlx::core::int64: {
        std::vector<int64_t> host(flat.size());
        for (size_t i = 0; i < flat.size(); ++i)
          host[i] = static_cast<int64_t>(std::llround(flat[i]));
        out = mlx::core::array(
            host.begin(),
            shape.empty() ? mlx::core::Shape{(int)host.size()} : shape,
            dtype);
        break;
      }
      case mlx::core::bool_: {
        std::vector<bool> host(flat.size());
        for (size_t i = 0; i < flat.size(); ++i)
          host[i] = (flat[i] != 0.0);
        out = mlx::core::array(
            host.begin(),
            shape.empty() ? mlx::core::Shape{(int)host.size()} : shape,
            dtype);
        break;
      }
      case mlx::core::float64: {
        // Allocate buffer and write float64_t (double-double) values directly
        auto out_shape =
            shape.empty() ? mlx::core::Shape{(int)flat.size()} : shape;
        size_t nbytes = flat.size() * sizeof(mlx::core::float64_t);
        auto buf = mlx::core::allocator::malloc(nbytes);
        auto* ptr = static_cast<mlx::core::float64_t*>(buf.raw_ptr());
        for (size_t i = 0; i < flat.size(); i++) {
          ptr[i] = mlx::core::float64_t(flat[i]);
        }
        out = mlx::core::array(buf, out_shape, dtype);
        break;
      }
      default: { // float32 and others default to float32 path
        std::vector<float> host(flat.size());
        for (size_t i = 0; i < flat.size(); ++i)
          host[i] = static_cast<float>(flat[i]);
        auto dt =
            (dtype == mlx::core::float16 || dtype == mlx::core::bfloat16 ||
             dtype == mlx::core::complex64)
            ? dtype
            : mlx::core::float32;
        out = mlx::core::array(
            host.begin(),
            shape.empty() ? mlx::core::Shape{(int)host.size()} : shape,
            dt);
        if (dtype != dt)
          out = mlx::core::astype(*out, dtype);
        break;
      }
    }
    auto a = std::move(out.value());
    if (!std::holds_alternative<std::monostate>(streamArg)) {
      a = mlx::core::copy(a, streamArg);
    }
    return a;
  }

  // Case: scalar
  if (x.IsBoolean() || x.IsBigInt() || x.IsNumber()) {
    auto infer = [&](const Napi::Value& v) {
      if (v.IsBoolean())
        return mlx::core::bool_;
      if (v.IsBigInt())
        return mlx::core::int64;
      double d = v.As<Napi::Number>().DoubleValue();
      double r = std::floor(d);
      return (std::fabs(d - r) < 1e-12 ? mlx::core::int32 : mlx::core::float32);
    };
    auto dtype = requestedDtype.value_or(infer(x));
    double scalar = x.IsBoolean() ? (x.As<Napi::Boolean>().Value() ? 1.0 : 0.0)
        : x.IsBigInt() ? (double)x.As<Napi::BigInt>().Int64Value(nullptr)
                       : x.As<Napi::Number>().DoubleValue();
    auto a = mlx::core::array(scalar, dtype);
    if (!std::holds_alternative<std::monostate>(streamArg))
      a = mlx::core::copy(a, streamArg);
    return a;
  }

  Napi::TypeError::New(env, "Unsupported input").ThrowAsJavaScriptException();
  return {};
}

// arange(stop[, step], dtype=None, *, stream)
// arange(start, stop[, step], dtype=None, *, stream)
// Default dtype rules (Python parity):
//  - start/stop form: float32 if any arg is float, else int32
//  - stop form: dtype(step ? promote(dtype(stop), dtype(step)) : dtype(stop))
//  - If any BigInt present and no float, use int64
Napi::Value Arange(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for arange")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "arange expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto is_num_like = [](const Napi::Value& v) {
    return v.IsNumber() || v.IsBigInt();
  };
  auto is_float_num = [](const Napi::Value& v) {
    if (!v.IsNumber())
      return false;
    double d = v.As<Napi::Number>().DoubleValue();
    double r = std::floor(d);
    return std::fabs(d - r) > 1e-12; // non-integer
  };

  bool two_positional =
      (info.Length() >= 2 && is_num_like(info[0]) && is_num_like(info[1]));

  std::optional<Napi::Value> start_v;
  Napi::Value stop_v = info[0];
  std::optional<Napi::Value> step_v;
  size_t idx = 1;
  if (two_positional) {
    start_v = info[0];
    stop_v = info[1];
    idx = 2;
  }

  if (info.Length() > idx && is_num_like(info[idx])) {
    step_v = info[idx];
    idx += 1;
  }

  // Optional dtype and stream
  std::optional<mlx::core::Dtype> maybeDtype;
  if (info.Length() > idx && IsDtypeArg(env, info[idx], *addon)) {
    maybeDtype = MaybeParseDtype(env, info[idx], mlx::core::float32, *addon);
    if (env.IsExceptionPending())
      return env.Null();
    idx += 1;
  }
  auto streamArg = GetStreamArgument(info, idx);
  if (env.IsExceptionPending())
    return env.Null();

  // Defaults
  bool any_float = (is_float_num(stop_v)) ||
      (start_v && is_float_num(*start_v)) || (step_v && is_float_num(*step_v));
  bool any_bigint = stop_v.IsBigInt() || (start_v && (*start_v).IsBigInt()) ||
      (step_v && (*step_v).IsBigInt());

  auto default_dtype = [&]() -> mlx::core::Dtype {
    if (two_positional) {
      if (any_float)
        return mlx::core::float32;
      if (any_bigint)
        return mlx::core::int64;
      return mlx::core::int32;
    } else {
      // stop[, step]
      if (any_float)
        return mlx::core::float32;
      if (any_bigint)
        return mlx::core::int64;
      return mlx::core::int32;
    }
  }();

  auto dtype = maybeDtype.value_or(default_dtype);

  // Convert to concrete scalars
  auto to_i64 = [&](const Napi::Value& v) -> int64_t {
    if (v.IsBigInt()) {
      bool lossless = false;
      return v.As<Napi::BigInt>().Int64Value(&lossless);
    }
    return static_cast<int64_t>(v.As<Napi::Number>().DoubleValue());
  };
  auto to_f64 = [&](const Napi::Value& v) -> double {
    if (v.IsBigInt()) {
      // represent BigInt as double (may lose precision, but dtype chosen is
      // int64 unless overridden)
      return static_cast<double>(to_i64(v));
    }
    return v.As<Napi::Number>().DoubleValue();
  };

  std::shared_ptr<mlx::core::array> out;
  try {
    double start = two_positional ? to_f64(*start_v) : 0.0;
    double stop = to_f64(stop_v);
    double step = step_v ? to_f64(*step_v) : 1.0;
    out = std::make_shared<mlx::core::array>(
        mlx::core::arange(start, stop, step, dtype, streamArg));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
  return WrapArray(env, out);
}

mlx::core::array MakeFilledArray(
    const mlx::core::Shape& shape,
    double fill,
    mlx::core::Dtype dtype) {
  return mlx::core::full(shape, fill, dtype);
}

Napi::Value Zeros(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for zeros")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "zeros expects a shape array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto dtype = mlx::core::float32;
  // Signature: zeros(shape, dtype?, streamOrDevice?)
  // If arg1 is a dtype (preferred) or a string key -> dtype; else
  // stream/device.
  size_t streamIndex = 1;
  if (info.Length() >= 2 && IsDtypeArg(env, info[1], *addon)) {
    dtype = MaybeParseDtype(env, info[1], dtype, *addon);
    if (env.IsExceptionPending()) {
      return env.Null();
    }
    streamIndex = 2;
  }
  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  std::shared_ptr<mlx::core::array> tensor = std::make_shared<mlx::core::array>(
      mlx::core::zeros(shape, dtype, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Ones(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for ones")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "ones expects a shape array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto dtype = mlx::core::float32;
  size_t streamIndex = 1;
  if (info.Length() >= 2 && IsDtypeArg(env, info[1], *addon)) {
    dtype = MaybeParseDtype(env, info[1], dtype, *addon);
    if (env.IsExceptionPending()) {
      return env.Null();
    }
    streamIndex = 2;
  }
  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::ones(shape, dtype, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Full(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for full")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "full expects shape and vals")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending())
    return env.Null();

  // Optional dtype and stream
  std::optional<mlx::core::Dtype> maybeDtype;
  size_t streamIndex = 2;
  if (info.Length() >= 3 && IsDtypeArg(env, info[2], *addon)) {
    maybeDtype = MaybeParseDtype(env, info[2], mlx::core::float32, *addon);
    if (env.IsExceptionPending())
      return env.Null();
    streamIndex = 3;
  }
  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending())
    return env.Null();

  // Case: vals is an Array (broadcast)
  if (info[1].IsObject()) {
    auto obj = info[1].As<Napi::Object>();
    auto ctor = addon->array_constructor.Value();
    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
      const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
      if (!wrapper) {
        Napi::TypeError::New(env, "Invalid mlx.core.Array")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
      auto src = wrapper->tensor();
      if (maybeDtype.has_value()) {
        auto casted = mlx::core::astype(src, *maybeDtype, streamArg);
        return WrapArray(
            env,
            std::make_shared<mlx::core::array>(
                mlx::core::full(shape, casted, streamArg)));
      } else {
        return WrapArray(
            env,
            std::make_shared<mlx::core::array>(
                mlx::core::full(shape, src, streamArg)));
      }
    }
  }

  // Case: vals is TypedArray or nested lists → use ToArrayValue then broadcast
  if (info[1].IsTypedArray() || info[1].IsArray()) {
    auto srcOpt =
        ToArrayValue(env, info[1], maybeDtype, streamArg, std::nullopt);
    if (!srcOpt.has_value())
      return env.Null();
    return WrapArray(
        env,
        std::make_shared<mlx::core::array>(
            mlx::core::full(shape, srcOpt.value(), streamArg)));
  }

  double scalar = 0.0;
  if (info[1].IsBoolean())
    scalar = info[1].As<Napi::Boolean>().Value() ? 1.0 : 0.0;
  else if (info[1].IsBigInt())
    scalar = info[1].As<Napi::BigInt>().ToNumber().DoubleValue();
  else {
    scalar = ParseScalarValue<double>(env, info[1]);
    if (env.IsExceptionPending())
      return env.Null();
  }
  auto final_dtype = maybeDtype.value_or(InferScalarDtype(info[1]));
  return WrapArray(
      env,
      std::make_shared<mlx::core::array>(
          mlx::core::full(shape, scalar, final_dtype, streamArg)));
}

Napi::Value ZerosLike(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(env, "zeros_like expects exactly one array argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* wrapper = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || wrapper == nullptr) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::zeros_like(wrapper->tensor()));
  return WrapArray(env, tensor);
}

Napi::Value OnesLike(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() != 1) {
    Napi::TypeError::New(env, "ones_like expects exactly one array argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* wrapper = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || wrapper == nullptr) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::ones_like(wrapper->tensor()));
  return WrapArray(env, tensor);
}

Napi::Value Reshape(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "reshape expects an array and a shape")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* wrapper = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || wrapper == nullptr) {
    return env.Null();
  }

  // Parse shape allowing -1 as wildcard (unlike ParseShapeArgument)
  if (!info[1].IsArray()) {
    Napi::TypeError::New(env, "reshape: shape must be an array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shapeArr = info[1].As<Napi::Array>();
  mlx::core::Shape shape;
  shape.reserve(shapeArr.Length());
  for (uint32_t i = 0; i < shapeArr.Length(); ++i) {
    auto dimVal = shapeArr.Get(i);
    if (!dimVal.IsNumber()) {
      Napi::TypeError::New(env, "reshape: shape entries must be numbers")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    shape.push_back(
        static_cast<mlx::core::ShapeElem>(dimVal.As<Napi::Number>().Int64Value()));
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::reshape(wrapper->tensor(), shape, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Transpose(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "transpose expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* wrapper = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || wrapper == nullptr) {
    return env.Null();
  }

  bool axesProvided = false;
  std::vector<int> axes;
  size_t streamIndex = 1;

  if (info.Length() >= 2) {
    auto arg1 = info[1];
    if (arg1.IsArray() || arg1.IsNumber()) {
      axes = ParseAxisVector(env, arg1, "axes");
      if (env.IsExceptionPending()) {
        return env.Null();
      }
      axesProvided = true;
      streamIndex = 2;
    } else if (arg1.IsUndefined() || arg1.IsNull()) {
      streamIndex = 2;
    } else {
      streamIndex = 1;
    }
  }

  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  std::shared_ptr<mlx::core::array> tensor;
  if (axesProvided) {
    tensor = std::make_shared<mlx::core::array>(
        mlx::core::transpose(wrapper->tensor(), axes, streamArg));
  } else {
    tensor = std::make_shared<mlx::core::array>(
        mlx::core::transpose(wrapper->tensor(), streamArg));
  }
  return WrapArray(env, tensor);
}

Napi::Value MoveAxis(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "moveaxis expects array, source, destination")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* wrapper = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || wrapper == nullptr) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  if (info[1].IsArray() || info[2].IsArray()) {
    auto srcVec = ParseAxisVector(env, info[1], "source axes");
    if (env.IsExceptionPending()) {
      return env.Null();
    }
    auto dstVec = ParseAxisVector(env, info[2], "destination axes");
    if (env.IsExceptionPending()) {
      return env.Null();
    }
    if (srcVec.size() != dstVec.size()) {
      Napi::RangeError::New(
          env, "source and destination axes must have the same length")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    mlx::core::array tensor = wrapper->tensor();
    for (size_t i = 0; i < srcVec.size(); ++i) {
      tensor = mlx::core::moveaxis(tensor, srcVec[i], dstVec[i], streamArg);
    }
    return WrapArray(
        env, std::make_shared<mlx::core::array>(std::move(tensor)));
  }

  if (!info[1].IsNumber() || !info[2].IsNumber()) {
    Napi::TypeError::New(
        env, "moveaxis expects numeric source/destination axes")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int source = info[1].As<Napi::Number>().Int32Value();
  int destination = info[2].As<Napi::Number>().Int32Value();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::moveaxis(wrapper->tensor(), source, destination, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value SwapAxes(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "swapaxes expects array, axis1, axis2")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* wrapper = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || wrapper == nullptr) {
    return env.Null();
  }

  if (!info[1].IsNumber() || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "swapaxes expects numeric axis values")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int axis1 = info[1].As<Napi::Number>().Int32Value();
  int axis2 = info[2].As<Napi::Number>().Int32Value();

  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::swapaxes(wrapper->tensor(), axis1, axis2, streamArg));
  return WrapArray(env, tensor);
}

// Helper to convert scalar or array to mlx::core::array
mlx::core::array ToArray(Napi::Env env, const Napi::Value& value) {
  // If it's already an MLX array, unwrap it
  if (value.IsObject()) {
    auto obj = value.As<Napi::Object>();
    auto& data = mlx::node::GetAddonData(env);
    auto ctor = data.array_constructor.Value();
    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
      auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
      if (wrapper) {
        return wrapper->tensor();
      }
    }
  }

  // Convert scalar to array
  if (value.IsNumber() || value.IsBoolean()) {
    double d = value.IsBoolean() ? (value.As<Napi::Boolean>().Value() ? 1.0 : 0.0)
                                 : value.As<Napi::Number>().DoubleValue();
    return mlx::core::array(d, InferScalarDtype(value));
  }
  if (value.IsBigInt()) {
    bool lossless = false;
    return mlx::core::array(value.As<Napi::BigInt>().Int64Value(&lossless), mlx::core::int64);
  }

  Napi::TypeError::New(env, "Expected array or scalar (number/boolean/bigint)")
      .ThrowAsJavaScriptException();
  return mlx::core::array(0.0); // Never reached, but needed for return type
}

Napi::Value ArrayEqual(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "array_equal expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto a = ToArray(env, info[0]);
    if (env.IsExceptionPending()) return env.Null();
    auto b = ToArray(env, info[1]);
    if (env.IsExceptionPending()) return env.Null();
    auto res = mlx::core::array_equal(a, b);
    return WrapArray(env, std::make_shared<mlx::core::array>(res));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value Add(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "add expects two arguments (arrays or scalars)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  std::cerr << "Add: a.dtype=" << mlx::core::dtype_to_string(a.dtype()) 
            << " b.dtype=" << mlx::core::dtype_to_string(b.dtype()) << std::endl;

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto res = mlx::core::add(a, b, streamArg);
  std::cerr << "Add: result.dtype=" << mlx::core::dtype_to_string(res.dtype()) << std::endl;

  auto tensor =
      std::make_shared<mlx::core::array>(res);
  return WrapArray(env, tensor);
}

Napi::Value Multiply(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "multiply expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::multiply(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Subtract(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(
        env, "subtract expects two arguments (arrays or scalars)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::subtract(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Matmul(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "matmul expects two arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::matmul(a, b, streamArg));
  return WrapArray(env, tensor);
}
Napi::Value Where(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "where expects condition, x, y")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto condition = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto x = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto y = ToArray(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(mlx::core::where(
      condition, x, y, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Tan(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "tan expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::tan(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Sin(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "sin expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::sin(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Cos(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "cos expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::cos(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Arcsin(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "arcsin expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::arcsin(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Arccos(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "arccos expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::arccos(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Arctan(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "arctan expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::arctan(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Arctan2(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "arctan2 expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::arctan2(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Rsqrt(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "rsqrt expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::rsqrt(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Square(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "square expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::square(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Sign(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "sign expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::sign(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Abs(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "abs expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::abs(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Sqrt(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "sqrt expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::sqrt(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Exp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "exp expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::exp(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Log(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "log expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::log(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Divide(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "divide expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::divide(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Power(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "power expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::power(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Equal(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "equal expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::equal(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value NotEqual(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "not_equal expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::not_equal(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Less(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "less expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::less(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value LessEqual(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "less_equal expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::less_equal(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Greater(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "greater expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::greater(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value GreaterEqual(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "greater_equal expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::greater_equal(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Maximum(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "maximum expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::maximum(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Minimum(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "minimum expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }

  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::minimum(a, b, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Hello(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  const auto version = mlx::core::version();
  return Napi::String::New(env, "mlx core version " + version);
}

Napi::Value GPUInfo(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  try {
    // Ensure GPU is initialised and we have a valid pool for ObjC bridging.
    mlx::node::Runtime::Instance().EnsureMetalInit();
    auto map = mlx::core::metal::device_info();
    Napi::Object out = Napi::Object::New(env);
    for (const auto& kv : map) {
      const auto& key = kv.first;
      const auto& v = kv.second;
      if (std::holds_alternative<std::string>(v)) {
        out.Set(key, Napi::String::New(env, std::get<std::string>(v)));
      } else {
        out.Set(
            key,
            Napi::Number::New(env, static_cast<double>(std::get<size_t>(v))));
      }
    }
    return out;
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value GPUSanity(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
    auto result =
        mlx::node::Runtime::Instance().WithMetalPool([&]() -> Napi::Value {
          auto s2 = mlx::core::new_stream(mlx::core::Device::gpu);
          auto s3 = mlx::core::new_stream(mlx::core::Device::gpu);

          auto a = mlx::core::arange(1.f, 10.f, 1.f, mlx::core::float32, s2);
          auto b = mlx::core::arange(1.f, 10.f, 1.f, mlx::core::float32, s3);
          auto x = mlx::core::add(a, a, s2);
          auto y = mlx::core::add(b, b, s3);
          auto z = mlx::core::multiply(x, y);
          mlx::core::eval(z);

          const size_t length = z.size();
          auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(float));
          auto jsArray = Napi::Float32Array::New(env, length, buffer, 0);
          std::memcpy(buffer.Data(), z.data<float>(), length * sizeof(float));
          return jsArray;
        });
    return result;
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// Random functions
Napi::Value RandomUniform(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  
  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
  
  // Parse arguments: uniform(low, high, shape, dtype?, stream?)
  // uniform(shape, dtype?, stream?) - for [0, 1) range
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "random.uniform requires at least 1 argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  
  try {
    mlx::core::Shape shape;
    mlx::core::Dtype dtype = mlx::core::float32;
    int shapeArgIdx = 0;
    bool hasLowHigh = false;

    // Check if first two arguments are numbers/arrays (low, high variant)
    // Note: first arg being an Array (JS array) means it's a shape, not a
    // scalar
    if (info.Length() >= 2 && info[0].IsNumber() &&
        (info[1].IsNumber() || info[1].IsObject())) {
      hasLowHigh = true;
      shapeArgIdx = 2;
    }

    // Parse shape
    if (static_cast<size_t>(info.Length()) <= static_cast<size_t>(shapeArgIdx) ||
        !info[shapeArgIdx].IsArray()) {
      Napi::TypeError::New(env, "shape must be an array")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto shapeArray = info[shapeArgIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < shapeArray.Length(); i++) {
      Napi::Value val = shapeArray[i];
      if (!val.IsNumber()) {
        Napi::TypeError::New(env, "shape elements must be numbers")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
      shape.push_back(val.As<Napi::Number>().Int32Value());
    }

    // Parse optional dtype
    size_t dtypeArgIdx = shapeArgIdx + 1;
    if (info.Length() > dtypeArgIdx && IsDtypeArg(env, info[dtypeArgIdx], *addon)) {
      dtype = MaybeParseDtype(env, info[dtypeArgIdx], mlx::core::float32, *addon);
      if (env.IsExceptionPending()) return env.Null();
    }

    // Parse stream
    auto streamArg = GetStreamArgument(info, dtypeArgIdx + 1);
    if (env.IsExceptionPending()) return env.Null();

    mlx::core::array result = [&]() -> mlx::core::array {
      if (hasLowHigh) {
        auto low = ToArray(env, info[0]);
        auto high = ToArray(env, info[1]);
        return mlx::core::random::uniform(
            low, high, shape, dtype, std::nullopt, streamArg);
      } else {
        return mlx::core::random::uniform(
            shape, dtype, std::nullopt, streamArg);
      }
    }();
    if (env.IsExceptionPending()) return env.Null();

    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(result)));
  } catch (const std::exception& e) {
    Napi::Error::New(env, std::string("random.uniform failed: ") + e.what())
        .ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value CPUSanity(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  try {
    // CPU stream demo: mirror the GPU example on CPU to prove the
    // binding+streams.
    auto result = [&]() -> Napi::Value {
      auto s2 = mlx::core::new_stream(mlx::core::Device::cpu);
      auto s3 = mlx::core::new_stream(mlx::core::Device::cpu);

      auto a = mlx::core::arange(1.f, 10.f, 1.f, mlx::core::float32, s2);
      auto b = mlx::core::arange(1.f, 10.f, 1.f, mlx::core::float32, s3);
      auto x = mlx::core::add(a, a, s2);
      auto y = mlx::core::add(b, b, s3);
      auto z = mlx::core::multiply(x, y);
      mlx::core::eval(z);

      const size_t length = z.size();
      auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(float));
      auto jsArray = Napi::Float32Array::New(env, length, buffer, 0);
      std::memcpy(buffer.Data(), z.data<float>(), length * sizeof(float));
      return jsArray;
    }();
    return result;
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value Normal(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (addon == nullptr) {
    Napi::Error::New(env, "AddonData missing for normal")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse shape (required)
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "normal expects a shape array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  
  // Parse optional dtype (default: float32)
  // Signature: normal(shape, dtype?, loc?, scale?, key?, stream?)
  auto dtype = mlx::core::float32;
  size_t argIndex = 1;
  if (info.Length() > argIndex && IsDtypeArg(env, info[argIndex], *addon)) {
    dtype = MaybeParseDtype(env, info[argIndex], dtype, *addon);
    if (env.IsExceptionPending()) {
      return env.Null();
    }
    argIndex++;
  }
  
  // Helper to check if arg is an MLX array
  auto isArray = [&](size_t idx) -> bool {
    if (info.Length() <= idx || !info[idx].IsObject()) {
      return false;
    }
    auto obj = info[idx].As<Napi::Object>();
    auto ctor = addon->array_constructor.Value();
    return !ctor.IsEmpty() && obj.InstanceOf(ctor);
  };
  
  // Helper to parse scalar or array
  auto parseScalarOrArray = [&](size_t idx) -> std::optional<mlx::core::array> {
    if (info.Length() <= idx || info[idx].IsUndefined() || info[idx].IsNull()) {
      return std::nullopt;
    }
    if (isArray(idx)) {
      auto obj = info[idx].As<Napi::Object>();
      const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
      if (wrapper) {
        return wrapper->tensor();
      }
    } else if (info[idx].IsNumber()) {
      double val = info[idx].As<Napi::Number>().DoubleValue();
      return mlx::core::array(val, dtype);
    }
    return std::nullopt;
  };
  
  // Parse optional loc (mean)
  std::optional<mlx::core::array> loc;
  if (info.Length() > argIndex && 
      (info[argIndex].IsNumber() || isArray(argIndex))) {
    loc = parseScalarOrArray(argIndex);
    argIndex++;
  }
  
  // Parse optional scale (std dev)
  std::optional<mlx::core::array> scale;
  if (info.Length() > argIndex && 
      (info[argIndex].IsNumber() || isArray(argIndex))) {
    scale = parseScalarOrArray(argIndex);
    argIndex++;
  }
  
  // Parse optional key
  std::optional<mlx::core::array> key;
  if (info.Length() > argIndex && isArray(argIndex)) {
    key = parseScalarOrArray(argIndex);
    argIndex++;
  }
  
  // If key not provided, use default KeySequence
  if (!key.has_value()) {
    key = mlx::core::random::KeySequence::default_().next();
  }
  
  // Parse optional stream
  auto streamArg = GetStreamArgument(info, argIndex);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  
  // Call mlx::core::random::normal
  auto result = std::make_shared<mlx::core::array>(
      mlx::core::random::normal(shape, dtype, loc, scale, key, streamArg));
  return WrapArray(env, result);
}

/**
 * Sparse initializer for neural network weights
 *
 * Creates a sparse matrix by:
 * 1. Filling with samples from a normal distribution
 * 2. Randomly setting a fraction of elements in each row to zero
 *
 * Note: The Python documentation says "per column" but the actual implementation
 * (both Python and this C++ version) applies sparsity per row: each row gets
 * num_zeros elements zeroed, where num_zeros = ceil(sparsity * num_columns).
 *
 * Args:
 *   - a: Input array (must be 2D)
 *   - sparsity: Fraction of columns to zero out in each row (0.0-1.0)
 *   - mean: Mean of normal distribution (default: 0.0)
 *   - std: Standard deviation (default: 1.0)
 *   - stream (optional)
 *
 * Returns: Array with same shape as input, sparsified normal distribution
 */
Napi::Value Sparse(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());

  // Ensure Metal is initialized
  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }

  // Validate argument count (at least 2: array and sparsity)
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "sparse requires at least 2 arguments: array and sparsity")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse input array
  auto* wrapper = UnwrapArray(env, info[0]);
  if (!wrapper) return env.Null();
  const auto& a = wrapper->tensor();

  // Validate 2D array
  if (a.ndim() != 2) {
    Napi::TypeError::New(env, "sparse: only tensors with 2 dimensions are supported")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse sparsity
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "sparsity must be a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  float sparsity = info[1].As<Napi::Number>().FloatValue();

  // Validate sparsity range
  if (sparsity < 0.0f || sparsity > 1.0f) {
    Napi::TypeError::New(env, "sparsity must be between 0 and 1")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse optional parameters
  float mean = 0.0f;
  float std = 1.0f;

  int next_arg = 2;

  // Check for mean parameter
  if (info.Length() > next_arg && info[next_arg].IsNumber()) {
    mean = info[next_arg].As<Napi::Number>().FloatValue();
    next_arg++;
  }

  // Check for std parameter
  if (info.Length() > next_arg && info[next_arg].IsNumber()) {
    std = info[next_arg].As<Napi::Number>().FloatValue();
    next_arg++;
  }

  // Parse stream (last argument if present and not a number)
  auto stream = GetStreamArgument(info, next_arg);
  if (env.IsExceptionPending()) return env.Null();

  try {
    // Get array dimensions
    int rows = a.shape(0);
    int cols = a.shape(1);
    
    // Calculate number of zeros per row
    // (Despite docs saying "per column", this is actually per row in the implementation)
    int num_zeros = static_cast<int>(std::ceil(sparsity * static_cast<float>(cols)));

    // Generate random order for each element (argsort of uniform random values)
    // This determines which elements to zero out
    auto order = mlx::core::argsort(
        mlx::core::random::uniform(a.shape(), mlx::core::float32, std::nullopt, stream),
        1,  // axis=1 (sort along columns for each row)
        stream
    );

    // Generate normal distribution with specified mean and std
    auto result = mlx::core::random::normal(
        a.shape(),
        a.dtype(),
        mlx::core::array(mean),
        mlx::core::array(std),
        std::nullopt,
        stream
    );

    // Evaluate both arrays to get actual data
    mlx::core::eval({order, result});
    
    // Note: MLX doesn't have easy advanced indexing in C++ (like Python's a[indices] = 0)
    // So we evaluate the arrays, modify the data in memory, and create a new array.
    // This is less efficient than pure MLX operations but necessary without scatter operations.
    
    // Get raw data pointers
    const int* order_ptr = order.data<int>();
    
    // Create a vector to hold the modified result
    std::vector<float> result_vec;
    result_vec.reserve(rows * cols);
    
    // Copy result data to vector based on dtype
    // TODO: Support other dtypes (float16, bfloat16, int types, etc.)
    // This requires template specialization or switch-case for each dtype's data<T>() call
    // and appropriate handling of zero values for each type.
    if (a.dtype() == mlx::core::float32) {
      const float* result_ptr = result.data<float>();
      result_vec.assign(result_ptr, result_ptr + rows * cols);
      
      // Zero out the selected elements based on argsort order
      // For each row, zero out the first num_zeros elements according to random order
      for (int row = 0; row < rows; row++) {
        for (int j = 0; j < num_zeros; j++) {
          int col = order_ptr[row * cols + j];
          result_vec[row * cols + col] = 0.0f;
        }
      }
      
      // Create new array from modified data
      auto final_result = mlx::core::array(
          result_vec.data(),
          {rows, cols},
          a.dtype()
      );
      
      return WrapArray(env, std::make_shared<mlx::core::array>(std::move(final_result)));
    } else {
      // For other dtypes, we need different handling
      // For now, just support float32
      Napi::TypeError::New(env, "sparse currently only supports float32 dtype")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
  } catch (const std::exception& e) {
    Napi::Error::New(env, std::string("sparse failed: ") + e.what())
        .ThrowAsJavaScriptException();
    return env.Null();
  }
}

// ---------------------------------------------------------------------------
// Reduction helper: parse optional axis argument into a vector of ints.
// When no axis is provided (or it's null/undefined), returns all axes [0..ndim).
// ---------------------------------------------------------------------------
std::vector<int> GetReduceAxes(
    Napi::Env env,
    const Napi::CallbackInfo& info,
    size_t axisIndex,
    const mlx::core::array& a) {
  if (info.Length() <= axisIndex || info[axisIndex].IsNull() ||
      info[axisIndex].IsUndefined()) {
    // No axis => reduce over all dims
    int ndim = a.ndim();
    std::vector<int> axes(ndim);
    std::iota(axes.begin(), axes.end(), 0);
    return axes;
  }
  return ParseAxisVector(env, info[axisIndex], "axis");
}

// ---------------------------------------------------------------------------
// Reduction ops: sum, mean, logsumexp
// Signature: (array, axis?, keepdims?, stream?) where axis is int|int[]|null
// ---------------------------------------------------------------------------
Napi::Value Sum(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "sum expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();

  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }

  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::sum(a, axes, keepdims, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Mean(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "mean expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();

  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }

  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::mean(a, axes, keepdims, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value LogSumExp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "logsumexp expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();

  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }

  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::logsumexp(a, axes, keepdims, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Min: (array, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Min(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "min expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();
  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::min(a, axes, keepdims, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Max: (array, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Max(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "max expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();
  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::max(a, axes, keepdims, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Prod: (array, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Prod(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "prod expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();
  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::prod(a, axes, keepdims, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Argmin: (array, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Argmin(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "argmin expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int argIdx = 1;
  int axis = 0;
  bool hasAxis = false;
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsNumber()) {
    axis = info[argIdx].As<Napi::Number>().Int32Value();
    hasAxis = true;
    argIdx++;
  }
  bool keepdims = false;
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsBoolean()) {
    keepdims = info[argIdx].As<Napi::Boolean>().Value();
    argIdx++;
  }
  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();
  std::shared_ptr<mlx::core::array> tensor;
  if (hasAxis) {
    tensor = std::make_shared<mlx::core::array>(
        mlx::core::argmin(a, axis, keepdims, streamArg));
  } else {
    tensor = std::make_shared<mlx::core::array>(
        mlx::core::argmin(a, keepdims, streamArg));
  }
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Argmax: (array, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Argmax(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "argmax expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int argIdx = 1;
  int axis = 0;
  bool hasAxis = false;
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsNumber()) {
    axis = info[argIdx].As<Napi::Number>().Int32Value();
    hasAxis = true;
    argIdx++;
  }
  bool keepdims = false;
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsBoolean()) {
    keepdims = info[argIdx].As<Napi::Boolean>().Value();
    argIdx++;
  }
  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();
  std::shared_ptr<mlx::core::array> tensor;
  if (hasAxis) {
    tensor = std::make_shared<mlx::core::array>(
        mlx::core::argmax(a, axis, keepdims, streamArg));
  } else {
    tensor = std::make_shared<mlx::core::array>(
        mlx::core::argmax(a, keepdims, streamArg));
  }
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Std: (array, axis?, keepdims?, ddof?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Std(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "std expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  // Parse optional axes
  std::optional<std::vector<int>> axes = std::nullopt;
  int argIdx = 1;
  if (info.Length() > argIdx && !info[argIdx].IsUndefined() &&
      !info[argIdx].IsNull()) {
    if (info[argIdx].IsNumber()) {
      axes = std::vector<int>{info[argIdx].As<Napi::Number>().Int32Value()};
      argIdx++;
    } else if (info[argIdx].IsArray()) {
      auto arr = info[argIdx].As<Napi::Array>();
      std::vector<int> axesVec;
      axesVec.reserve(arr.Length());
      for (uint32_t i = 0; i < arr.Length(); ++i) {
        axesVec.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
      }
      axes = std::move(axesVec);
      argIdx++;
    } else if (info[argIdx].IsBoolean()) {
      // No axes, this is keepdims — leave axes as nullopt
    } else {
      argIdx++;
    }
  }

  // Parse optional keepdims
  bool keepdims = false;
  if (info.Length() > static_cast<size_t>(argIdx) &&
      info[argIdx].IsBoolean()) {
    keepdims = info[argIdx].As<Napi::Boolean>().Value();
    argIdx++;
  }

  // Parse optional ddof
  int ddof = 0;
  if (info.Length() > static_cast<size_t>(argIdx) &&
      info[argIdx].IsNumber()) {
    ddof = info[argIdx].As<Napi::Number>().Int32Value();
    argIdx++;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  mlx::core::array result = [&]() {
    if (axes.has_value()) {
      return mlx::core::std(a, axes.value(), keepdims, ddof, streamArg);
    } else {
      return mlx::core::std(a, keepdims, ddof, streamArg);
    }
  }();

  return WrapArray(env,
                   std::make_shared<mlx::core::array>(std::move(result)));
}

// ---------------------------------------------------------------------------
// LogCumSumExp: (array, axis, reverse?, stream?)
// ---------------------------------------------------------------------------
Napi::Value LogCumSumExp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "logcumsumexp expects at least 2 arguments (array, axis)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int axis = info[1].As<Napi::Number>().Int32Value();
  int argIdx = 2;
  bool reverse = false;
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsBoolean()) {
    reverse = info[argIdx].As<Napi::Boolean>().Value();
    argIdx++;
  }
  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::logcumsumexp(a, axis, reverse, /*inclusive=*/true, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Trace: (array, offset?, axis1?, axis2?, dtype?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Trace(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "trace expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto& data = *static_cast<mlx::node::AddonData*>(info.Data());
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  int argIdx = 1;
  int offset = 0;
  int axis1 = 0;
  int axis2 = 1;
  bool hasDtype = false;
  mlx::core::Dtype dtype = mlx::core::float32;

  // Parse offset
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsNumber()) {
    offset = info[argIdx].As<Napi::Number>().Int32Value();
    argIdx++;
  }
  // Parse axis1
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsNumber()) {
    axis1 = info[argIdx].As<Napi::Number>().Int32Value();
    argIdx++;
  }
  // Parse axis2
  if (static_cast<size_t>(argIdx) < info.Length() && info[argIdx].IsNumber()) {
    axis2 = info[argIdx].As<Napi::Number>().Int32Value();
    argIdx++;
  }
  // Parse optional dtype
  if (static_cast<size_t>(argIdx) < info.Length() &&
      IsDtypeArg(env, info[argIdx], data)) {
    dtype = MaybeParseDtype(env, info[argIdx], mlx::core::float32, data);
    hasDtype = true;
    argIdx++;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  mlx::core::array result = [&]() {
    if (hasDtype) {
      return mlx::core::trace(a, offset, axis1, axis2, dtype, streamArg);
    } else {
      return mlx::core::trace(a, offset, axis1, axis2, streamArg);
    }
  }();

  return WrapArray(env,
                   std::make_shared<mlx::core::array>(std::move(result)));
}

// ---------------------------------------------------------------------------
// Softmax: (array, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Softmax(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "softmax expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::softmax(a, axes, false, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Binary op: logaddexp(a, b, stream?)
// ---------------------------------------------------------------------------
Napi::Value LogAddExp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "logaddexp expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::logaddexp(a, b, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// clip(a, a_min?, a_max?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Clip(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "clip expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  std::optional<mlx::core::array> min_val = std::nullopt;
  std::optional<mlx::core::array> max_val = std::nullopt;

  if (info.Length() > 1 && !info[1].IsNull() && !info[1].IsUndefined()) {
    min_val = ToArray(env, info[1]);
    if (env.IsExceptionPending()) return env.Null();
  }
  if (info.Length() > 2 && !info[2].IsNull() && !info[2].IsUndefined()) {
    max_val = ToArray(env, info[2]);
    if (env.IsExceptionPending()) return env.Null();
  }

  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::clip(a, min_val, max_val, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// take_along_axis(a, indices, axis, stream?)
// ---------------------------------------------------------------------------
Napi::Value TakeAlongAxis(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "take_along_axis expects (array, indices, axis)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* a = UnwrapArray(env, info[0]);
  auto* indices = UnwrapArray(env, info[1]);
  if (env.IsExceptionPending() || a == nullptr || indices == nullptr) {
    return env.Null();
  }

  int axis = 0;
  if (info[2].IsNull() || info[2].IsUndefined()) {
    // None axis → flatten to 1D first
    auto flat = mlx::core::reshape(a->tensor(), {-1});
    auto streamArg = GetStreamArgument(info, 3);
    if (env.IsExceptionPending()) return env.Null();
    auto tensor = std::make_shared<mlx::core::array>(
        mlx::core::take_along_axis(flat, indices->tensor(), 0, streamArg));
    return WrapArray(env, tensor);
  }

  if (!info[2].IsNumber()) {
    Napi::TypeError::New(env, "take_along_axis axis must be an integer")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  axis = info[2].As<Napi::Number>().Int32Value();

  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::take_along_axis(a->tensor(), indices->tensor(), axis, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// Unary ops: log1p, negative, reciprocal
// ---------------------------------------------------------------------------
Napi::Value Log1p(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "log1p expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::log1p(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Negative(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "negative expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::negative(a, streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Reciprocal(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "reciprocal expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::reciprocal(a, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// expand_dims(a, axis, stream?) — axis is int or int[]
// ---------------------------------------------------------------------------
Napi::Value ExpandDims(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "expand_dims expects (array, axis)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* a = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || a == nullptr) return env.Null();

  if (info[1].IsNumber()) {
    int axis = info[1].As<Napi::Number>().Int32Value();
    auto streamArg = GetStreamArgument(info, 2);
    if (env.IsExceptionPending()) return env.Null();
    auto tensor = std::make_shared<mlx::core::array>(
        mlx::core::expand_dims(a->tensor(), axis, streamArg));
    return WrapArray(env, tensor);
  }

  auto axes = ParseAxisVector(env, info[1], "axis");
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::expand_dims(a->tensor(), axes, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// squeeze(a, axis?, stream?) — axis is int|int[]|null
// ---------------------------------------------------------------------------
Napi::Value Squeeze(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "squeeze expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* a = UnwrapArray(env, info[0]);
  if (env.IsExceptionPending() || a == nullptr) return env.Null();

  // No axis or null/undefined → squeeze all size-1 dims
  if (info.Length() < 2 || info[1].IsNull() || info[1].IsUndefined()) {
    auto streamArg = GetStreamArgument(info, 2);
    if (env.IsExceptionPending()) return env.Null();
    auto tensor = std::make_shared<mlx::core::array>(
        mlx::core::squeeze(a->tensor(), streamArg));
    return WrapArray(env, tensor);
  }

  if (info[1].IsNumber()) {
    int axis = info[1].As<Napi::Number>().Int32Value();
    auto streamArg = GetStreamArgument(info, 2);
    if (env.IsExceptionPending()) return env.Null();
    auto tensor = std::make_shared<mlx::core::array>(
        mlx::core::squeeze(a->tensor(), axis, streamArg));
    return WrapArray(env, tensor);
  }

  auto axes = ParseAxisVector(env, info[1], "axis");
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::squeeze(a->tensor(), axes, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// concatenate(arrays, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Concatenate(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "concatenate expects an array of arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto jsArr = info[0].As<Napi::Array>();
  std::vector<mlx::core::array> arrays;
  arrays.reserve(jsArr.Length());
  for (uint32_t i = 0; i < jsArr.Length(); ++i) {
    auto* w = UnwrapArray(env, jsArr.Get(i));
    if (env.IsExceptionPending() || w == nullptr) return env.Null();
    arrays.push_back(w->tensor());
  }

  // Default axis = 0
  int axis = 0;
  size_t streamIdx = 1;
  if (info.Length() > 1 && info[1].IsNumber()) {
    axis = info[1].As<Napi::Number>().Int32Value();
    streamIdx = 2;
  } else if (info.Length() > 1 && (info[1].IsNull() || info[1].IsUndefined())) {
    // null axis → concatenate flattened
    streamIdx = 2;
    auto streamArg = GetStreamArgument(info, streamIdx);
    if (env.IsExceptionPending()) return env.Null();
    auto tensor = std::make_shared<mlx::core::array>(
        mlx::core::concatenate(arrays, streamArg));
    return WrapArray(env, tensor);
  }

  auto streamArg = GetStreamArgument(info, streamIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::concatenate(arrays, axis, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// linalg.norm(a, ord?, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Norm(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (info.Length() < 1) {
    Napi::TypeError::New(env, "norm requires at least 1 argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  // Parse optional ord (index 1): null/undefined → no ord, number → double,
  // string → string ord
  bool hasOrd = info.Length() > 1 && !info[1].IsNull() && !info[1].IsUndefined();
  bool ordIsString = hasOrd && info[1].IsString();
  double ordNum = 0;
  std::string ordStr;
  if (hasOrd) {
    if (ordIsString) {
      ordStr = info[1].As<Napi::String>().Utf8Value();
    } else {
      ordNum = info[1].As<Napi::Number>().DoubleValue();
    }
  }

  // Parse optional axis (index 2): null/undefined → nullopt, number → single
  // axis, array → vector of axes
  std::optional<std::vector<int>> axis = std::nullopt;
  if (info.Length() > 2 && !info[2].IsNull() && !info[2].IsUndefined()) {
    if (info[2].IsArray()) {
      auto arr = info[2].As<Napi::Array>();
      std::vector<int> axes;
      axes.reserve(arr.Length());
      for (uint32_t i = 0; i < arr.Length(); ++i) {
        axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
      }
      axis = axes;
    } else {
      axis = std::vector<int>{info[2].As<Napi::Number>().Int32Value()};
    }
  }

  // Parse optional keepdims (index 3)
  bool keepdims = false;
  if (info.Length() > 3 && !info[3].IsNull() && !info[3].IsUndefined()) {
    keepdims = info[3].As<Napi::Boolean>().Value();
  }

  // Parse optional stream (index 4)
  auto streamArg = GetStreamArgument(info, 4);
  if (env.IsExceptionPending()) return env.Null();

  auto compute = [&]() -> mlx::core::array {
    if (hasOrd) {
      if (ordIsString) {
        return mlx::core::linalg::norm(a, ordStr, axis, keepdims, streamArg);
      } else {
        return mlx::core::linalg::norm(a, ordNum, axis, keepdims, streamArg);
      }
    } else {
      return mlx::core::linalg::norm(a, axis, keepdims, streamArg);
    }
  };

  auto tensor = std::make_shared<mlx::core::array>(compute());
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// sigmoid(a, stream?) — element-wise logistic sigmoid
// ---------------------------------------------------------------------------
Napi::Value Sigmoid(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "sigmoid expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::sigmoid(a, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// erf(a, stream?) — Gauss error function
// ---------------------------------------------------------------------------
Napi::Value Erf(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "erf expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::erf(a, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// tanh(a, stream?) — hyperbolic tangent
// ---------------------------------------------------------------------------
Napi::Value Tanh(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "tanh expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto tensor =
      std::make_shared<mlx::core::array>(mlx::core::tanh(a, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// split(a, indices_or_sections, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Split(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "split expects at least 2 arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  // Determine axis (optional, default 0)
  int axisArgIdx = 2;
  int axis = 0;
  if (info.Length() > 2 && !info[2].IsNull() && !info[2].IsUndefined() &&
      info[2].IsNumber()) {
    axis = info[2].As<Napi::Number>().Int32Value();
    axisArgIdx = 3;
  }

  auto streamArg = GetStreamArgument(info, axisArgIdx);
  if (env.IsExceptionPending()) return env.Null();

  std::vector<mlx::core::array> results;

  if (info[1].IsNumber()) {
    // split(a, num_splits, axis, stream)
    int numSplits = info[1].As<Napi::Number>().Int32Value();
    results = mlx::core::split(a, numSplits, axis, streamArg);
  } else if (info[1].IsArray()) {
    // split(a, indices, axis, stream)
    auto arr = info[1].As<Napi::Array>();
    mlx::core::Shape indices;
    indices.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
      indices.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    results = mlx::core::split(a, indices, axis, streamArg);
  } else {
    Napi::TypeError::New(env,
        "split: second argument must be a number or array of indices")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Return as JS array of wrapped MLXArrays
  auto jsArr = Napi::Array::New(env, results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    auto tensor = std::make_shared<mlx::core::array>(std::move(results[i]));
    jsArr.Set(static_cast<uint32_t>(i), WrapArray(env, tensor));
  }
  return jsArr;
}

// ---------------------------------------------------------------------------
// addmm(c, a, b, alpha?, beta?, stream?) — fused alpha*(a@b) + beta*c
// ---------------------------------------------------------------------------
Napi::Value Addmm(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "addmm expects at least 3 arguments (c, a, b)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto c = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto a = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();

  float alpha = 1.0f;
  float beta = 1.0f;
  int streamIdx = 3;

  if (info.Length() > 3 && info[3].IsNumber()) {
    alpha = info[3].As<Napi::Number>().FloatValue();
    streamIdx = 4;
  }
  if (info.Length() > 4 && info[4].IsNumber()) {
    beta = info[4].As<Napi::Number>().FloatValue();
    streamIdx = 5;
  }

  auto streamArg = GetStreamArgument(info, streamIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::addmm(c, a, b, alpha, beta, streamArg));
  return WrapArray(env, tensor);
}

// ---------------------------------------------------------------------------
// var(a, axes?, keepdims?, ddof?, stream?) — variance reduction
// ---------------------------------------------------------------------------
Napi::Value Var(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "var expects at least 1 argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  // Parse optional axes
  std::optional<std::vector<int>> axes = std::nullopt;
  int argIdx = 1;
  if (info.Length() > argIdx && !info[argIdx].IsUndefined() &&
      !info[argIdx].IsNull()) {
    if (info[argIdx].IsNumber()) {
      axes = std::vector<int>{info[argIdx].As<Napi::Number>().Int32Value()};
      argIdx++;
    } else if (info[argIdx].IsArray()) {
      auto arr = info[argIdx].As<Napi::Array>();
      std::vector<int> axesVec;
      axesVec.reserve(arr.Length());
      for (uint32_t i = 0; i < arr.Length(); ++i) {
        axesVec.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
      }
      axes = std::move(axesVec);
      argIdx++;
    } else if (info[argIdx].IsBoolean()) {
      // No axes, this is keepdims — leave axes as nullopt
    } else {
      argIdx++;
    }
  }

  // Parse optional keepdims
  bool keepdims = false;
  if (info.Length() > static_cast<size_t>(argIdx) &&
      info[argIdx].IsBoolean()) {
    keepdims = info[argIdx].As<Napi::Boolean>().Value();
    argIdx++;
  }

  // Parse optional ddof
  int ddof = 0;
  if (info.Length() > static_cast<size_t>(argIdx) &&
      info[argIdx].IsNumber()) {
    ddof = info[argIdx].As<Napi::Number>().Int32Value();
    argIdx++;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  mlx::core::array result = [&]() {
    if (axes.has_value()) {
      return mlx::core::var(a, axes.value(), keepdims, ddof, streamArg);
    } else {
      return mlx::core::var(a, keepdims, ddof, streamArg);
    }
  }();

  return WrapArray(env,
                   std::make_shared<mlx::core::array>(std::move(result)));
}

// ---------------------------------------------------------------------------
// random.bernoulli(p?, shape?, key?, stream?) — Bernoulli samples
// ---------------------------------------------------------------------------
Napi::Value Bernoulli(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());

  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }

  try {
    // bernoulli(p?, shape?, key?, stream?)
    // p defaults to 0.5, shape defaults to p.shape or {}
    mlx::core::array p = mlx::core::array(0.5f);
    mlx::core::Shape shape;
    bool hasShape = false;
    size_t argIdx = 0;

    // Parse optional p (number or MLXArray)
    if (info.Length() > argIdx && !info[argIdx].IsUndefined() &&
        !info[argIdx].IsNull()) {
      if (info[argIdx].IsNumber()) {
        p = mlx::core::array(
            static_cast<float>(info[argIdx].As<Napi::Number>().DoubleValue()));
        argIdx++;
      } else if (info[argIdx].IsObject() && !info[argIdx].IsArray()) {
        // Could be MLXArray
        auto obj = info[argIdx].As<Napi::Object>();
        auto ctor = addon->array_constructor.Value();
        if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
          auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
          if (wrapper) {
            p = wrapper->tensor();
          }
          argIdx++;
        }
      }
    }

    // Parse optional shape
    if (info.Length() > argIdx && info[argIdx].IsArray()) {
      auto shapeArr = info[argIdx].As<Napi::Array>();
      shape.reserve(shapeArr.Length());
      for (uint32_t i = 0; i < shapeArr.Length(); ++i) {
        shape.push_back(shapeArr.Get(i).As<Napi::Number>().Int32Value());
      }
      hasShape = true;
      argIdx++;
    }

    // Parse optional key
    std::optional<mlx::core::array> key;
    if (info.Length() > argIdx && info[argIdx].IsObject() &&
        !info[argIdx].IsArray()) {
      auto obj = info[argIdx].As<Napi::Object>();
      auto ctor = addon->array_constructor.Value();
      if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
        auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
        if (wrapper) {
          key = wrapper->tensor();
        }
        argIdx++;
      }
    }

    if (!key.has_value()) {
      key = mlx::core::random::KeySequence::default_().next();
    }

    auto streamArg = GetStreamArgument(info, argIdx);
    if (env.IsExceptionPending()) return env.Null();

    mlx::core::array result = [&]() {
      if (hasShape) {
        return mlx::core::random::bernoulli(p, shape, key, streamArg);
      } else {
        return mlx::core::random::bernoulli(p, key, streamArg);
      }
    }();

    return WrapArray(env,
                     std::make_shared<mlx::core::array>(std::move(result)));
  } catch (const std::exception& e) {
    Napi::Error::New(env,
                     std::string("random.bernoulli failed: ") + e.what())
        .ThrowAsJavaScriptException();
    return env.Null();
  }
}

// ---------------------------------------------------------------------------
// stack(arrays, axis?, stream?) — stack arrays along a new axis
// ---------------------------------------------------------------------------
Napi::Value Stack(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "stack expects an array of arrays as first arg")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto jsArr = info[0].As<Napi::Array>();
  std::vector<mlx::core::array> arrays;
  arrays.reserve(jsArr.Length());
  for (uint32_t i = 0; i < jsArr.Length(); ++i) {
    auto val = jsArr.Get(i);
    if (!val.IsObject()) {
      Napi::TypeError::New(env, "stack: all elements must be arrays")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
        val.As<Napi::Object>());
    if (!wrapper) {
      Napi::TypeError::New(env, "stack: invalid array element")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    arrays.push_back(wrapper->tensor());
  }

  int axis = 0;
  int argIdx = 1;
  if (info.Length() > 1 && info[1].IsNumber()) {
    axis = info[1].As<Napi::Number>().Int32Value();
    argIdx = 2;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(
      mlx::core::stack(arrays, axis, streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// conv1d(input, weight, stride?, padding?, dilation?, groups?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Conv1d(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv1d expects at least 2 arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto* inputW = UnwrapArray(env, info[0]);
  auto* weightW = UnwrapArray(env, info[1]);
  if (env.IsExceptionPending() || !inputW || !weightW) return env.Null();

  int stride = 1, padding = 0, dilation = 1, groups = 1;
  int argIdx = 2;

  if (info.Length() > argIdx && info[argIdx].IsNumber()) {
    stride = info[argIdx++].As<Napi::Number>().Int32Value();
  }
  if (info.Length() > argIdx && info[argIdx].IsNumber()) {
    padding = info[argIdx++].As<Napi::Number>().Int32Value();
  }
  if (info.Length() > argIdx && info[argIdx].IsNumber()) {
    dilation = info[argIdx++].As<Napi::Number>().Int32Value();
  }
  if (info.Length() > argIdx && info[argIdx].IsNumber()) {
    groups = info[argIdx++].As<Napi::Number>().Int32Value();
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(mlx::core::conv1d(
      inputW->tensor(), weightW->tensor(), stride, padding, dilation, groups,
      streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// conv2d(input, weight, stride?, padding?, dilation?, groups?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Conv2d(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv2d expects at least 2 arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto* inputW = UnwrapArray(env, info[0]);
  auto* weightW = UnwrapArray(env, info[1]);
  if (env.IsExceptionPending() || !inputW || !weightW) return env.Null();

  // Parse stride, padding, dilation as pairs
  auto parsePair = [&](int idx, std::pair<int,int> def) -> std::pair<int,int> {
    if (info.Length() <= static_cast<size_t>(idx)) return def;
    if (info[idx].IsNumber()) {
      int v = info[idx].As<Napi::Number>().Int32Value();
      return {v, v};
    }
    if (info[idx].IsArray()) {
      auto arr = info[idx].As<Napi::Array>();
      if (arr.Length() >= 2) {
        return {
          arr.Get((uint32_t)0).As<Napi::Number>().Int32Value(),
          arr.Get((uint32_t)1).As<Napi::Number>().Int32Value()
        };
      }
    }
    return def;
  };

  int argIdx = 2;
  auto stride = parsePair(argIdx, {1, 1});
  if (info.Length() > static_cast<size_t>(argIdx) && (info[argIdx].IsNumber() || info[argIdx].IsArray())) argIdx++;
  auto padding = parsePair(argIdx, {0, 0});
  if (info.Length() > static_cast<size_t>(argIdx) && (info[argIdx].IsNumber() || info[argIdx].IsArray())) argIdx++;
  auto dilation = parsePair(argIdx, {1, 1});
  if (info.Length() > static_cast<size_t>(argIdx) && (info[argIdx].IsNumber() || info[argIdx].IsArray())) argIdx++;

  int groups = 1;
  if (info.Length() > static_cast<size_t>(argIdx) && info[argIdx].IsNumber()) {
    groups = info[argIdx++].As<Napi::Number>().Int32Value();
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(mlx::core::conv2d(
      inputW->tensor(), weightW->tensor(),
      stride, padding, dilation, groups, streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// conv3d(input, weight, stride?, padding?, dilation?, groups?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Conv3d(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv3d expects at least 2 arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto* inputW = UnwrapArray(env, info[0]);
  auto* weightW = UnwrapArray(env, info[1]);
  if (env.IsExceptionPending() || !inputW || !weightW) return env.Null();

  auto parseTuple3 = [&](int idx,
                         std::tuple<int,int,int> def) -> std::tuple<int,int,int> {
    if (info.Length() <= static_cast<size_t>(idx)) return def;
    if (info[idx].IsNumber()) {
      int v = info[idx].As<Napi::Number>().Int32Value();
      return {v, v, v};
    }
    if (info[idx].IsArray()) {
      auto arr = info[idx].As<Napi::Array>();
      if (arr.Length() >= 3) {
        return {
          arr.Get((uint32_t)0).As<Napi::Number>().Int32Value(),
          arr.Get((uint32_t)1).As<Napi::Number>().Int32Value(),
          arr.Get((uint32_t)2).As<Napi::Number>().Int32Value()
        };
      }
    }
    return def;
  };

  int argIdx = 2;
  auto stride = parseTuple3(argIdx, {1, 1, 1});
  if (info.Length() > static_cast<size_t>(argIdx) && (info[argIdx].IsNumber() || info[argIdx].IsArray())) argIdx++;
  auto padding = parseTuple3(argIdx, {0, 0, 0});
  if (info.Length() > static_cast<size_t>(argIdx) && (info[argIdx].IsNumber() || info[argIdx].IsArray())) argIdx++;
  auto dilation = parseTuple3(argIdx, {1, 1, 1});
  if (info.Length() > static_cast<size_t>(argIdx) && (info[argIdx].IsNumber() || info[argIdx].IsArray())) argIdx++;

  // groups (int, default 1)
  int groups = 1;
  if (info.Length() > static_cast<size_t>(argIdx) && info[argIdx].IsNumber()) {
    groups = info[argIdx].As<Napi::Number>().Int32Value();
    argIdx++;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(mlx::core::conv3d(
      inputW->tensor(), weightW->tensor(),
      stride, padding, dilation, groups, streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// take(a, indices, axis?, stream?) — gather elements by index
// ---------------------------------------------------------------------------
Napi::Value Take(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "take expects at least 2 arguments (a, indices)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto indices = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  // Ensure indices are integral (cast from float if needed)
  if (!mlx::core::issubdtype(indices.dtype(), mlx::core::integer)) {
    indices = mlx::core::astype(indices, mlx::core::int32);
  }

  int argIdx = 2;
  std::optional<int> axis;
  if (info.Length() > 2 && info[2].IsNumber()) {
    axis = info[2].As<Napi::Number>().Int32Value();
    argIdx = 3;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  mlx::core::array result = [&]() {
    if (axis.has_value()) {
      return mlx::core::take(a, indices, axis.value(), streamArg);
    } else {
      return mlx::core::take(a, indices, streamArg);
    }
  }();

  return WrapArray(env,
                   std::make_shared<mlx::core::array>(std::move(result)));
}

// ---------------------------------------------------------------------------
// pad(a, pad_width, pad_value?, stream?)
// pad_width is an array of [low, high] pairs, one per axis
// ---------------------------------------------------------------------------
Napi::Value Pad(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "pad expects at least 2 arguments (a, pad_width)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  // Parse pad_width: array of [low, high] pairs
  std::vector<std::pair<int, int>> pad_width;
  if (info[1].IsArray()) {
    auto arr = info[1].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); i++) {
      auto pair = arr.Get(i).As<Napi::Array>();
      int low = pair.Get(static_cast<uint32_t>(0)).As<Napi::Number>().Int32Value();
      int high = pair.Get(static_cast<uint32_t>(1)).As<Napi::Number>().Int32Value();
      pad_width.push_back({low, high});
    }
  }

  int argIdx = 2;
  mlx::core::array pad_value = mlx::core::array(0);
  if (static_cast<size_t>(argIdx) < info.Length() && !info[argIdx].IsUndefined() && !info[argIdx].IsNull()) {
    // Accept number or MLXArray for pad_value
    if (info[argIdx].IsNumber()) {
      pad_value = mlx::core::array(info[argIdx].As<Napi::Number>().DoubleValue());
      argIdx++;
    } else if (info[argIdx].IsObject()) {
      pad_value = ToArray(env, info[argIdx]);
      if (env.IsExceptionPending()) return env.Null();
      argIdx++;
    }
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(
      mlx::core::pad(a, pad_width, pad_value, "constant", streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// slice(a, start, stop, strides?, stream?) — extract a sub-array
// ---------------------------------------------------------------------------
Napi::Value Astype(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "astype expects array and dtype")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto a = ToArray(env, info[0]);
    if (env.IsExceptionPending()) return env.Null();
    auto dtype = MaybeParseDtype(env, info[1], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    auto streamArg = GetStreamArgument(info, 2);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env,
        std::make_shared<mlx::core::array>(
            mlx::core::astype(a, dtype, streamArg)));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value Slice(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "slice expects at least 3 arguments (a, start, stop)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  auto parseShape = [&](Napi::Value val) -> mlx::core::Shape {
    mlx::core::Shape s;
    if (val.IsArray()) {
      auto arr = val.As<Napi::Array>();
      for (uint32_t i = 0; i < arr.Length(); i++) {
        s.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
      }
    }
    return s;
  };

  auto start = parseShape(info[1]);
  auto stop = parseShape(info[2]);

  int argIdx = 3;
  mlx::core::Shape strides;
  if (info.Length() > 3 && info[3].IsArray()) {
    strides = parseShape(info[3]);
    argIdx = 4;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  mlx::core::array result = [&]() {
    if (strides.empty()) {
      return mlx::core::slice(a, start, stop, streamArg);
    } else {
      return mlx::core::slice(a, start, stop, strides, streamArg);
    }
  }();

  return WrapArray(env, std::make_shared<mlx::core::array>(std::move(result)));
}

// ---------------------------------------------------------------------------
// as_strided(a, shape, strides, offset, stream?)
// ---------------------------------------------------------------------------
Napi::Value AsStrided(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 4) {
    Napi::TypeError::New(env,
        "as_strided expects 4 arguments (a, shape, strides, offset)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  auto parseShape = [&](Napi::Value val) -> mlx::core::Shape {
    mlx::core::Shape s;
    if (val.IsArray()) {
      auto arr = val.As<Napi::Array>();
      for (uint32_t i = 0; i < arr.Length(); i++) {
        s.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
      }
    }
    return s;
  };

  auto shape = parseShape(info[1]);

  // Parse strides as Strides (SmallVector<int64_t>)
  mlx::core::Strides strides;
  if (info[2].IsArray()) {
    auto arr = info[2].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); i++) {
      strides.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
    }
  }

  size_t offset = 0;
  if (info[3].IsNumber()) {
    offset = static_cast<size_t>(info[3].As<Napi::Number>().Int64Value());
  }

  auto streamArg = GetStreamArgument(info, 4);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(
      mlx::core::as_strided(a, shape, strides, offset, streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// scaled_dot_product_attention(q, k, v, scale, mask?, stream?)
// ---------------------------------------------------------------------------
Napi::Value ScaledDotProductAttention(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 4) {
    Napi::TypeError::New(env,
        "scaled_dot_product_attention expects at least 4 arguments (q, k, v, scale)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto* qW = UnwrapArray(env, info[0]);
  auto* kW = UnwrapArray(env, info[1]);
  auto* vW = UnwrapArray(env, info[2]);
  if (env.IsExceptionPending() || !qW || !kW || !vW) return env.Null();

  float scale = info[3].As<Napi::Number>().FloatValue();

  int argIdx = 4;
  std::string mask_mode = "";
  std::vector<mlx::core::array> mask_arrs;

  // Optional mask: can be an MLXArray or the string "causal"
  if (static_cast<size_t>(argIdx) < info.Length() && !info[argIdx].IsNull() && !info[argIdx].IsUndefined()) {
    if (info[argIdx].IsString()) {
      mask_mode = info[argIdx].As<Napi::String>().Utf8Value();
      argIdx++;
    } else if (info[argIdx].IsObject()) {
      auto* maskW = UnwrapArray(env, info[argIdx]);
      if (maskW && !env.IsExceptionPending()) {
        mask_mode = "array";
        mask_arrs.push_back(maskW->tensor());
        argIdx++;
      }
    }
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(
      mlx::core::fast::scaled_dot_product_attention(
          qW->tensor(), kW->tensor(), vW->tensor(),
          scale, mask_mode, mask_arrs, streamArg));
  return WrapArray(env, result);
}

// ---------------------------------------------------------------------------
// number_of_elements(a, axes?, inverted?, dtype?, stream?)
// ---------------------------------------------------------------------------
Napi::Value NumberOfElements(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "number_of_elements expects at least 1 argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  std::vector<int> axes;
  bool inverted = false;
  mlx::core::Dtype dtype = mlx::core::int32;
  int argIdx = 1;

  // axes
  if (info.Length() > static_cast<size_t>(argIdx) && info[argIdx].IsArray()) {
    auto arr = info[argIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); i++) {
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    argIdx++;
  }

  // inverted
  if (info.Length() > static_cast<size_t>(argIdx) && info[argIdx].IsBoolean()) {
    inverted = info[argIdx].As<Napi::Boolean>().Value();
    argIdx++;
  }

  auto streamArg = GetStreamArgument(info, argIdx);
  if (env.IsExceptionPending()) return env.Null();

  auto result = std::make_shared<mlx::core::array>(
      mlx::core::number_of_elements(a, axes, inverted, dtype, streamArg));
  return WrapArray(env, result);
}

/**
 * Import a function from a .mlxfn file.
 *
 * Returns a callable JavaScript function that can be invoked with:
 * - Positional array arguments: fn(a, b, c)
 * - A single array/list of arrays: fn([a, b, c])
 * - A single object/dict of arrays: fn({x: a, y: b})
 * - Combined: fn([a, b], {x: c, y: d})
 *
 * The returned function always returns a tuple (array) of output arrays.
 *
 * Args:
 *   - file (string): Path to the .mlxfn file
 *
 * Returns: A JavaScript function wrapping the imported MLX function
 */
Napi::Value ImportFunction(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());

  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse file path argument
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "import_function expects a string file path")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string file_path = info[0].As<Napi::String>().Utf8Value();

  try {
    // Import the function from file
    auto imported_fn = mlx::core::import_function(file_path);

    // Create a shared pointer to keep the imported function alive
    auto fn_ptr = std::make_shared<mlx::core::ImportedFunction>(std::move(imported_fn));

    // Create a JavaScript function that captures the imported function
    auto js_function = Napi::Function::New(
        env,
        [fn_ptr, addon](const Napi::CallbackInfo& call_info) -> Napi::Value {
          auto call_env = call_info.Env();

          try {
            mlx::core::Args args;
            mlx::core::Kwargs kwargs;

            // Parse arguments based on how they're provided
            if (call_info.Length() == 0) {
              // No arguments
            } else if (call_info.Length() == 1) {
              // Check if it's an array, object, or single MLX array
              if (call_info[0].IsArray()) {
                // Single array argument containing MLX arrays
                auto arr = call_info[0].As<Napi::Array>();
                for (uint32_t i = 0; i < arr.Length(); i++) {
                  auto elem = arr.Get(i);
                  if (elem.IsObject()) {
                    auto obj = elem.As<Napi::Object>();
                    auto ctor = addon->array_constructor.Value();
                    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
                      const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
                      if (wrapper) {
                        args.push_back(wrapper->tensor());
                      }
                    }
                  }
                }
              } else if (call_info[0].IsObject()) {
                auto obj = call_info[0].As<Napi::Object>();
                auto ctor = addon->array_constructor.Value();

                // Check if it's an MLX array
                if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
                  const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
                  if (wrapper) {
                    args.push_back(wrapper->tensor());
                  }
                } else {
                  // It's a dictionary/object of arrays
                  auto prop_names = obj.GetPropertyNames();
                  for (uint32_t i = 0; i < prop_names.Length(); i++) {
                    auto key = prop_names.Get(i);
                    if (key.IsString()) {
                      auto key_str = key.As<Napi::String>().Utf8Value();
                      auto val = obj.Get(key);
                      if (val.IsObject()) {
                        auto val_obj = val.As<Napi::Object>();
                        if (!ctor.IsEmpty() && val_obj.InstanceOf(ctor)) {
                          const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(val_obj);
                          if (wrapper) {
                            kwargs.insert_or_assign(key_str, wrapper->tensor());
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else if (call_info.Length() == 2) {
              // Two arguments: args and kwargs
              // First argument should be an array
              if (call_info[0].IsArray()) {
                auto arr = call_info[0].As<Napi::Array>();
                for (uint32_t i = 0; i < arr.Length(); i++) {
                  auto elem = arr.Get(i);
                  if (elem.IsObject()) {
                    auto obj = elem.As<Napi::Object>();
                    auto ctor = addon->array_constructor.Value();
                    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
                      const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
                      if (wrapper) {
                        args.push_back(wrapper->tensor());
                      }
                    }
                  }
                }
              }

              // Second argument should be an object (kwargs)
              if (call_info[1].IsObject()) {
                auto obj = call_info[1].As<Napi::Object>();
                auto ctor = addon->array_constructor.Value();
                auto prop_names = obj.GetPropertyNames();
                for (uint32_t i = 0; i < prop_names.Length(); i++) {
                  auto key = prop_names.Get(i);
                  if (key.IsString()) {
                    auto key_str = key.As<Napi::String>().Utf8Value();
                    auto val = obj.Get(key);
                    if (val.IsObject()) {
                      auto val_obj = val.As<Napi::Object>();
                      if (!ctor.IsEmpty() && val_obj.InstanceOf(ctor)) {
                        const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(val_obj);
                        if (wrapper) {
                          kwargs.insert_or_assign(key_str, wrapper->tensor());
                        }
                      }
                    }
                  }
                }
              }
            } else {
              // Multiple positional arguments (all should be MLX arrays)
              auto ctor = addon->array_constructor.Value();
              for (size_t i = 0; i < call_info.Length(); i++) {
                if (call_info[i].IsObject()) {
                  auto obj = call_info[i].As<Napi::Object>();
                  if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
                    const auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(obj);
                    if (wrapper) {
                      args.push_back(wrapper->tensor());
                    }
                  }
                }
              }
            }

            // Call the imported function
            std::vector<mlx::core::array> results;
            if (!args.empty() && !kwargs.empty()) {
              results = (*fn_ptr)(args, kwargs);
            } else if (!kwargs.empty()) {
              results = (*fn_ptr)(kwargs);
            } else {
              results = (*fn_ptr)(args);
            }

            // Return results as an array (tuple)
            auto result_array = Napi::Array::New(call_env, results.size());
            for (size_t i = 0; i < results.size(); i++) {
              result_array.Set(
                  i,
                  WrapArray(call_env, std::make_shared<mlx::core::array>(std::move(results[i]))));
            }
            return result_array;

          } catch (const std::exception& e) {
            Napi::Error::New(call_env, std::string("Imported function call failed: ") + e.what())
                .ThrowAsJavaScriptException();
            return call_env.Null();
          }
        },
        "imported_function");

    return js_function;

  } catch (const std::exception& e) {
    Napi::Error::New(env, std::string("import_function failed: ") + e.what())
        .ThrowAsJavaScriptException();
    return env.Null();
  }
}

// ---------------------------------------------------------------------------
// Batch: simple unary ops
// ---------------------------------------------------------------------------
#define DEFINE_UNARY_OP(Name, mlx_fn)                                         \
  Napi::Value Name(const Napi::CallbackInfo& info) {                          \
    auto env = info.Env();                                                    \
    if (info.Length() < 1) {                                                  \
      Napi::TypeError::New(env, #mlx_fn " expects at least one argument")    \
          .ThrowAsJavaScriptException();                                      \
      return env.Null();                                                      \
    }                                                                         \
    auto a = ToArray(env, info[0]);                                           \
    if (env.IsExceptionPending()) return env.Null();                          \
    auto streamArg = GetStreamArgument(info, 1);                              \
    if (env.IsExceptionPending()) return env.Null();                          \
    return WrapArray(env,                                                     \
        std::make_shared<mlx::core::array>(mlx::core::mlx_fn(a, streamArg))); \
  }

DEFINE_UNARY_OP(Ceil, ceil)
DEFINE_UNARY_OP(Floor, floor)
DEFINE_UNARY_OP(Round, round)
DEFINE_UNARY_OP(IsNan, isnan)
DEFINE_UNARY_OP(IsInf, isinf)
DEFINE_UNARY_OP(IsFinite, isfinite)
DEFINE_UNARY_OP(LogicalNot, logical_not)
DEFINE_UNARY_OP(Sinh, sinh)
DEFINE_UNARY_OP(Cosh, cosh)
DEFINE_UNARY_OP(ArcSinh, arcsinh)
DEFINE_UNARY_OP(ArcCosh, arccosh)
DEFINE_UNARY_OP(ArcTanh, arctanh)
DEFINE_UNARY_OP(Degrees, degrees)
DEFINE_UNARY_OP(Radians, radians)
DEFINE_UNARY_OP(ErfInv, erfinv)
DEFINE_UNARY_OP(Expm1, expm1)
#undef DEFINE_UNARY_OP

// ---------------------------------------------------------------------------
// Cumulative ops: cumsum/cumprod/cummax/cummin(a, axis?, reverse?, inclusive?, stream?)
// ---------------------------------------------------------------------------
#define DEFINE_CUM_OP(Name, mlx_fn)                                            \
  Napi::Value Name(const Napi::CallbackInfo& info) {                           \
    auto env = info.Env();                                                     \
    if (info.Length() < 1) {                                                   \
      Napi::TypeError::New(env, #mlx_fn " expects at least one argument")      \
          .ThrowAsJavaScriptException();                                        \
      return env.Null();                                                        \
    }                                                                          \
    auto a = ToArray(env, info[0]);                                            \
    if (env.IsExceptionPending()) return env.Null();                           \
    int axis = 0;                                                              \
    bool reverse = false;                                                      \
    bool inclusive = true;                                                      \
    size_t nextIdx = 1;                                                        \
    if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {                  \
      axis = info[nextIdx].As<Napi::Number>().Int32Value();                    \
      nextIdx++;                                                               \
    }                                                                          \
    if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {                 \
      reverse = info[nextIdx].As<Napi::Boolean>().Value();                     \
      nextIdx++;                                                               \
    }                                                                          \
    if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {                 \
      inclusive = info[nextIdx].As<Napi::Boolean>().Value();                    \
      nextIdx++;                                                               \
    }                                                                          \
    auto streamArg = GetStreamArgument(info, nextIdx);                          \
    if (env.IsExceptionPending()) return env.Null();                           \
    return WrapArray(env,                                                      \
        std::make_shared<mlx::core::array>(                                    \
            mlx::core::mlx_fn(a, axis, reverse, inclusive, streamArg)));        \
  }

DEFINE_CUM_OP(CumSum, cumsum)
DEFINE_CUM_OP(CumProd, cumprod)
DEFINE_CUM_OP(CumMax, cummax)
DEFINE_CUM_OP(CumMin, cummin)
#undef DEFINE_CUM_OP

// ---------------------------------------------------------------------------
// Batch: simple binary ops
// ---------------------------------------------------------------------------
#define DEFINE_BINARY_OP(Name, mlx_fn)                                         \
  Napi::Value Name(const Napi::CallbackInfo& info) {                           \
    auto env = info.Env();                                                     \
    if (info.Length() < 2) {                                                   \
      Napi::TypeError::New(env, #mlx_fn " expects two arguments")             \
          .ThrowAsJavaScriptException();                                       \
      return env.Null();                                                       \
    }                                                                          \
    auto a = ToArray(env, info[0]);                                            \
    if (env.IsExceptionPending()) return env.Null();                           \
    auto b = ToArray(env, info[1]);                                            \
    if (env.IsExceptionPending()) return env.Null();                           \
    auto streamArg = GetStreamArgument(info, 2);                               \
    if (env.IsExceptionPending()) return env.Null();                           \
    return WrapArray(env,                                                      \
        std::make_shared<mlx::core::array>(                                    \
            mlx::core::mlx_fn(a, b, streamArg)));                              \
  }

DEFINE_BINARY_OP(FloorDivide, floor_divide)
DEFINE_BINARY_OP(Remainder, remainder)
DEFINE_BINARY_OP(LogicalAnd, logical_and)
DEFINE_BINARY_OP(LogicalOr, logical_or)
DEFINE_BINARY_OP(BitwiseAnd, bitwise_and)
DEFINE_BINARY_OP(BitwiseOr, bitwise_or)
DEFINE_BINARY_OP(BitwiseXor, bitwise_xor)
DEFINE_BINARY_OP(LeftShift, left_shift)
DEFINE_BINARY_OP(RightShift, right_shift)
#undef DEFINE_BINARY_OP

// ---------------------------------------------------------------------------
// all(a, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value All(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "all expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();
  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::all(a, axes, keepdims, streamArg)));
}

// ---------------------------------------------------------------------------
// any(a, axis?, keepdims?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Any(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "any expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto axes = GetReduceAxes(env, info, 1, a);
  if (env.IsExceptionPending()) return env.Null();
  bool keepdims = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    keepdims = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::any(a, axes, keepdims, streamArg)));
}

// ---------------------------------------------------------------------------
// flatten(a, start_axis?, end_axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Flatten(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "flatten expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int start_axis = 0;
  int end_axis = -1;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    start_axis = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    end_axis = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::flatten(a, start_axis, end_axis, streamArg)));
}

// ---------------------------------------------------------------------------
// eye(n, m?, k?, dtype?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Eye(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "eye expects at least n (number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int n = info[0].As<Napi::Number>().Int32Value();
  int m = n;
  int k = 0;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    m = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    k = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto dtype = mlx::core::float32;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::eye(n, m, k, dtype, streamArg)));
}

// ---------------------------------------------------------------------------
// identity(n, dtype?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Identity(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "identity expects n (number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int n = info[0].As<Napi::Number>().Int32Value();
  auto dtype = mlx::core::float32;
  size_t nextIdx = 1;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::identity(n, dtype, streamArg)));
}

// ---------------------------------------------------------------------------
// linspace(start, stop, num?, dtype?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Linspace(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "linspace expects start and stop")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  double start = info[0].As<Napi::Number>().DoubleValue();
  double stop = info[1].As<Napi::Number>().DoubleValue();
  int num = 50;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    num = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto dtype = mlx::core::float32;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::linspace(start, stop, num, dtype, streamArg)));
}

// ---------------------------------------------------------------------------
// tril(a, k?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Tril(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "tril expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int k = 0;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    k = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::tril(a, k, streamArg)));
}

// ---------------------------------------------------------------------------
// triu(a, k?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Triu(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "triu expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int k = 0;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    k = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::triu(a, k, streamArg)));
}

// ---------------------------------------------------------------------------
// broadcast_to(a, shape, stream?)
// ---------------------------------------------------------------------------
Napi::Value BroadcastTo(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "broadcast_to expects array and shape")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto shape = ParseShapeArgument(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::broadcast_to(a, shape, streamArg)));
}

// ---------------------------------------------------------------------------
// repeat(a, repeats, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Repeat(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "repeat expects array and repeats")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int repeats = info[1].As<Napi::Number>().Int32Value();
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    int axis = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env,
        std::make_shared<mlx::core::array>(
            mlx::core::repeat(a, repeats, axis, streamArg)));
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::repeat(a, repeats, streamArg)));
}

// ---------------------------------------------------------------------------
// tile(a, reps, stream?)
// ---------------------------------------------------------------------------
Napi::Value Tile(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "tile expects array and reps")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  std::vector<int> reps;
  if (info[1].IsArray()) {
    auto arr = info[1].As<Napi::Array>();
    reps.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
      reps.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
  } else if (info[1].IsNumber()) {
    reps.push_back(info[1].As<Napi::Number>().Int32Value());
  }
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::tile(a, reps, streamArg)));
}

// ---------------------------------------------------------------------------
// sort(a, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Sort(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "sort expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int axis = -1;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && (info[nextIdx].IsNumber() || info[nextIdx].IsBigInt())) {
    axis = info[nextIdx].ToNumber().As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::sort(a, axis, streamArg)));
}

// ---------------------------------------------------------------------------
// argsort(a, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value ArgSort(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "argsort expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int axis = -1;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && (info[nextIdx].IsNumber() || info[nextIdx].IsBigInt())) {
    axis = info[nextIdx].ToNumber().As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::argsort(a, axis, streamArg)));
}

// ---------------------------------------------------------------------------
// diag(a, k?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Diag(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "diag expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int k = 0;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    k = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::diag(a, k, streamArg)));
}

// ---------------------------------------------------------------------------
// diagonal(a, offset?, axis1?, axis2?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Diagonal(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "diagonal expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int offset = 0, axis1 = 0, axis2 = 1;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    offset = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis1 = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis2 = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::diagonal(a, offset, axis1, axis2, streamArg)));
}

// ---------------------------------------------------------------------------
// topk(a, k, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value TopK(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "topk expects array and k")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int k = info[1].As<Napi::Number>().Int32Value();
  int axis = -1;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::topk(a, k, axis, streamArg)));
}

// ===========================================================================
// Random ops
// ===========================================================================

// random.seed(seed)
Napi::Value RandomSeed(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "random.seed expects a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint64_t seed = static_cast<uint64_t>(info[0].As<Napi::Number>().Int64Value());
  mlx::core::random::seed(seed);
  return env.Undefined();
}

// random.key(seed)
Napi::Value RandomKey(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "random.key expects a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint64_t seed = static_cast<uint64_t>(info[0].As<Napi::Number>().Int64Value());
  return WrapArray(env,
      std::make_shared<mlx::core::array>(mlx::core::random::key(seed)));
}

// random.split(key, num?, stream?)
Napi::Value RandomSplit(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "random.split expects a key array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto key = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    int num = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env,
        std::make_shared<mlx::core::array>(
            mlx::core::random::split(key, num, streamArg)));
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto [k1, k2] = mlx::core::random::split(key, streamArg);
  auto result = Napi::Array::New(env, 2);
  result.Set(uint32_t(0), WrapArray(env, std::make_shared<mlx::core::array>(k1)));
  result.Set(uint32_t(1), WrapArray(env, std::make_shared<mlx::core::array>(k2)));
  return result;
}

// random.randint(low, high, shape, dtype?, key?, stream?)
Napi::Value RandomRandint(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "random.randint expects low, high, and shape")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto low = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto high = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto shape = ParseShapeArgument(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();
  auto dtype = mlx::core::int32;
  size_t nextIdx = 3;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::int32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  std::optional<mlx::core::array> key = std::nullopt;
  if (info.Length() > nextIdx) {
    if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
      auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
          info[nextIdx].As<Napi::Object>());
      if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
    }
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::random::randint(low, high, shape, dtype, key, streamArg)));
}

// random.categorical(logits, axis?, key?, stream?)
Napi::Value RandomCategorical(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "random.categorical expects logits array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto logits = ToArray(env, info[0]);
    if (env.IsExceptionPending()) return env.Null();
    int axis = -1;
    size_t nextIdx = 1;
    if (info.Length() > nextIdx && (info[nextIdx].IsNumber() || info[nextIdx].IsBigInt())) {
      axis = info[nextIdx].ToNumber().As<Napi::Number>().Int32Value();
      nextIdx++;
    }
    std::optional<mlx::core::array> key = std::nullopt;
    if (info.Length() > nextIdx) {
      if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
        auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
            info[nextIdx].As<Napi::Object>());
        if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
      }
      nextIdx++;
    }
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    auto result = mlx::core::random::categorical(logits, axis, key, streamArg);
    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(result)));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// random.permutation(x_or_n, axis?, key?, stream?)
Napi::Value RandomPermutation(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "random.permutation expects array or int")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    size_t nextIdx = 1;
    if (info[0].IsNumber()) {
      int n = info[0].As<Napi::Number>().Int32Value();
      std::optional<mlx::core::array> key = std::nullopt;
      if (info.Length() > nextIdx) {
        if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
          auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
              info[nextIdx].As<Napi::Object>());
          if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
        }
        nextIdx++;
      }
      auto streamArg = GetStreamArgument(info, nextIdx);
      if (env.IsExceptionPending()) return env.Null();
      auto result = mlx::core::random::permutation(n, key, streamArg);
      return WrapArray(env, std::make_shared<mlx::core::array>(std::move(result)));    }
    auto x = ToArray(env, info[0]);
    if (env.IsExceptionPending()) return env.Null();
    int axis = 0;
    if (info.Length() > nextIdx && (info[nextIdx].IsNumber() || info[nextIdx].IsBigInt())) {
      axis = info[nextIdx].ToNumber().As<Napi::Number>().Int32Value();
      nextIdx++;
    }
    std::optional<mlx::core::array> key = std::nullopt;
    if (info.Length() > nextIdx) {
      if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
        auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
            info[nextIdx].As<Napi::Object>());
        if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
      }
      nextIdx++;
    }
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    auto result = mlx::core::random::permutation(x, axis, key, streamArg);
    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(result)));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// random.gumbel(shape, dtype?, key?, stream?)
Napi::Value RandomGumbel(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "random.gumbel expects shape")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto dtype = mlx::core::float32;
  size_t nextIdx = 1;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  std::optional<mlx::core::array> key = std::nullopt;
  if (info.Length() > nextIdx) {
    if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
      auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
          info[nextIdx].As<Napi::Object>());
      if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
    }
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::random::gumbel(shape, dtype, key, streamArg)));
}

// random.laplace(shape, dtype?, key?, stream?)
Napi::Value RandomLaplace(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "random.laplace expects shape")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto dtype = mlx::core::float32;
  size_t nextIdx = 1;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  std::optional<mlx::core::array> key = std::nullopt;
  if (info.Length() > nextIdx) {
    if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
      auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
          info[nextIdx].As<Napi::Object>());
      if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
    }
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::random::laplace(shape, dtype, key, streamArg)));
}

// random.truncated_normal(lower, upper, shape?, dtype?, key?, stream?)
Napi::Value RandomTruncatedNormal(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "random.truncated_normal expects lower and upper")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto lower = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto upper = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 2;
  // Check for optional shape array
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto shape = ParseShapeArgument(env, info[nextIdx]);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
    auto dtype = mlx::core::float32;
    if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
      dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
      if (env.IsExceptionPending()) return env.Null();
      nextIdx++;
    }
    std::optional<mlx::core::array> key = std::nullopt;
    if (info.Length() > nextIdx && info[nextIdx].IsObject() &&
        !info[nextIdx].IsArray()) {
      auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
          info[nextIdx].As<Napi::Object>());
      if (wrapper) { key = wrapper->tensor(); nextIdx++; }
    }
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env,
        std::make_shared<mlx::core::array>(
            mlx::core::random::truncated_normal(
                lower, upper, shape, dtype, key, streamArg)));
  }
  // No shape — use lower/upper shapes
  auto dtype = mlx::core::float32;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  std::optional<mlx::core::array> key = std::nullopt;
  if (info.Length() > nextIdx) {
    if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
      auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
          info[nextIdx].As<Napi::Object>());
      if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
    }
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::random::truncated_normal(
              lower, upper, dtype, key, streamArg)));
}

// random.multivariate_normal(mean, cov, shape, dtype?, key?, stream?)
Napi::Value RandomMultivariateNormal(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "random.multivariate_normal expects mean, cov, and shape")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto mean = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto cov = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto shape = ParseShapeArgument(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();
  auto dtype = mlx::core::float32;
  size_t nextIdx = 3;
  if (addon && info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], *addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, *addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  std::optional<mlx::core::array> key = std::nullopt;
  if (info.Length() > nextIdx) {
    if (info[nextIdx].IsObject() && !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
      auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
          info[nextIdx].As<Napi::Object>());
      if (wrapper) { key = std::make_optional<mlx::core::array>(wrapper->tensor()); }
    }
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env,
      std::make_shared<mlx::core::array>(
          mlx::core::random::multivariate_normal(
              mean, cov, shape, dtype, key, streamArg)));
}

// ===========================================================================
// Device management
// ===========================================================================

Napi::Value DefaultDevice(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto& d = mlx::core::default_device();
  auto result = Napi::Object::New(env);
  result.Set("type", d.type == mlx::core::Device::cpu ? "cpu" : "gpu");
  result.Set("index", Napi::Number::New(env, d.index));
  return result;
}

Napi::Value SetDefaultDevice(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "set_default_device expects a device type string")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string typeStr;
  if (info[0].IsString()) {
    typeStr = info[0].As<Napi::String>().Utf8Value();
  } else if (info[0].IsObject()) {
    auto obj = info[0].As<Napi::Object>();
    typeStr = obj.Get("type").As<Napi::String>().Utf8Value();
  }
  mlx::core::Device::DeviceType dt = mlx::core::Device::cpu;
  if (typeStr == "gpu") dt = mlx::core::Device::gpu;
  mlx::core::set_default_device(mlx::core::Device(dt));
  return env.Undefined();
}

Napi::Value IsAvailable(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "is_available expects a device type string")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string typeStr;
  if (info[0].IsString()) {
    typeStr = info[0].As<Napi::String>().Utf8Value();
  } else if (info[0].IsObject()) {
    auto obj = info[0].As<Napi::Object>();
    typeStr = obj.Get("type").As<Napi::String>().Utf8Value();
  }
  mlx::core::Device::DeviceType dt = mlx::core::Device::cpu;
  if (typeStr == "gpu") dt = mlx::core::Device::gpu;
  return Napi::Boolean::New(env, mlx::core::is_available(mlx::core::Device(dt)));
}

// ===========================================================================
// Memory management
// ===========================================================================

Napi::Value ClearCache(const Napi::CallbackInfo& info) {
  mlx::core::clear_cache();
  return info.Env().Undefined();
}

Napi::Value GetActiveMemory(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(mlx::core::get_active_memory()));
}

Napi::Value GetCacheMemory(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(mlx::core::get_cache_memory()));
}

Napi::Value GetPeakMemory(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(mlx::core::get_peak_memory()));
}

Napi::Value ResetPeakMemory(const Napi::CallbackInfo& info) {
  mlx::core::reset_peak_memory();
  return info.Env().Undefined();
}

Napi::Value SetCacheLimit(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "set_cache_limit expects a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  size_t limit = static_cast<size_t>(info[0].As<Napi::Number>().Int64Value());
  size_t prev = mlx::core::set_cache_limit(limit);
  return Napi::Number::New(env, static_cast<double>(prev));
}

Napi::Value SetMemoryLimit(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "set_memory_limit expects a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  size_t limit = static_cast<size_t>(info[0].As<Napi::Number>().Int64Value());
  size_t prev = mlx::core::set_memory_limit(limit);
  return Napi::Number::New(env, static_cast<double>(prev));
}

Napi::Value SetWiredLimit(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "set_wired_limit expects a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  size_t limit = static_cast<size_t>(info[0].As<Napi::Number>().Int64Value());
  size_t prev = mlx::core::set_wired_limit(limit);
  return Napi::Number::New(env, static_cast<double>(prev));
}

// ===========================================================================
// FFT ops
// ===========================================================================

// Helper to parse optional n (shape) and axes from args
// fftn(a, n?, axes?, stream?)
Napi::Value FFTn(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "fftn expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  // Check for n (shape array) and axes (array of ints)
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    n.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    axes.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  mlx::core::array result = hasN && hasAxes
      ? mlx::core::fft::fftn(a, mlx::core::Shape(n.begin(), n.end()), axes, streamArg)
      : hasAxes ? mlx::core::fft::fftn(a, axes, streamArg)
      : hasN ? mlx::core::fft::fftn(a, mlx::core::Shape(n.begin(), n.end()), axes, streamArg)
      : mlx::core::fft::fftn(a, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value IFFTn(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "ifftn expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    n.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    axes.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  mlx::core::array result = hasN && hasAxes
      ? mlx::core::fft::ifftn(a, mlx::core::Shape(n.begin(), n.end()), axes, streamArg)
      : hasAxes ? mlx::core::fft::ifftn(a, axes, streamArg)
      : mlx::core::fft::ifftn(a, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

// fft(a, n?, axis?, stream?)
Napi::Value FFT(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "fft expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  int axis = -1;
  bool hasN = false;
  int n = 0;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    n = info[nextIdx].As<Napi::Number>().Int32Value();
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN
      ? mlx::core::fft::fft(a, n, axis, streamArg)
      : mlx::core::fft::fft(a, axis, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value IFFT(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "ifft expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  int axis = -1;
  bool hasN = false;
  int n = 0;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    n = info[nextIdx].As<Napi::Number>().Int32Value();
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN
      ? mlx::core::fft::ifft(a, n, axis, streamArg)
      : mlx::core::fft::ifft(a, axis, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

// fft2(a, n?, axes?, stream?)
Napi::Value FFT2(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "fft2 expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    n.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    axes.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN
      ? mlx::core::fft::fft2(a, mlx::core::Shape(n.begin(), n.end()), axes.empty() ? std::vector<int>{-2, -1} : axes, streamArg)
      : hasAxes ? mlx::core::fft::fft2(a, axes, streamArg)
      : mlx::core::fft::fft2(a, std::vector<int>{-2, -1}, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value IFFT2(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "ifft2 expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    n.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    axes.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN
      ? mlx::core::fft::ifft2(a, mlx::core::Shape(n.begin(), n.end()), axes.empty() ? std::vector<int>{-2, -1} : axes, streamArg)
      : hasAxes ? mlx::core::fft::ifft2(a, axes, streamArg)
      : mlx::core::fft::ifft2(a, std::vector<int>{-2, -1}, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

// Real FFT variants
Napi::Value RFFTn(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "rfftn expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    n.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    axes.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN && hasAxes
      ? mlx::core::fft::rfftn(a, mlx::core::Shape(n.begin(), n.end()), axes, streamArg)
      : hasAxes ? mlx::core::fft::rfftn(a, axes, streamArg)
      : mlx::core::fft::rfftn(a, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value IRFFTn(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "irfftn expects at least one argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    n.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true;
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    axes.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN && hasAxes
      ? mlx::core::fft::irfftn(a, mlx::core::Shape(n.begin(), n.end()), axes, streamArg)
      : hasAxes ? mlx::core::fft::irfftn(a, axes, streamArg)
      : mlx::core::fft::irfftn(a, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value RFFT(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "rfft expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  int axis = -1;
  bool hasN = false;
  int n = 0;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    n = info[nextIdx].As<Napi::Number>().Int32Value(); hasN = true; nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN ? mlx::core::fft::rfft(a, n, axis, streamArg)
                     : mlx::core::fft::rfft(a, axis, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value IRFFT(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "irfft expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  int axis = -1;
  bool hasN = false;
  int n = 0;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    n = info[nextIdx].As<Napi::Number>().Int32Value(); hasN = true; nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN ? mlx::core::fft::irfft(a, n, axis, streamArg)
                     : mlx::core::fft::irfft(a, axis, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value RFFT2(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "rfft2 expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); ++i) n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true; nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); ++i) axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true; nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN
      ? mlx::core::fft::rfft2(a, mlx::core::Shape(n.begin(), n.end()), axes.empty() ? std::vector<int>{-2,-1} : axes, streamArg)
      : hasAxes ? mlx::core::fft::rfft2(a, axes, streamArg)
      : mlx::core::fft::rfft2(a, std::vector<int>{-2,-1}, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

Napi::Value IRFFT2(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "irfft2 expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  bool hasN = false, hasAxes = false;
  std::vector<int> n, axes;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); ++i) n.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasN = true; nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); ++i) axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    hasAxes = true; nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = hasN
      ? mlx::core::fft::irfft2(a, mlx::core::Shape(n.begin(), n.end()), axes.empty() ? std::vector<int>{-2,-1} : axes, streamArg)
      : hasAxes ? mlx::core::fft::irfft2(a, axes, streamArg)
      : mlx::core::fft::irfft2(a, std::vector<int>{-2,-1}, streamArg);
  return WrapArray(env, std::make_shared<mlx::core::array>(result));
}

// fftshift / ifftshift
Napi::Value FFTShift(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "fftshift expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    std::vector<int> axes;
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    nextIdx++;
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::fft::fftshift(a, axes, streamArg)));
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::fft::fftshift(a, streamArg)));
}

Napi::Value IFFTShift(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "ifftshift expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    auto arr = info[nextIdx].As<Napi::Array>();
    std::vector<int> axes;
    for (uint32_t i = 0; i < arr.Length(); ++i)
      axes.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    nextIdx++;
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::fft::ifftshift(a, axes, streamArg)));
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::fft::ifftshift(a, streamArg)));
}

// ===========================================================================
// Fast ops (layer_norm, rms_norm, rope)
// ===========================================================================

// fast.rms_norm(x, weight, eps, stream?)
Napi::Value FastRmsNorm(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "fast.rms_norm expects x, weight, and eps")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto x = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  std::optional<mlx::core::array> weight = std::nullopt;
  if (!info[1].IsNull() && !info[1].IsUndefined()) {
    weight = ToArray(env, info[1]);
    if (env.IsExceptionPending()) return env.Null();
  }
  float eps = info[2].As<Napi::Number>().FloatValue();
  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::fast::rms_norm(x, weight, eps, streamArg)));
}

// fast.layer_norm(x, weight, bias, eps, stream?)
Napi::Value FastLayerNorm(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 4) {
    Napi::TypeError::New(env, "fast.layer_norm expects x, weight, bias, and eps")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto x = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  std::optional<mlx::core::array> weight = std::nullopt;
  if (!info[1].IsNull() && !info[1].IsUndefined()) {
    weight = ToArray(env, info[1]);
    if (env.IsExceptionPending()) return env.Null();
  }
  std::optional<mlx::core::array> bias = std::nullopt;
  if (!info[2].IsNull() && !info[2].IsUndefined()) {
    bias = ToArray(env, info[2]);
    if (env.IsExceptionPending()) return env.Null();
  }
  float eps = info[3].As<Napi::Number>().FloatValue();
  auto streamArg = GetStreamArgument(info, 4);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::fast::layer_norm(x, weight, bias, eps, streamArg)));
}

// fast.rope(x, dims, traditional, base, scale, offset, freqs?, stream?)
Napi::Value FastRope(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 6) {
    Napi::TypeError::New(env, "fast.rope expects x, dims, traditional, base, scale, offset")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto x = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int dims = info[1].As<Napi::Number>().Int32Value();
  bool traditional = info[2].As<Napi::Boolean>().Value();
  std::optional<float> base = std::nullopt;
  if (!info[3].IsNull() && !info[3].IsUndefined()) {
    base = info[3].As<Napi::Number>().FloatValue();
  }
  float scale = info[4].As<Napi::Number>().FloatValue();
  int offset = info[5].As<Napi::Number>().Int32Value();
  size_t nextIdx = 6;
  std::optional<mlx::core::array> freqs = std::nullopt;
  if (info.Length() > nextIdx && info[nextIdx].IsObject() &&
      !info[nextIdx].IsArray() && !info[nextIdx].IsNull()) {
    auto wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(
        info[nextIdx].As<Napi::Object>());
    if (wrapper) { freqs = wrapper->tensor(); nextIdx++; }
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::fast::rope(x, dims, traditional, base, scale, offset, freqs, streamArg)));
}

// ---------------------------------------------------------------------------
// Batch 3A: Simple unary ops
// ---------------------------------------------------------------------------
#define DEFINE_UNARY_OP3(Name, mlx_fn)                                         \
  Napi::Value Name(const Napi::CallbackInfo& info) {                          \
    auto env = info.Env();                                                    \
    if (info.Length() < 1) {                                                  \
      Napi::TypeError::New(env, #mlx_fn " expects at least one argument")    \
          .ThrowAsJavaScriptException();                                      \
      return env.Null();                                                      \
    }                                                                         \
    auto a = ToArray(env, info[0]);                                           \
    if (env.IsExceptionPending()) return env.Null();                          \
    auto streamArg = GetStreamArgument(info, 1);                              \
    if (env.IsExceptionPending()) return env.Null();                          \
    return WrapArray(env,                                                     \
        std::make_shared<mlx::core::array>(mlx::core::mlx_fn(a, streamArg))); \
  }

DEFINE_UNARY_OP3(Log2, log2)
DEFINE_UNARY_OP3(Log10, log10)
DEFINE_UNARY_OP3(IsPosInf, isposinf)
DEFINE_UNARY_OP3(IsNegInf, isneginf)
DEFINE_UNARY_OP3(BitwiseInvert, bitwise_invert)
DEFINE_UNARY_OP3(Conjugate, conjugate)
DEFINE_UNARY_OP3(Real, real)
DEFINE_UNARY_OP3(Imag, imag)
DEFINE_UNARY_OP3(StopGradient, stop_gradient)
#undef DEFINE_UNARY_OP3

// ---------------------------------------------------------------------------
// Batch 3B: Simple binary ops
// ---------------------------------------------------------------------------
#define DEFINE_BINARY_OP3(Name, mlx_fn)                                        \
  Napi::Value Name(const Napi::CallbackInfo& info) {                           \
    auto env = info.Env();                                                     \
    if (info.Length() < 2) {                                                   \
      Napi::TypeError::New(env, #mlx_fn " expects two arguments")             \
          .ThrowAsJavaScriptException();                                       \
      return env.Null();                                                       \
    }                                                                          \
    auto a = ToArray(env, info[0]);                                            \
    if (env.IsExceptionPending()) return env.Null();                           \
    auto b = ToArray(env, info[1]);                                            \
    if (env.IsExceptionPending()) return env.Null();                           \
    auto streamArg = GetStreamArgument(info, 2);                               \
    if (env.IsExceptionPending()) return env.Null();                           \
    return WrapArray(env,                                                      \
        std::make_shared<mlx::core::array>(                                    \
            mlx::core::mlx_fn(a, b, streamArg)));                              \
  }

DEFINE_BINARY_OP3(Outer, outer)
DEFINE_BINARY_OP3(Inner, inner)
DEFINE_BINARY_OP3(Kron, kron)
#undef DEFINE_BINARY_OP3

// ---------------------------------------------------------------------------
// nan_to_num(a, nan?, posinf?, neginf?, stream?)
// ---------------------------------------------------------------------------
Napi::Value NanToNum(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "nan_to_num expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  float nan_val = 0.0f;
  std::optional<float> posinf_val = std::nullopt;
  std::optional<float> neginf_val = std::nullopt;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    nan_val = info[nextIdx].As<Napi::Number>().FloatValue();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    posinf_val = info[nextIdx].As<Napi::Number>().FloatValue();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    neginf_val = info[nextIdx].As<Napi::Number>().FloatValue();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::nan_to_num(a, nan_val, posinf_val, neginf_val, streamArg)));
}

// ---------------------------------------------------------------------------
// allclose(a, b, rtol?, atol?, equal_nan?, stream?)
// ---------------------------------------------------------------------------
Napi::Value AllClose(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "allclose expects at least two arguments").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  double rtol = 1e-5;
  double atol = 1e-8;
  bool equal_nan = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    rtol = info[nextIdx].As<Napi::Number>().DoubleValue();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    atol = info[nextIdx].As<Napi::Number>().DoubleValue();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    equal_nan = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::allclose(a, b, rtol, atol, equal_nan, streamArg)));
}

// ---------------------------------------------------------------------------
// isclose(a, b, rtol?, atol?, equal_nan?, stream?)
// ---------------------------------------------------------------------------
Napi::Value IsClose(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "isclose expects at least two arguments").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  double rtol = 1e-5;
  double atol = 1e-8;
  bool equal_nan = false;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    rtol = info[nextIdx].As<Napi::Number>().DoubleValue();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    atol = info[nextIdx].As<Napi::Number>().DoubleValue();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    equal_nan = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::isclose(a, b, rtol, atol, equal_nan, streamArg)));
}

// ---------------------------------------------------------------------------
// view(a, dtype, stream?)
// ---------------------------------------------------------------------------
Napi::Value View(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "view expects (array, dtype)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto& addon = mlx::node::GetAddonData(env);
  auto dtype = MaybeParseDtype(env, info[1], mlx::core::float32, addon);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::view(a, dtype, streamArg)));
}

// ---------------------------------------------------------------------------
// contiguous(a, allow_col_major?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Contiguous(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "contiguous expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  bool allow_col_major = false;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    allow_col_major = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::contiguous(a, allow_col_major, streamArg)));
}

// ---------------------------------------------------------------------------
// hadamard_transform(a, scale?, stream?)
// ---------------------------------------------------------------------------
Napi::Value HadamardTransform(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "hadamard_transform expects at least one argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  std::optional<float> scale = std::nullopt;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    scale = info[nextIdx].As<Napi::Number>().FloatValue();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::hadamard_transform(a, scale, streamArg)));
}

// ---------------------------------------------------------------------------
// unflatten(a, axis, shape, stream?)
// ---------------------------------------------------------------------------
Napi::Value Unflatten(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "unflatten expects (array, axis, shape)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int axis = info[1].As<Napi::Number>().Int32Value();
  auto shapeArr = info[2].As<Napi::Array>();
  mlx::core::Shape shape;
  for (uint32_t i = 0; i < shapeArr.Length(); i++) {
    shape.push_back(shapeArr.Get(i).As<Napi::Number>().Int64Value());
  }
  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::unflatten(a, axis, shape, streamArg)));
}

// ---------------------------------------------------------------------------
// partition(a, kth, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Partition(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "partition expects (array, kth)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int kth = info[1].As<Napi::Number>().Int32Value();
  size_t nextIdx = 2;
  bool hasAxis = false;
  int axis = 0;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value();
    hasAxis = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  if (hasAxis) {
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::partition(a, kth, axis, streamArg)));
  } else {
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::partition(a, kth, streamArg)));
  }
}

// ---------------------------------------------------------------------------
// argpartition(a, kth, axis?, stream?)
// ---------------------------------------------------------------------------
Napi::Value ArgPartition(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "argpartition expects (array, kth)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int kth = info[1].As<Napi::Number>().Int32Value();
  size_t nextIdx = 2;
  bool hasAxis = false;
  int axis = 0;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    axis = info[nextIdx].As<Napi::Number>().Int32Value();
    hasAxis = true;
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  if (hasAxis) {
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::argpartition(a, kth, axis, streamArg)));
  } else {
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::argpartition(a, kth, streamArg)));
  }
}

// ---------------------------------------------------------------------------
// put_along_axis(a, indices, values, axis, stream?)
// ---------------------------------------------------------------------------
Napi::Value PutAlongAxis(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 4) {
    Napi::TypeError::New(env, "put_along_axis expects (array, indices, values, axis)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto indices = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto values = ToArray(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();
  int axis = info[3].As<Napi::Number>().Int32Value();
  auto streamArg = GetStreamArgument(info, 4);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::put_along_axis(a, indices, values, axis, streamArg)));
}

// ---------------------------------------------------------------------------
// roll(a, shift, axis?, stream?)
// shift can be a number or an array of numbers
// axis can be a number or an array of numbers
// ---------------------------------------------------------------------------
Napi::Value Roll(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "roll expects (array, shift[, axis])").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();

  // Parse shift: single int or array of ints
  bool shiftIsArray = info[1].IsArray();
  int singleShift = 0;
  mlx::core::Shape shiftVec;
  if (shiftIsArray) {
    auto arr = info[1].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); i++) {
      shiftVec.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
    }
  } else {
    singleShift = info[1].As<Napi::Number>().Int32Value();
  }

  size_t nextIdx = 2;
  bool hasAxis = false;
  bool axisIsArray = false;
  int singleAxis = 0;
  std::vector<int> axesVec;
  if (info.Length() > nextIdx && (info[nextIdx].IsNumber() || info[nextIdx].IsArray())) {
    hasAxis = true;
    if (info[nextIdx].IsArray()) {
      axisIsArray = true;
      auto arr = info[nextIdx].As<Napi::Array>();
      for (uint32_t i = 0; i < arr.Length(); i++) {
        axesVec.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
      }
    } else {
      singleAxis = info[nextIdx].As<Napi::Number>().Int32Value();
    }
    nextIdx++;
  }

  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();

  if (!hasAxis) {
    if (shiftIsArray) {
      return WrapArray(env, std::make_shared<mlx::core::array>(
          mlx::core::roll(a, shiftVec, streamArg)));
    } else {
      return WrapArray(env, std::make_shared<mlx::core::array>(
          mlx::core::roll(a, singleShift, streamArg)));
    }
  } else if (axisIsArray) {
    if (shiftIsArray) {
      return WrapArray(env, std::make_shared<mlx::core::array>(
          mlx::core::roll(a, shiftVec, axesVec, streamArg)));
    } else {
      return WrapArray(env, std::make_shared<mlx::core::array>(
          mlx::core::roll(a, singleShift, axesVec, streamArg)));
    }
  } else {
    if (shiftIsArray) {
      return WrapArray(env, std::make_shared<mlx::core::array>(
          mlx::core::roll(a, shiftVec, singleAxis, streamArg)));
    } else {
      return WrapArray(env, std::make_shared<mlx::core::array>(
          mlx::core::roll(a, singleShift, singleAxis, streamArg)));
    }
  }
}

// ---------------------------------------------------------------------------
// tri(n, m?, k?, dtype?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Tri(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "tri expects at least n").ThrowAsJavaScriptException();
    return env.Null();
  }
  int n = info[0].As<Napi::Number>().Int32Value();
  int m = n;
  int k = 0;
  mlx::core::Dtype dtype = mlx::core::float32;
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    m = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    k = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto& addon = mlx::node::GetAddonData(env);
  if (info.Length() > nextIdx && IsDtypeArg(env, info[nextIdx], addon)) {
    dtype = MaybeParseDtype(env, info[nextIdx], mlx::core::float32, addon);
    if (env.IsExceptionPending()) return env.Null();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::tri(n, m, k, dtype, streamArg)));
}

// ---------------------------------------------------------------------------
// meshgrid(arrays..., sparse?, indexing?, stream?)  → returns JS array of MLXArrays
// ---------------------------------------------------------------------------
Napi::Value Meshgrid(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "meshgrid expects ([arrays], sparse?, indexing?)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto jsArrays = info[0].As<Napi::Array>();
  std::vector<mlx::core::array> arrays;
  for (uint32_t i = 0; i < jsArrays.Length(); i++) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(jsArrays.Get(i).As<Napi::Object>());
    arrays.push_back(wrapper->tensor());
  }
  bool sparse = false;
  std::string indexing = "xy";
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    sparse = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsString()) {
    indexing = info[nextIdx].As<Napi::String>().Utf8Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = mlx::core::meshgrid(arrays, sparse, indexing, streamArg);
  auto jsResult = Napi::Array::New(env, result.size());
  for (size_t i = 0; i < result.size(); i++) {
    jsResult.Set(i, WrapArray(env, std::make_shared<mlx::core::array>(std::move(result[i]))));
  }
  return jsResult;
}

// ---------------------------------------------------------------------------
// broadcast_arrays(arrays, stream?) → returns JS array of MLXArrays
// ---------------------------------------------------------------------------
Napi::Value BroadcastArrays(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "broadcast_arrays expects ([arrays])").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto jsArrays = info[0].As<Napi::Array>();
  std::vector<mlx::core::array> arrays;
  for (uint32_t i = 0; i < jsArrays.Length(); i++) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(jsArrays.Get(i).As<Napi::Object>());
    arrays.push_back(wrapper->tensor());
  }
  auto streamArg = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();
  auto result = mlx::core::broadcast_arrays(arrays, streamArg);
  auto jsResult = Napi::Array::New(env, result.size());
  for (size_t i = 0; i < result.size(); i++) {
    jsResult.Set(i, WrapArray(env, std::make_shared<mlx::core::array>(std::move(result[i]))));
  }
  return jsResult;
}

// ---------------------------------------------------------------------------
// atleast_1d / atleast_2d / atleast_3d (single array overload)
// ---------------------------------------------------------------------------
#define DEFINE_ATLEAST(Name, mlx_fn)                                           \
  Napi::Value Name(const Napi::CallbackInfo& info) {                          \
    auto env = info.Env();                                                    \
    if (info.Length() < 1) {                                                  \
      Napi::TypeError::New(env, #mlx_fn " expects at least one argument")    \
          .ThrowAsJavaScriptException();                                      \
      return env.Null();                                                      \
    }                                                                         \
    auto a = ToArray(env, info[0]);                                           \
    if (env.IsExceptionPending()) return env.Null();                          \
    auto streamArg = GetStreamArgument(info, 1);                              \
    if (env.IsExceptionPending()) return env.Null();                          \
    return WrapArray(env,                                                     \
        std::make_shared<mlx::core::array>(mlx::core::mlx_fn(a, streamArg))); \
  }

DEFINE_ATLEAST(AtLeast1d, atleast_1d)
DEFINE_ATLEAST(AtLeast2d, atleast_2d)
DEFINE_ATLEAST(AtLeast3d, atleast_3d)
#undef DEFINE_ATLEAST

// ---------------------------------------------------------------------------
// slice_update(src, update, start, stop, strides?, stream?)
// ---------------------------------------------------------------------------
Napi::Value SliceUpdate(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 4) {
    Napi::TypeError::New(env, "slice_update expects (src, update, start, stop)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto src = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto update = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  auto startArr = info[2].As<Napi::Array>();
  auto stopArr = info[3].As<Napi::Array>();
  mlx::core::Shape start, stop;
  for (uint32_t i = 0; i < startArr.Length(); i++)
    start.push_back(startArr.Get(i).As<Napi::Number>().Int64Value());
  for (uint32_t i = 0; i < stopArr.Length(); i++)
    stop.push_back(stopArr.Get(i).As<Napi::Number>().Int64Value());

  size_t nextIdx = 4;
  bool hasStrides = false;
  mlx::core::Shape strides;
  if (info.Length() > nextIdx && info[nextIdx].IsArray()) {
    hasStrides = true;
    auto stridesArr = info[nextIdx].As<Napi::Array>();
    for (uint32_t i = 0; i < stridesArr.Length(); i++)
      strides.push_back(stridesArr.Get(i).As<Napi::Number>().Int64Value());
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();

  if (hasStrides) {
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::slice_update(src, update, start, stop, strides, streamArg)));
  } else {
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::slice_update(src, update, start, stop, streamArg)));
  }
}

// ---------------------------------------------------------------------------
// conv_general(input, weight, stride?, padding_lo?, padding_hi?, kernel_dilation?, input_dilation?, groups?, stream?)
// ---------------------------------------------------------------------------
Napi::Value ConvGeneral(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv_general expects (input, weight, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto input = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto weight = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  auto parseIntVec = [&](size_t idx) -> std::vector<int> {
    std::vector<int> v;
    if (info.Length() > idx && info[idx].IsArray()) {
      auto arr = info[idx].As<Napi::Array>();
      for (uint32_t i = 0; i < arr.Length(); i++)
        v.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return v;
  };

  auto stride = parseIntVec(2);
  auto padding_lo = parseIntVec(3);
  auto padding_hi = parseIntVec(4);
  auto kernel_dilation = parseIntVec(5);
  auto input_dilation = parseIntVec(6);
  int groups = 1;
  bool flip = false;
  size_t nextIdx = 7;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    groups = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    flip = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::conv_general(input, weight, stride, padding_lo, padding_hi,
                               kernel_dilation, input_dilation, groups, flip, streamArg)));
}

// ---------------------------------------------------------------------------
// conv_transpose1d(input, weight, stride?, padding?, dilation?, output_padding?, groups?, stream?)
// ---------------------------------------------------------------------------
Napi::Value ConvTranspose1d(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv_transpose1d expects (input, weight, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto input = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto weight = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  int stride = 1, padding = 0, dilation = 1, output_padding = 0, groups = 1;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { stride = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { padding = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { dilation = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { output_padding = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { groups = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::conv_transpose1d(input, weight, stride, padding, dilation, output_padding, groups, streamArg)));
}

// ---------------------------------------------------------------------------
// conv_transpose2d(input, weight, stride?, padding?, dilation?, output_padding?, groups?, stream?)
// Each of stride/padding/dilation/output_padding is a pair [h,w] or scalar
// ---------------------------------------------------------------------------
Napi::Value ConvTranspose2d(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv_transpose2d expects (input, weight, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto input = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto weight = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  auto parsePair = [&](size_t idx, std::pair<int,int> def) -> std::pair<int,int> {
    if (info.Length() > idx && info[idx].IsArray()) {
      auto arr = info[idx].As<Napi::Array>();
      return {arr.Get((uint32_t)0).As<Napi::Number>().Int32Value(),
              arr.Get((uint32_t)1).As<Napi::Number>().Int32Value()};
    } else if (info.Length() > idx && info[idx].IsNumber()) {
      int v = info[idx].As<Napi::Number>().Int32Value();
      return {v, v};
    }
    return def;
  };

  auto stride = parsePair(2, {1,1});
  auto padding = parsePair(3, {0,0});
  auto dilation = parsePair(4, {1,1});
  auto output_padding = parsePair(5, {0,0});
  int groups = 1;
  size_t nextIdx = 6;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    groups = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::conv_transpose2d(input, weight, stride, padding, dilation, output_padding, groups, streamArg)));
}

// ---------------------------------------------------------------------------
// conv_transpose3d(input, weight, stride?, padding?, dilation?, output_padding?, groups?, stream?)
// ---------------------------------------------------------------------------
Napi::Value ConvTranspose3d(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "conv_transpose3d expects (input, weight, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto input = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto weight = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  auto parseTuple3 = [&](size_t idx, std::tuple<int,int,int> def) -> std::tuple<int,int,int> {
    if (info.Length() > idx && info[idx].IsArray()) {
      auto arr = info[idx].As<Napi::Array>();
      return {arr.Get((uint32_t)0).As<Napi::Number>().Int32Value(),
              arr.Get((uint32_t)1).As<Napi::Number>().Int32Value(),
              arr.Get((uint32_t)2).As<Napi::Number>().Int32Value()};
    } else if (info.Length() > idx && info[idx].IsNumber()) {
      int v = info[idx].As<Napi::Number>().Int32Value();
      return {v, v, v};
    }
    return def;
  };

  auto stride = parseTuple3(2, {1,1,1});
  auto padding = parseTuple3(3, {0,0,0});
  auto dilation = parseTuple3(4, {1,1,1});
  auto output_padding = parseTuple3(5, {0,0,0});
  int groups = 1;
  size_t nextIdx = 6;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) {
    groups = info[nextIdx].As<Napi::Number>().Int32Value();
    nextIdx++;
  }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::conv_transpose3d(input, weight, stride, padding, dilation, output_padding, groups, streamArg)));
}

// ---------------------------------------------------------------------------
// einsum(subscripts, operands, stream?)
// ---------------------------------------------------------------------------
Napi::Value Einsum(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "einsum expects (subscripts, [operands])").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string subscripts = info[0].As<Napi::String>().Utf8Value();
  auto jsOps = info[1].As<Napi::Array>();
  std::vector<mlx::core::array> operands;
  for (uint32_t i = 0; i < jsOps.Length(); i++) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(jsOps.Get(i).As<Napi::Object>());
    operands.push_back(wrapper->tensor());
  }
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::einsum(subscripts, operands, streamArg)));
}

// ---------------------------------------------------------------------------
// tensordot(a, b, axes_or_dims, stream?)
// axes can be an int (number of dims) or [axes_a, axes_b]
// ---------------------------------------------------------------------------
Napi::Value Tensordot(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "tensordot expects (a, b, axes)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  size_t nextIdx = 3;
  if (info[2].IsNumber()) {
    int dims = info[2].As<Napi::Number>().Int32Value();
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::tensordot(a, b, dims, streamArg)));
  } else if (info[2].IsArray()) {
    auto axesArr = info[2].As<Napi::Array>();
    auto axesA_arr = axesArr.Get((uint32_t)0).As<Napi::Array>();
    auto axesB_arr = axesArr.Get((uint32_t)1).As<Napi::Array>();
    std::vector<int> axes_a, axes_b;
    for (uint32_t i = 0; i < axesA_arr.Length(); i++)
      axes_a.push_back(axesA_arr.Get(i).As<Napi::Number>().Int32Value());
    for (uint32_t i = 0; i < axesB_arr.Length(); i++)
      axes_b.push_back(axesB_arr.Get(i).As<Napi::Number>().Int32Value());
    auto streamArg = GetStreamArgument(info, nextIdx);
    if (env.IsExceptionPending()) return env.Null();
    return WrapArray(env, std::make_shared<mlx::core::array>(
        mlx::core::tensordot(a, b, axes_a, axes_b, streamArg)));
  }
  Napi::TypeError::New(env, "tensordot axes must be int or [[axes_a], [axes_b]]").ThrowAsJavaScriptException();
  return env.Null();
}

// ---------------------------------------------------------------------------
// block_masked_mm(a, b, block_size, mask_out?, mask_lhs?, mask_rhs?, stream?)
// ---------------------------------------------------------------------------
Napi::Value BlockMaskedMM(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "block_masked_mm expects (a, b, block_size)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  int block_size = info[2].As<Napi::Number>().Int32Value();

  std::optional<mlx::core::array> mask_out = std::nullopt;
  std::optional<mlx::core::array> mask_lhs = std::nullopt;
  std::optional<mlx::core::array> mask_rhs = std::nullopt;
  size_t nextIdx = 3;

  // mask_out (can be null/undefined to skip)
  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) mask_out = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx) { nextIdx++; }

  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) mask_lhs = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx) { nextIdx++; }

  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) mask_rhs = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx) { nextIdx++; }

  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::block_masked_mm(a, b, block_size, mask_out, mask_lhs, mask_rhs, streamArg)));
}

// ---------------------------------------------------------------------------
// gather_mm(a, b, lhs_indices?, rhs_indices?, sorted_indices?, stream?)
// ---------------------------------------------------------------------------
Napi::Value GatherMM(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "gather_mm expects (a, b, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  std::optional<mlx::core::array> lhs_indices = std::nullopt;
  std::optional<mlx::core::array> rhs_indices = std::nullopt;
  bool sorted_indices = false;
  size_t nextIdx = 2;

  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) lhs_indices = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx) { nextIdx++; }

  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) rhs_indices = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx) { nextIdx++; }

  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) {
    sorted_indices = info[nextIdx].As<Napi::Boolean>().Value();
    nextIdx++;
  }

  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::gather_mm(a, b, lhs_indices, rhs_indices, sorted_indices, streamArg)));
}

// ---------------------------------------------------------------------------
// segmented_mm(a, b, segments, stream?)
// ---------------------------------------------------------------------------
Napi::Value SegmentedMM(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "segmented_mm expects (a, b, segments)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto a = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto segments = ToArray(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();
  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::segmented_mm(a, b, segments, streamArg)));
}

// ---------------------------------------------------------------------------
// quantize(w, group_size?, bits?, mode?, stream?) → [quantized, scales, biases]
// ---------------------------------------------------------------------------
Napi::Value Quantize(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "quantize expects (array, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto w = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  int group_size = 64;
  int bits = 4;
  std::string mode = "affine";
  size_t nextIdx = 1;
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { group_size = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { bits = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsString()) { mode = info[nextIdx].As<Napi::String>().Utf8Value(); nextIdx++; }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  auto result = mlx::core::quantize(w, group_size, bits, mode, streamArg);
  auto jsResult = Napi::Array::New(env, result.size());
  for (size_t i = 0; i < result.size(); i++) {
    jsResult.Set(i, WrapArray(env, std::make_shared<mlx::core::array>(std::move(result[i]))));
  }
  return jsResult;
}

// ---------------------------------------------------------------------------
// dequantize(w, scales, biases?, group_size?, bits?, stream?)
// ---------------------------------------------------------------------------
Napi::Value Dequantize(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "dequantize expects (w, scales, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto w = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto scales = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  std::optional<mlx::core::array> biases = std::nullopt;
  size_t nextIdx = 2;
  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) biases = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx && (info[nextIdx].IsNull() || info[nextIdx].IsUndefined())) {
    nextIdx++;
  }

  int group_size = 64;
  int bits = 4;
  std::string mode = "affine";
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { group_size = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { bits = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsString()) { mode = info[nextIdx].As<Napi::String>().Utf8Value(); nextIdx++; }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::dequantize(w, scales, biases, group_size, bits, mode, streamArg)));
}

// ---------------------------------------------------------------------------
// quantized_matmul(x, w, scales, biases?, transpose?, group_size?, bits?, stream?)
// ---------------------------------------------------------------------------
Napi::Value QuantizedMatmul(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "quantized_matmul expects (x, w, scales, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto x = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto w = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto scales = ToArray(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();

  std::optional<mlx::core::array> biases = std::nullopt;
  size_t nextIdx = 3;
  if (info.Length() > nextIdx && info[nextIdx].IsObject() && !info[nextIdx].IsNull() && !info[nextIdx].IsUndefined()) {
    auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[nextIdx].As<Napi::Object>());
    if (wrapper) biases = wrapper->tensor();
    nextIdx++;
  } else if (info.Length() > nextIdx && (info[nextIdx].IsNull() || info[nextIdx].IsUndefined())) {
    nextIdx++;
  }

  bool transpose = true;
  int group_size = 64;
  int bits = 4;
  std::string mode = "affine";
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) { transpose = info[nextIdx].As<Napi::Boolean>().Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { group_size = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { bits = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsString()) { mode = info[nextIdx].As<Napi::String>().Utf8Value(); nextIdx++; }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode, streamArg)));
}

// ---------------------------------------------------------------------------
// gather_qmm(x, w, scales, biases?, lhs_indices?, rhs_indices?, transpose?, group_size?, bits?, stream?)
// ---------------------------------------------------------------------------
Napi::Value GatherQMM(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "gather_qmm expects (x, w, scales, ...)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto x = ToArray(env, info[0]);
  if (env.IsExceptionPending()) return env.Null();
  auto w = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  auto scales = ToArray(env, info[2]);
  if (env.IsExceptionPending()) return env.Null();

  auto parseOptArray = [&](size_t idx) -> std::pair<std::optional<mlx::core::array>, bool> {
    if (info.Length() > idx && info[idx].IsObject() && !info[idx].IsNull() && !info[idx].IsUndefined()) {
      auto* wrapper = Napi::ObjectWrap<ArrayWrapper>::Unwrap(info[idx].As<Napi::Object>());
      if (wrapper) return {wrapper->tensor(), true};
    }
    if (info.Length() > idx && (info[idx].IsNull() || info[idx].IsUndefined())) return {std::nullopt, true};
    return {std::nullopt, false};
  };

  size_t nextIdx = 3;
  auto [biases, hasBiases] = parseOptArray(nextIdx);
  if (hasBiases) nextIdx++;
  auto [lhs_indices, hasLhs] = parseOptArray(nextIdx);
  if (hasLhs) nextIdx++;
  auto [rhs_indices, hasRhs] = parseOptArray(nextIdx);
  if (hasRhs) nextIdx++;

  bool transpose = true;
  int group_size = 64;
  int bits = 4;
  std::string mode = "affine";
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) { transpose = info[nextIdx].As<Napi::Boolean>().Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { group_size = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsNumber()) { bits = info[nextIdx].As<Napi::Number>().Int32Value(); nextIdx++; }
  if (info.Length() > nextIdx && info[nextIdx].IsString()) { mode = info[nextIdx].As<Napi::String>().Utf8Value(); nextIdx++; }
  bool sorted_indices = false;
  if (info.Length() > nextIdx && info[nextIdx].IsBoolean()) { sorted_indices = info[nextIdx].As<Napi::Boolean>().Value(); nextIdx++; }
  auto streamArg = GetStreamArgument(info, nextIdx);
  if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(
      mlx::core::gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode, sorted_indices, streamArg)));
}

// ---------------------------------------------------------------------------
// Linalg ops (mlx::core::linalg::*)
// CPU-only: default to CPU device when no stream specified
// ---------------------------------------------------------------------------
static mlx::core::StreamOrDevice LinalgStreamDefault(const Napi::CallbackInfo& info, size_t index) {
  if (info.Length() <= index) {
    return mlx::core::Device::cpu;
  }
  return ParseStreamOrDeviceValue(info.Env(), info[index]);
}

Napi::Value LinalgInv(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::inv(a, s)));
}

Napi::Value LinalgPinv(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::pinv(a, s)));
}

Napi::Value LinalgSolve(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 2); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::solve(a, b, s)));
}

Napi::Value LinalgSolveTriangular(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]); if (env.IsExceptionPending()) return env.Null();
  bool upper = false; size_t ni = 2;
  if (info.Length() > ni && info[ni].IsBoolean()) { upper = info[ni].As<Napi::Boolean>().Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::solve_triangular(a, b, upper, s)));
}

Napi::Value LinalgCholesky(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  bool upper = false; size_t ni = 1;
  if (info.Length() > ni && info[ni].IsBoolean()) { upper = info[ni].As<Napi::Boolean>().Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::cholesky(a, upper, s)));
}

Napi::Value LinalgCholeskyInv(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  bool upper = false; size_t ni = 1;
  if (info.Length() > ni && info[ni].IsBoolean()) { upper = info[ni].As<Napi::Boolean>().Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::cholesky_inv(a, upper, s)));
}

Napi::Value LinalgTriInv(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  bool upper = false; size_t ni = 1;
  if (info.Length() > ni && info[ni].IsBoolean()) { upper = info[ni].As<Napi::Boolean>().Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::tri_inv(a, upper, s)));
}

Napi::Value LinalgSvd(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  auto result = mlx::core::linalg::svd(a, s);
  auto js = Napi::Array::New(env, result.size());
  for (size_t i = 0; i < result.size(); i++)
    js.Set(i, WrapArray(env, std::make_shared<mlx::core::array>(std::move(result[i]))));
  return js;
}

Napi::Value LinalgQr(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  auto [Q, R] = mlx::core::linalg::qr(a, s);
  auto js = Napi::Array::New(env, 2);
  js.Set((uint32_t)0, WrapArray(env, std::make_shared<mlx::core::array>(std::move(Q))));
  js.Set((uint32_t)1, WrapArray(env, std::make_shared<mlx::core::array>(std::move(R))));
  return js;
}

Napi::Value LinalgLu(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  auto result = mlx::core::linalg::lu(a, s);
  auto js = Napi::Array::New(env, result.size());
  for (size_t i = 0; i < result.size(); i++)
    js.Set(i, WrapArray(env, std::make_shared<mlx::core::array>(std::move(result[i]))));
  return js;
}

Napi::Value LinalgLuFactor(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  auto [lu, pivots] = mlx::core::linalg::lu_factor(a, s);
  auto js = Napi::Array::New(env, 2);
  js.Set((uint32_t)0, WrapArray(env, std::make_shared<mlx::core::array>(std::move(lu))));
  js.Set((uint32_t)1, WrapArray(env, std::make_shared<mlx::core::array>(std::move(pivots))));
  return js;
}

Napi::Value LinalgEig(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  auto [vals, vecs] = mlx::core::linalg::eig(a, s);
  auto js = Napi::Array::New(env, 2);
  js.Set((uint32_t)0, WrapArray(env, std::make_shared<mlx::core::array>(std::move(vals))));
  js.Set((uint32_t)1, WrapArray(env, std::make_shared<mlx::core::array>(std::move(vecs))));
  return js;
}

Napi::Value LinalgEigvals(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto s = LinalgStreamDefault(info, 1); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::eigvals(a, s)));
}

Napi::Value LinalgEigh(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  std::string UPLO = "L"; size_t ni = 1;
  if (info.Length() > ni && info[ni].IsString()) { UPLO = info[ni].As<Napi::String>().Utf8Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  auto [vals, vecs] = mlx::core::linalg::eigh(a, UPLO, s);
  auto js = Napi::Array::New(env, 2);
  js.Set((uint32_t)0, WrapArray(env, std::make_shared<mlx::core::array>(std::move(vals))));
  js.Set((uint32_t)1, WrapArray(env, std::make_shared<mlx::core::array>(std::move(vecs))));
  return js;
}

Napi::Value LinalgEigvalsh(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  std::string UPLO = "L"; size_t ni = 1;
  if (info.Length() > ni && info[ni].IsString()) { UPLO = info[ni].As<Napi::String>().Utf8Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::eigvalsh(a, UPLO, s)));
}

Napi::Value LinalgCross(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto a = ToArray(env, info[0]); if (env.IsExceptionPending()) return env.Null();
  auto b = ToArray(env, info[1]); if (env.IsExceptionPending()) return env.Null();
  int axis = -1; size_t ni = 2;
  if (info.Length() > ni && info[ni].IsNumber()) { axis = info[ni].As<Napi::Number>().Int32Value(); ni++; }
  auto s = LinalgStreamDefault(info, ni); if (env.IsExceptionPending()) return env.Null();
  return WrapArray(env, std::make_shared<mlx::core::array>(mlx::core::linalg::cross(a, b, axis, s)));
}

// ============================================================
// Transform infrastructure
// ============================================================

// Base class for type-erased callable data stored on wrapped JS functions
struct TransformFnData {
  virtual ~TransformFnData() = default;
  virtual Napi::Value call(const Napi::CallbackInfo& info) = 0;
};

// Single dispatch function for all wrapped transform functions
static Napi::Value TransformFnDispatch(const Napi::CallbackInfo& info) {
  auto* data = static_cast<TransformFnData*>(info.Data());
  return data->call(info);
}

// Create a JS function from a TransformFnData, with GC-safe cleanup
static Napi::Function MakeTransformFn(
    Napi::Env env, TransformFnData* data, const char* name) {
  auto jsFn = Napi::Function::New(env, TransformFnDispatch, name, data);
  auto ext = Napi::External<TransformFnData>::New(env, data,
    [](Napi::Env, TransformFnData* p) { delete p; });
  jsFn.Set("_prevent_gc", ext);
  return jsFn;
}

// Wrap a JS function as C++ std::function<vector<array>(const vector<array>&)>
// Each input array is passed as a separate argument to the JS function.
static std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>
WrapJsFn(Napi::Function jsFn) {
  auto fnRef = std::make_shared<Napi::FunctionReference>(Napi::Persistent(jsFn));
  return [fnRef](const std::vector<mlx::core::array>& inputs)
      -> std::vector<mlx::core::array> {
    auto env = fnRef->Env();
    std::vector<napi_value> jsArgs;
    jsArgs.reserve(inputs.size());
    for (const auto& input : inputs) {
      jsArgs.push_back(WrapArray(env, std::make_shared<mlx::core::array>(input)));
    }
    auto result = fnRef->Call(jsArgs);
    if (env.IsExceptionPending()) {
      throw std::runtime_error("JavaScript function threw an error");
    }
    std::vector<mlx::core::array> outputs;
    if (result.IsArray()) {
      auto jsArr = result.As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++) {
        outputs.push_back(ToArray(env, jsArr.Get(i)));
      }
    } else {
      outputs.push_back(ToArray(env, result));
    }
    return outputs;
  };
}

// Scalar variant: wrap JS fn as vector<array> -> array (single output)
static std::function<mlx::core::array(const std::vector<mlx::core::array>&)>
WrapJsFnScalar(Napi::Function jsFn) {
  auto multi = WrapJsFn(jsFn);
  return [multi](const std::vector<mlx::core::array>& inputs) -> mlx::core::array {
    auto outputs = multi(inputs);
    if (outputs.empty()) throw std::runtime_error("Function must return at least one array");
    return outputs[0];
  };
}

// Parse argnums from JS arg at index idx (number, array of numbers, or default {0})
static std::vector<int> ParseArgnums(const Napi::CallbackInfo& info, size_t idx) {
  if (info.Length() <= idx) return {0};
  if (info[idx].IsNumber()) return {info[idx].As<Napi::Number>().Int32Value()};
  if (info[idx].IsArray()) {
    auto jsArr = info[idx].As<Napi::Array>();
    std::vector<int> argnums;
    for (uint32_t i = 0; i < jsArr.Length(); i++)
      argnums.push_back(jsArr.Get(i).As<Napi::Number>().Int32Value());
    return argnums;
  }
  return {0};
}

// ============================================================
// Transform implementations
// ============================================================

Napi::Value EnableCompile(const Napi::CallbackInfo& info) {
  mlx::core::enable_compile();
  return info.Env().Undefined();
}

Napi::Value DisableCompile(const Napi::CallbackInfo& info) {
  mlx::core::disable_compile();
  return info.Env().Undefined();
}

// Callable data for functions that return vector<array>
struct ArrayFnData : TransformFnData {
  explicit ArrayFnData(std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> f) : fn(std::move(f)) {}
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> fn;
  Napi::Value call(const Napi::CallbackInfo& info) override {
    auto env = info.Env();
    std::vector<mlx::core::array> inputs;
    for (size_t i = 0; i < info.Length(); i++) {
      inputs.push_back(ToArray(env, info[i]));
      if (env.IsExceptionPending()) return env.Null();
    }
    try {
      auto result = fn(inputs);
      if (result.size() == 1)
        return WrapArray(env, std::make_shared<mlx::core::array>(result[0]));
      auto jsArr = Napi::Array::New(env, result.size());
      for (size_t i = 0; i < result.size(); i++)
        jsArr.Set(uint32_t(i), WrapArray(env, std::make_shared<mlx::core::array>(result[i])));
      return jsArr;
    } catch (const std::exception& e) {
      if (!env.IsExceptionPending())
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
      return env.Null();
    }
  }
};

// Callable data for value_and_grad (returns [scalar_value, [grads...]])
struct ValueAndGradData : TransformFnData {
  explicit ValueAndGradData(mlx::core::SimpleValueAndGradFn f) : fn(std::move(f)) {}
  mlx::core::SimpleValueAndGradFn fn;
  Napi::Value call(const Napi::CallbackInfo& info) override {
    auto env = info.Env();
    std::vector<mlx::core::array> inputs;
    for (size_t i = 0; i < info.Length(); i++) {
      inputs.push_back(ToArray(env, info[i]));
      if (env.IsExceptionPending()) return env.Null();
    }
    try {
      auto [value, grads] = fn(inputs);
      auto result = Napi::Array::New(env, 2);
      result.Set(uint32_t(0), WrapArray(env, std::make_shared<mlx::core::array>(value)));
      auto jsGrads = Napi::Array::New(env, grads.size());
      for (size_t i = 0; i < grads.size(); i++)
        jsGrads.Set(uint32_t(i), WrapArray(env, std::make_shared<mlx::core::array>(grads[i])));
      result.Set(uint32_t(1), jsGrads);
      return result;
    } catch (const std::exception& e) {
      if (!env.IsExceptionPending())
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
      return env.Null();
    }
  }
};

// grad(fn, argnums?) -> fn
Napi::Value GradOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "grad requires a function argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto scalarFn = WrapJsFnScalar(info[0].As<Napi::Function>());
    auto argnums = ParseArgnums(info, 1);
    auto gradFn = mlx::core::grad(scalarFn, argnums);
    return MakeTransformFn(env, new ArrayFnData{std::move(gradFn)}, "grad");
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// value_and_grad(fn, argnums?) -> fn
Napi::Value ValueAndGradOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "value_and_grad requires a function argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto scalarFn = WrapJsFnScalar(info[0].As<Napi::Function>());
    auto argnums = ParseArgnums(info, 1);
    auto vgFn = mlx::core::value_and_grad(scalarFn, argnums);
    return MakeTransformFn(env, new ValueAndGradData{std::move(vgFn)}, "value_and_grad");
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// vjp(fn, primals, cotangents) -> [outputs, vjps]
Napi::Value VjpOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "vjp requires (fn, primals, cotangents)").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto multiFn = WrapJsFn(info[0].As<Napi::Function>());
    // Parse primals and cotangents (arrays of MLXArrays)
    std::vector<mlx::core::array> primals, cotangents;
    if (info[1].IsArray()) {
      auto jsArr = info[1].As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++) {
        primals.push_back(ToArray(env, jsArr.Get(i)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      primals.push_back(ToArray(env, info[1]));
      if (env.IsExceptionPending()) return env.Null();
    }
    if (info[2].IsArray()) {
      auto jsArr = info[2].As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++) {
        cotangents.push_back(ToArray(env, jsArr.Get(i)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      cotangents.push_back(ToArray(env, info[2]));
      if (env.IsExceptionPending()) return env.Null();
    }
    auto [outputs, vjps] = mlx::core::vjp(multiFn, primals, cotangents);
    auto result = Napi::Array::New(env, 2);
    auto jsOutputs = Napi::Array::New(env, outputs.size());
    for (size_t i = 0; i < outputs.size(); i++)
      jsOutputs.Set(uint32_t(i), WrapArray(env, std::make_shared<mlx::core::array>(outputs[i])));
    auto jsVjps = Napi::Array::New(env, vjps.size());
    for (size_t i = 0; i < vjps.size(); i++)
      jsVjps.Set(uint32_t(i), WrapArray(env, std::make_shared<mlx::core::array>(vjps[i])));
    result.Set(uint32_t(0), jsOutputs);
    result.Set(uint32_t(1), jsVjps);
    return result;
  } catch (const std::exception& e) {
    if (!env.IsExceptionPending())
      Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// jvp(fn, primals, tangents) -> [outputs, jvps]
Napi::Value JvpOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "jvp requires (fn, primals, tangents)").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto multiFn = WrapJsFn(info[0].As<Napi::Function>());
    std::vector<mlx::core::array> primals, tangents;
    if (info[1].IsArray()) {
      auto jsArr = info[1].As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++) {
        primals.push_back(ToArray(env, jsArr.Get(i)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      primals.push_back(ToArray(env, info[1]));
      if (env.IsExceptionPending()) return env.Null();
    }
    if (info[2].IsArray()) {
      auto jsArr = info[2].As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++) {
        tangents.push_back(ToArray(env, jsArr.Get(i)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      tangents.push_back(ToArray(env, info[2]));
      if (env.IsExceptionPending()) return env.Null();
    }
    auto [outputs, jvps] = mlx::core::jvp(multiFn, primals, tangents);
    auto result = Napi::Array::New(env, 2);
    auto jsOutputs = Napi::Array::New(env, outputs.size());
    for (size_t i = 0; i < outputs.size(); i++)
      jsOutputs.Set(uint32_t(i), WrapArray(env, std::make_shared<mlx::core::array>(outputs[i])));
    auto jsJvps = Napi::Array::New(env, jvps.size());
    for (size_t i = 0; i < jvps.size(); i++)
      jsJvps.Set(uint32_t(i), WrapArray(env, std::make_shared<mlx::core::array>(jvps[i])));
    result.Set(uint32_t(0), jsOutputs);
    result.Set(uint32_t(1), jsJvps);
    return result;
  } catch (const std::exception& e) {
    if (!env.IsExceptionPending())
      Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// vmap(fn, in_axes?, out_axes?) -> fn
Napi::Value VmapOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "vmap requires a function argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto multiFn = WrapJsFn(info[0].As<Napi::Function>());
    std::vector<int> in_axes, out_axes;
    if (info.Length() > 1 && info[1].IsArray()) {
      auto jsArr = info[1].As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++)
        in_axes.push_back(jsArr.Get(i).As<Napi::Number>().Int32Value());
    }
    if (info.Length() > 2 && info[2].IsArray()) {
      auto jsArr = info[2].As<Napi::Array>();
      for (uint32_t i = 0; i < jsArr.Length(); i++)
        out_axes.push_back(jsArr.Get(i).As<Napi::Number>().Int32Value());
    }
    auto vmapFn = mlx::core::vmap(multiFn, in_axes, out_axes);
    return MakeTransformFn(env, new ArrayFnData{std::move(vmapFn)}, "vmap");
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// compile(fn, shapeless?) -> fn
Napi::Value CompileOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "compile requires a function argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto multiFn = WrapJsFn(info[0].As<Napi::Function>());
    bool shapeless = false;
    if (info.Length() > 1 && info[1].IsBoolean())
      shapeless = info[1].As<Napi::Boolean>().Value();
    auto compiledFn = mlx::core::compile(multiFn, shapeless);
    return MakeTransformFn(env, new ArrayFnData{std::move(compiledFn)}, "compiled");
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// checkpoint(fn) -> fn
Napi::Value CheckpointOp(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsFunction()) {
    Napi::TypeError::New(env, "checkpoint requires a function argument").ThrowAsJavaScriptException();
    return env.Null();
  }
  try {
    auto multiFn = WrapJsFn(info[0].As<Napi::Function>());
    auto cpFn = mlx::core::checkpoint(std::move(multiFn));
    return MakeTransformFn(env, new ArrayFnData{std::move(cpFn)}, "checkpointed");
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// ============================================================
// Eval ops (mlx::core::eval / async_eval)
// ============================================================

Napi::Value Eval(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  std::vector<mlx::core::array> arrays;
  for (size_t i = 0; i < info.Length(); i++) {
    if (info[i].IsArray()) {
      auto jsArr = info[i].As<Napi::Array>();
      for (uint32_t j = 0; j < jsArr.Length(); j++) {
        arrays.push_back(ToArray(env, jsArr.Get(j)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      arrays.push_back(ToArray(env, info[i]));
      if (env.IsExceptionPending()) return env.Null();
    }
  }
  try {
    mlx::core::eval(std::move(arrays));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
  return env.Undefined();
}

Napi::Value AsyncEval(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  std::vector<mlx::core::array> arrays;
  for (size_t i = 0; i < info.Length(); i++) {
    if (info[i].IsArray()) {
      auto jsArr = info[i].As<Napi::Array>();
      for (uint32_t j = 0; j < jsArr.Length(); j++) {
        arrays.push_back(ToArray(env, jsArr.Get(j)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      arrays.push_back(ToArray(env, info[i]));
      if (env.IsExceptionPending()) return env.Null();
    }
  }
  try {
    mlx::core::async_eval(std::move(arrays));
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
  return env.Undefined();
}

// ============================================================
// IO ops (mlx::core::load / save / safetensors / gguf)
// ============================================================

// Helper: convert JS object {key: MLXArray} to C++ unordered_map
static std::unordered_map<std::string, mlx::core::array> JsObjectToArrayMap(
    Napi::Env env, Napi::Value val) {
  std::unordered_map<std::string, mlx::core::array> result;
  if (!val.IsObject()) {
    Napi::TypeError::New(env, "Expected object with string keys and array values")
        .ThrowAsJavaScriptException();
    return result;
  }
  auto obj = val.As<Napi::Object>();
  auto names = obj.GetPropertyNames();
  for (uint32_t i = 0; i < names.Length(); i++) {
    auto key = names.Get(i).As<Napi::String>().Utf8Value();
    auto arr = ToArray(env, obj.Get(key));
    if (env.IsExceptionPending()) return result;
    result.insert_or_assign(key, std::move(arr));
  }
  return result;
}

// Helper: convert C++ unordered_map to JS object {key: MLXArray}
static Napi::Object ArrayMapToJsObject(
    Napi::Env env,
    const std::unordered_map<std::string, mlx::core::array>& map) {
  auto obj = Napi::Object::New(env);
  for (auto& [key, arr] : map) {
    obj.Set(key, WrapArray(env, std::make_shared<mlx::core::array>(arr)));
  }
  return obj;
}

// Helper: convert C++ string map to JS object
static Napi::Object StringMapToJsObject(
    Napi::Env env,
    const std::unordered_map<std::string, std::string>& map) {
  auto obj = Napi::Object::New(env);
  for (auto& [key, val] : map) {
    obj.Set(key, Napi::String::New(env, val));
  }
  return obj;
}

// load(file: string, options?: {stream?}) -> MLXArray | {arrays, metadata}
// MLX auto-detects format from extension (.npy, .npz, .safetensors, .gguf)
Napi::Value Load(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "load requires a file path string").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string file = info[0].As<Napi::String>().Utf8Value();
  auto s = GetStreamArgument(info, 1);
  if (env.IsExceptionPending()) return env.Null();

  try {
    // Check extension to determine format
    std::string ext;
    auto dot = file.rfind('.');
    if (dot != std::string::npos) ext = file.substr(dot);

    if (ext == ".safetensors") {
      auto loaded = mlx::core::load_safetensors(file, s);
      auto result = Napi::Object::New(env);
      result.Set("arrays", ArrayMapToJsObject(env, loaded.first));
      result.Set("metadata", StringMapToJsObject(env, loaded.second));
      return result;
    } else if (ext == ".gguf") {
      auto loaded = mlx::core::load_gguf(file, s);
      auto result = Napi::Object::New(env);
      result.Set("arrays", ArrayMapToJsObject(env, loaded.first));
      // Convert GGUFMetaData variant to JS
      auto metaObj = Napi::Object::New(env);
      for (const auto& kv : loaded.second) {
        if (std::holds_alternative<std::string>(kv.second)) {
          metaObj.Set(kv.first, Napi::String::New(env, std::get<std::string>(kv.second)));
        } else if (std::holds_alternative<mlx::core::array>(kv.second)) {
          metaObj.Set(kv.first, WrapArray(env, std::make_shared<mlx::core::array>(std::get<mlx::core::array>(kv.second))));
        } else if (std::holds_alternative<std::vector<std::string>>(kv.second)) {
          const auto& vec = std::get<std::vector<std::string>>(kv.second);
          auto jsArr = Napi::Array::New(env, vec.size());
          for (size_t i = 0; i < vec.size(); i++) {
            jsArr.Set(i, Napi::String::New(env, vec[i]));
          }
          metaObj.Set(kv.first, jsArr);
        }
        // monostate: skip
      }
      result.Set("metadata", metaObj);
      return result;
    } else {
      // .npy or other single-array format
      auto arr = mlx::core::load(file, s);
      return WrapArray(env, std::make_shared<mlx::core::array>(arr));
    }
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// save(file: string, arr: MLXArray)
Napi::Value Save(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2 || !info[0].IsString()) {
    Napi::TypeError::New(env, "save requires (file: string, array: MLXArray)").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string file = info[0].As<Napi::String>().Utf8Value();
  auto arr = ToArray(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();
  try {
    mlx::core::save(file, arr);
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
  return env.Undefined();
}

// save_safetensors(file: string, arrays: {[key: string]: MLXArray}, metadata?: {[key: string]: string})
Napi::Value SaveSafetensors(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2 || !info[0].IsString()) {
    Napi::TypeError::New(env, "save_safetensors requires (file: string, arrays: object)").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string file = info[0].As<Napi::String>().Utf8Value();
  auto arrays = JsObjectToArrayMap(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  std::unordered_map<std::string, std::string> metadata;
  if (info.Length() > 2 && info[2].IsObject()) {
    auto metaObj = info[2].As<Napi::Object>();
    auto names = metaObj.GetPropertyNames();
    for (uint32_t i = 0; i < names.Length(); i++) {
      auto key = names.Get(i).As<Napi::String>().Utf8Value();
      metadata[key] = metaObj.Get(key).As<Napi::String>().Utf8Value();
    }
  }

  try {
    mlx::core::save_safetensors(file, arrays, metadata);
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
  return env.Undefined();
}

// save_gguf(file: string, arrays: {[key: string]: MLXArray})
Napi::Value SaveGguf(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2 || !info[0].IsString()) {
    Napi::TypeError::New(env, "save_gguf requires (file: string, arrays: object)").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string file = info[0].As<Napi::String>().Utf8Value();
  auto arrays = JsObjectToArrayMap(env, info[1]);
  if (env.IsExceptionPending()) return env.Null();

  try {
    mlx::core::save_gguf(file, arrays);
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
  return env.Undefined();
}

// ============================================================
// Export ops
// ============================================================

// export_function(file, fn, args, shapeless?)
Napi::Value ExportFunction(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3 || !info[0].IsString() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "export_function requires (file: string, fn: Function, args: MLXArray[])")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string file = info[0].As<Napi::String>().Utf8Value();
  auto multiFn = WrapJsFn(info[1].As<Napi::Function>());
  // Parse args array
  std::vector<mlx::core::array> args;
  if (info[2].IsArray()) {
    auto jsArr = info[2].As<Napi::Array>();
    for (uint32_t i = 0; i < jsArr.Length(); i++) {
      args.push_back(ToArray(env, jsArr.Get(i)));
      if (env.IsExceptionPending()) return env.Null();
    }
  } else {
    args.push_back(ToArray(env, info[2]));
    if (env.IsExceptionPending()) return env.Null();
  }
  bool shapeless = false;
  if (info.Length() > 3 && info[3].IsBoolean())
    shapeless = info[3].As<Napi::Boolean>().Value();
  try {
    mlx::core::export_function(file, multiFn, args, shapeless);
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
  return env.Undefined();
}

// export_to_dot(arrays) -> string (DOT format)
Napi::Value ExportToDot(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  std::vector<mlx::core::array> outputs;
  for (size_t i = 0; i < info.Length(); i++) {
    if (info[i].IsArray()) {
      auto jsArr = info[i].As<Napi::Array>();
      for (uint32_t j = 0; j < jsArr.Length(); j++) {
        outputs.push_back(ToArray(env, jsArr.Get(j)));
        if (env.IsExceptionPending()) return env.Null();
      }
    } else {
      outputs.push_back(ToArray(env, info[i]));
      if (env.IsExceptionPending()) return env.Null();
    }
  }
  try {
    std::ostringstream oss;
    mlx::core::export_to_dot(oss, outputs);
    return Napi::String::New(env, oss.str());
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

} // namespace

// ============================================================================
// ArrayBuilderWrapper
//
// A streaming row-at-a-time tensor construction path.  Allocates exactly
// one mlx::core::allocator::malloc buffer upfront sized for the full tensor.
// JS feeds one row (a TypedArray of the row stride) at a time via append_row().
// build() wraps the already-populated buffer in an mlx::core::array with no
// further copies.
//
// JS API (exposed as core.array_builder):
//
//   const b = mx.core.array_builder('float32', [rows, cols]);
//   for (const row of source) {          // row: Float32Array [cols]
//     b.append_row(row);
//   }
//   const t = b.build();                 // returns MLXArray
//
// ============================================================================
class ArrayBuilderWrapper : public Napi::ObjectWrap<ArrayBuilderWrapper> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "ArrayBuilder",
        {
            InstanceMethod("append_row", &ArrayBuilderWrapper::AppendRow),
            InstanceMethod("build",      &ArrayBuilderWrapper::Build),
        });
    exports.Set("array_builder",
                Napi::Function::New(env, &ArrayBuilderWrapper::Create,
                                    "array_builder"));
    auto& data = mlx::node::GetAddonData(env);
    data.array_builder_constructor = Napi::Persistent(func);
  }

  /** JS factory: array_builder(dtype, shape). */
  static Napi::Value Create(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    auto& addon = mlx::node::GetAddonData(env);
    if (info.Length() < 2) {
      Napi::TypeError::New(
          env, "array_builder(dtype, shape): expected 2 arguments")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    // arg0: dtype string or MLXDtype object
    if (!IsDtypeArg(env, info[0], addon)) {
      Napi::TypeError::New(env, "array_builder: first argument must be a dtype")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    auto dtype = MaybeParseDtype(env, info[0], mlx::core::float32, addon);
    if (env.IsExceptionPending()) return env.Null();

    // arg1: shape array
    if (!info[1].IsArray()) {
      Napi::TypeError::New(env, "array_builder: second argument must be a shape array")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    auto shapeJs = info[1].As<Napi::Array>();
    mlx::core::Shape shape;
    size_t total = 1;
    for (uint32_t i = 0; i < shapeJs.Length(); ++i) {
      auto dimVal = shapeJs.Get(i);
      if (!dimVal.IsNumber()) {
        Napi::TypeError::New(env, "array_builder: shape entries must be numbers")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
      int64_t dim = dimVal.As<Napi::Number>().Int64Value();
      if (dim < 0) {
        Napi::RangeError::New(env, "array_builder: shape dimensions must be non-negative")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
      shape.push_back(static_cast<mlx::core::ShapeElem>(dim));
      total *= static_cast<size_t>(dim);
    }

    size_t itemsize = mlx::core::size_of(dtype);
    size_t nbytes   = total * itemsize;
    size_t row_stride = (shape.size() > 1)
        ? static_cast<size_t>(shape.back()) * itemsize
        : nbytes; // 1D: entire array is one "row"

    // Pre-allocate the MLX buffer for the full tensor.
    mlx::core::allocator::Buffer buf = mlx::core::allocator::malloc(nbytes);

    // Construct via the DefineClass constructor using Napi::External to carry
    // the builder state.
    struct BuilderState {
      mlx::core::allocator::Buffer buf;
      mlx::core::Shape             shape;
      mlx::core::Dtype             dtype;
      size_t                       nbytes;
      size_t                       row_stride;   ///< bytes per row
      size_t                       written;      ///< bytes written so far
    };
    auto* state    = new BuilderState{buf, shape, dtype, nbytes, row_stride, 0};
    auto  external = Napi::External<BuilderState>::New(
        env, state, [](Napi::Env, BuilderState* s) { delete s; });

    return addon.array_builder_constructor.New({external});
  }

  explicit ArrayBuilderWrapper(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<ArrayBuilderWrapper>(info) {
    auto env = info.Env();
    if (info.Length() != 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "ArrayBuilder cannot be constructed directly")
          .ThrowAsJavaScriptException();
      return;
    }
    struct BuilderState {
      mlx::core::allocator::Buffer buf;
      mlx::core::Shape             shape;
      mlx::core::Dtype             dtype;
      size_t                       nbytes;
      size_t                       row_stride;
      size_t                       written;
    };
    auto* state = info[0].As<Napi::External<BuilderState>>().Data();
    buf_        = state->buf;
    shape_      = state->shape;
    dtype_      = state->dtype;
    nbytes_     = state->nbytes;
    row_stride_ = state->row_stride;
    written_    = state->written;
    built_      = false;
  }

  /** Append one row (TypedArray) into the pre-allocated MLX buffer. */
  Napi::Value AppendRow(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (built_) {
      Napi::Error::New(env, "array_builder: build() has already been called")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
      Napi::TypeError::New(env, "append_row: expected a TypedArray")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    auto typed = info[0].As<Napi::TypedArray>();
    const size_t nbytes_row =
        typed.ElementLength() * typed.ElementSize();

    if (nbytes_row != row_stride_) {
      Napi::RangeError::New(
          env,
          "append_row: TypedArray byte length " +
              std::to_string(nbytes_row) +
              " does not match expected row stride " +
              std::to_string(row_stride_))
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    if (written_ + nbytes_row > nbytes_) {
      Napi::RangeError::New(env, "append_row: buffer is full")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    const uint8_t* src =
        static_cast<const uint8_t*>(typed.ArrayBuffer().Data()) +
        typed.ByteOffset();
    uint8_t* dst = static_cast<uint8_t*>(buf_.raw_ptr()) + written_;
    std::memcpy(dst, src, nbytes_row);
    written_ += nbytes_row;
    return env.Undefined();
  }

  /** Finalise and return the MLXArray wrapping the pre-allocated buffer. */
  Napi::Value Build(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (built_) {
      Napi::Error::New(env, "array_builder: build() may only be called once")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    if (written_ != nbytes_) {
      Napi::RangeError::New(
          env,
          "array_builder: buffer not fully written (" +
              std::to_string(written_) + " of " +
              std::to_string(nbytes_) + " bytes)")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    built_ = true;
    // Wrap the buffer directly — zero copy.
    auto tensor = std::make_shared<mlx::core::array>(buf_, shape_, dtype_);
    return WrapArray(env, tensor);
  }

 private:
  mlx::core::allocator::Buffer buf_{nullptr};
  mlx::core::Shape             shape_;
  mlx::core::Dtype             dtype_{mlx::core::float32};
  size_t                       nbytes_{0};
  size_t                       row_stride_{0};
  size_t                       written_{0};
  bool                         built_{false};
};

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Defer GPU initialization to the first op to avoid loader-time hazards.
  auto& data = mlx::node::GetAddonData(env);

  // Build namespace: mlx.core.* with dtype constants, Array, ops.
  Napi::Object mlx = Napi::Object::New(env);
  Napi::Object core = Napi::Object::New(env);

  // Keep public surface Python-parity; diagnostics live in labs (not exported
  // here)

  // Dtype and stream bindings first (so dtype constructors exist before ops
  // parse dtype)
  mlx::node::InitDtype(env, core, data);
  mlx::node::InitDtype(env, mlx, data);
  mlx::node::InitStreamBindings(env, core, data);

  // Classes and ops under core
  ArrayWrapper::Init(env, core);
  ArrayBuilderWrapper::Init(env, core);
  core.Set("array", Napi::Function::New(env, ArrayFactory, "array", &data));
  core.Set("from_js_array", Napi::Function::New(env, FromJSArray, "from_js_array", &data));
  core.Set("array_equal", Napi::Function::New(env, ArrayEqual, "array_equal", &data));
  core.Set("asarray", Napi::Function::New(env, AsArray, "asarray", &data));
  core.Set("zeros", Napi::Function::New(env, Zeros, "zeros", &data));
  core.Set(
      "zeros_like", Napi::Function::New(env, ZerosLike, "zeros_like", &data));
  core.Set("ones", Napi::Function::New(env, Ones, "ones", &data));
  core.Set("ones_like", Napi::Function::New(env, OnesLike, "ones_like", &data));
  core.Set("full", Napi::Function::New(env, Full, "full", &data));
  core.Set("reshape", Napi::Function::New(env, Reshape, "reshape", &data));
  core.Set(
      "transpose", Napi::Function::New(env, Transpose, "transpose", &data));
  core.Set("moveaxis", Napi::Function::New(env, MoveAxis, "moveaxis", &data));
  core.Set("swapaxes", Napi::Function::New(env, SwapAxes, "swapaxes", &data));
  core.Set("arange", Napi::Function::New(env, Arange, "arange", &data));
  // Note: normal is registered under core.random.normal (see below)
  core.Set("add", Napi::Function::New(env, Add, "add", &data));
  core.Set("multiply", Napi::Function::New(env, Multiply, "multiply", &data));
  core.Set("subtract", Napi::Function::New(env, Subtract, "subtract", &data));
  core.Set("matmul", Napi::Function::New(env, Matmul, "matmul", &data));
  core.Set("where", Napi::Function::New(env, Where, "where", &data));
  core.Set("tan", Napi::Function::New(env, Tan, "tan", &data));
  core.Set("sin", Napi::Function::New(env, Sin, "sin", &data));
  core.Set("cos", Napi::Function::New(env, Cos, "cos", &data));
  core.Set("arcsin", Napi::Function::New(env, Arcsin, "arcsin", &data));
  core.Set("arccos", Napi::Function::New(env, Arccos, "arccos", &data));
  core.Set("arctan", Napi::Function::New(env, Arctan, "arctan", &data));
  core.Set("arctan2", Napi::Function::New(env, Arctan2, "arctan2", &data));
  core.Set("rsqrt", Napi::Function::New(env, Rsqrt, "rsqrt", &data));
  core.Set("square", Napi::Function::New(env, Square, "square", &data));
  core.Set("sign", Napi::Function::New(env, Sign, "sign", &data));
  core.Set("abs", Napi::Function::New(env, Abs, "abs", &data));
  core.Set("sqrt", Napi::Function::New(env, Sqrt, "sqrt", &data));
  core.Set("exp", Napi::Function::New(env, Exp, "exp", &data));
  core.Set("log", Napi::Function::New(env, Log, "log", &data));
  core.Set("divide", Napi::Function::New(env, Divide, "divide", &data));
  core.Set("power", Napi::Function::New(env, Power, "power", &data));
  core.Set("equal", Napi::Function::New(env, Equal, "equal", &data));
  core.Set("not_equal", Napi::Function::New(env, NotEqual, "not_equal", &data));
  core.Set("less", Napi::Function::New(env, Less, "less", &data));
  core.Set("less_equal", Napi::Function::New(env, LessEqual, "less_equal", &data));
  core.Set("greater", Napi::Function::New(env, Greater, "greater", &data));
  core.Set("greater_equal", Napi::Function::New(env, GreaterEqual, "greater_equal", &data));
  core.Set("maximum", Napi::Function::New(env, Maximum, "maximum", &data));
  core.Set("minimum", Napi::Function::New(env, Minimum, "minimum", &data));

  // Reduction ops
  core.Set("sum", Napi::Function::New(env, Sum, "sum", &data));
  core.Set("mean", Napi::Function::New(env, Mean, "mean", &data));
  core.Set("min", Napi::Function::New(env, Min, "min", &data));
  core.Set("max", Napi::Function::New(env, Max, "max", &data));
  core.Set("prod", Napi::Function::New(env, Prod, "prod", &data));
  core.Set("argmin", Napi::Function::New(env, Argmin, "argmin", &data));
  core.Set("argmax", Napi::Function::New(env, Argmax, "argmax", &data));
  core.Set("logsumexp", Napi::Function::New(env, LogSumExp, "logsumexp", &data));
  core.Set("logcumsumexp", Napi::Function::New(env, LogCumSumExp, "logcumsumexp", &data));
  core.Set("softmax", Napi::Function::New(env, Softmax, "softmax", &data));

  // Additional math ops
  core.Set("logaddexp", Napi::Function::New(env, LogAddExp, "logaddexp", &data));
  core.Set("clip", Napi::Function::New(env, Clip, "clip", &data));
  core.Set("log1p", Napi::Function::New(env, Log1p, "log1p", &data));
  core.Set("negative", Napi::Function::New(env, Negative, "negative", &data));
  core.Set("reciprocal", Napi::Function::New(env, Reciprocal, "reciprocal", &data));

  // Shape manipulation
  core.Set("expand_dims", Napi::Function::New(env, ExpandDims, "expand_dims", &data));
  core.Set("squeeze", Napi::Function::New(env, Squeeze, "squeeze", &data));
  core.Set("concatenate", Napi::Function::New(env, Concatenate, "concatenate", &data));

  // Activation primitives
  core.Set("sigmoid", Napi::Function::New(env, Sigmoid, "sigmoid", &data));
  core.Set("erf", Napi::Function::New(env, Erf, "erf", &data));
  core.Set("tanh", Napi::Function::New(env, Tanh, "tanh", &data));
  core.Set("split", Napi::Function::New(env, Split, "split", &data));

  // Shape manipulation / stacking
  core.Set("stack", Napi::Function::New(env, Stack, "stack", &data));

  // Convolution ops
  core.Set("conv1d", Napi::Function::New(env, Conv1d, "conv1d", &data));
  core.Set("conv2d", Napi::Function::New(env, Conv2d, "conv2d", &data));
  core.Set("conv3d", Napi::Function::New(env, Conv3d, "conv3d", &data));

  // Indexing
  core.Set("take", Napi::Function::New(env, Take, "take", &data));
  core.Set("take_along_axis", Napi::Function::New(env, TakeAlongAxis, "take_along_axis", &data));
  core.Set("pad", Napi::Function::New(env, Pad, "pad", &data));
  core.Set("slice", Napi::Function::New(env, Slice, "slice", &data));
  core.Set("astype", Napi::Function::New(env, Astype, "astype", &data));
  core.Set("as_strided", Napi::Function::New(env, AsStrided, "as_strided", &data));
  core.Set("number_of_elements", Napi::Function::New(env, NumberOfElements, "number_of_elements", &data));

  // Batch: unary math ops
  core.Set("ceil", Napi::Function::New(env, Ceil, "ceil", &data));
  core.Set("floor", Napi::Function::New(env, Floor, "floor", &data));
  core.Set("round", Napi::Function::New(env, Round, "round", &data));
  core.Set("isnan", Napi::Function::New(env, IsNan, "isnan", &data));
  core.Set("isinf", Napi::Function::New(env, IsInf, "isinf", &data));
  core.Set("isfinite", Napi::Function::New(env, IsFinite, "isfinite", &data));
  core.Set("logical_not", Napi::Function::New(env, LogicalNot, "logical_not", &data));
  core.Set("sinh", Napi::Function::New(env, Sinh, "sinh", &data));
  core.Set("cosh", Napi::Function::New(env, Cosh, "cosh", &data));
  core.Set("arcsinh", Napi::Function::New(env, ArcSinh, "arcsinh", &data));
  core.Set("arccosh", Napi::Function::New(env, ArcCosh, "arccosh", &data));
  core.Set("arctanh", Napi::Function::New(env, ArcTanh, "arctanh", &data));
  core.Set("degrees", Napi::Function::New(env, Degrees, "degrees", &data));
  core.Set("radians", Napi::Function::New(env, Radians, "radians", &data));
  core.Set("erfinv", Napi::Function::New(env, ErfInv, "erfinv", &data));
  core.Set("expm1", Napi::Function::New(env, Expm1, "expm1", &data));

  // Batch: cumulative ops
  core.Set("cumsum", Napi::Function::New(env, CumSum, "cumsum", &data));
  core.Set("cumprod", Napi::Function::New(env, CumProd, "cumprod", &data));
  core.Set("cummax", Napi::Function::New(env, CumMax, "cummax", &data));
  core.Set("cummin", Napi::Function::New(env, CumMin, "cummin", &data));

  // Batch: binary ops
  core.Set("floor_divide", Napi::Function::New(env, FloorDivide, "floor_divide", &data));
  core.Set("remainder", Napi::Function::New(env, Remainder, "remainder", &data));
  core.Set("logical_and", Napi::Function::New(env, LogicalAnd, "logical_and", &data));
  core.Set("logical_or", Napi::Function::New(env, LogicalOr, "logical_or", &data));
  core.Set("bitwise_and", Napi::Function::New(env, BitwiseAnd, "bitwise_and", &data));
  core.Set("bitwise_or", Napi::Function::New(env, BitwiseOr, "bitwise_or", &data));
  core.Set("bitwise_xor", Napi::Function::New(env, BitwiseXor, "bitwise_xor", &data));
  core.Set("left_shift", Napi::Function::New(env, LeftShift, "left_shift", &data));
  core.Set("right_shift", Napi::Function::New(env, RightShift, "right_shift", &data));

  // Batch: reduction & query ops
  core.Set("all", Napi::Function::New(env, All, "all", &data));
  core.Set("any", Napi::Function::New(env, Any, "any", &data));
  core.Set("array_equal", Napi::Function::New(env, ArrayEqual, "array_equal", &data));

  // Batch: shape & creation ops
  core.Set("flatten", Napi::Function::New(env, Flatten, "flatten", &data));
  core.Set("eye", Napi::Function::New(env, Eye, "eye", &data));
  core.Set("identity", Napi::Function::New(env, Identity, "identity", &data));
  core.Set("linspace", Napi::Function::New(env, Linspace, "linspace", &data));
  core.Set("tril", Napi::Function::New(env, Tril, "tril", &data));
  core.Set("triu", Napi::Function::New(env, Triu, "triu", &data));
  core.Set("broadcast_to", Napi::Function::New(env, BroadcastTo, "broadcast_to", &data));
  core.Set("repeat", Napi::Function::New(env, Repeat, "repeat", &data));
  core.Set("tile", Napi::Function::New(env, Tile, "tile", &data));
  core.Set("sort", Napi::Function::New(env, Sort, "sort", &data));
  core.Set("argsort", Napi::Function::New(env, ArgSort, "argsort", &data));
  core.Set("diag", Napi::Function::New(env, Diag, "diag", &data));
  core.Set("diagonal", Napi::Function::New(env, Diagonal, "diagonal", &data));
  core.Set("topk", Napi::Function::New(env, TopK, "topk", &data));

  // Batch 3A: simple unary ops
  core.Set("log2", Napi::Function::New(env, Log2, "log2", &data));
  core.Set("log10", Napi::Function::New(env, Log10, "log10", &data));
  core.Set("isposinf", Napi::Function::New(env, IsPosInf, "isposinf", &data));
  core.Set("isneginf", Napi::Function::New(env, IsNegInf, "isneginf", &data));
  core.Set("bitwise_invert", Napi::Function::New(env, BitwiseInvert, "bitwise_invert", &data));
  core.Set("conjugate", Napi::Function::New(env, Conjugate, "conjugate", &data));
  core.Set("real", Napi::Function::New(env, Real, "real", &data));
  core.Set("imag", Napi::Function::New(env, Imag, "imag", &data));
  core.Set("stop_gradient", Napi::Function::New(env, StopGradient, "stop_gradient", &data));

  // Batch 3B: simple binary ops
  core.Set("outer", Napi::Function::New(env, Outer, "outer", &data));
  core.Set("inner", Napi::Function::New(env, Inner, "inner", &data));
  core.Set("kron", Napi::Function::New(env, Kron, "kron", &data));

  // Batch 3B: parameterized ops
  core.Set("nan_to_num", Napi::Function::New(env, NanToNum, "nan_to_num", &data));
  core.Set("allclose", Napi::Function::New(env, AllClose, "allclose", &data));
  core.Set("isclose", Napi::Function::New(env, IsClose, "isclose", &data));
  core.Set("view", Napi::Function::New(env, View, "view", &data));
  core.Set("contiguous", Napi::Function::New(env, Contiguous, "contiguous", &data));
  core.Set("hadamard_transform", Napi::Function::New(env, HadamardTransform, "hadamard_transform", &data));
  core.Set("unflatten", Napi::Function::New(env, Unflatten, "unflatten", &data));
  core.Set("partition", Napi::Function::New(env, Partition, "partition", &data));
  core.Set("argpartition", Napi::Function::New(env, ArgPartition, "argpartition", &data));
  core.Set("put_along_axis", Napi::Function::New(env, PutAlongAxis, "put_along_axis", &data));
  core.Set("roll", Napi::Function::New(env, Roll, "roll", &data));
  core.Set("tri", Napi::Function::New(env, Tri, "tri", &data));

  // Batch 3C: multi-return and complex ops
  core.Set("meshgrid", Napi::Function::New(env, Meshgrid, "meshgrid", &data));
  core.Set("broadcast_arrays", Napi::Function::New(env, BroadcastArrays, "broadcast_arrays", &data));
  core.Set("atleast_1d", Napi::Function::New(env, AtLeast1d, "atleast_1d", &data));
  core.Set("atleast_2d", Napi::Function::New(env, AtLeast2d, "atleast_2d", &data));
  core.Set("atleast_3d", Napi::Function::New(env, AtLeast3d, "atleast_3d", &data));
  core.Set("slice_update", Napi::Function::New(env, SliceUpdate, "slice_update", &data));
  core.Set("conv_general", Napi::Function::New(env, ConvGeneral, "conv_general", &data));
  core.Set("conv_transpose1d", Napi::Function::New(env, ConvTranspose1d, "conv_transpose1d", &data));
  core.Set("conv_transpose2d", Napi::Function::New(env, ConvTranspose2d, "conv_transpose2d", &data));
  core.Set("conv_transpose3d", Napi::Function::New(env, ConvTranspose3d, "conv_transpose3d", &data));
  core.Set("einsum", Napi::Function::New(env, Einsum, "einsum", &data));
  core.Set("tensordot", Napi::Function::New(env, Tensordot, "tensordot", &data));
  core.Set("block_masked_mm", Napi::Function::New(env, BlockMaskedMM, "block_masked_mm", &data));
  core.Set("gather_mm", Napi::Function::New(env, GatherMM, "gather_mm", &data));
  core.Set("segmented_mm", Napi::Function::New(env, SegmentedMM, "segmented_mm", &data));
  core.Set("quantize", Napi::Function::New(env, Quantize, "quantize", &data));
  core.Set("dequantize", Napi::Function::New(env, Dequantize, "dequantize", &data));
  core.Set("quantized_matmul", Napi::Function::New(env, QuantizedMatmul, "quantized_matmul", &data));
  core.Set("gather_qmm", Napi::Function::New(env, GatherQMM, "gather_qmm", &data));

  // fast namespace
  Napi::Object fast = Napi::Object::New(env);
  fast.Set("scaled_dot_product_attention",
           Napi::Function::New(env, ScaledDotProductAttention,
                               "scaled_dot_product_attention", &data));
  core.Set("fast", fast);

  // Fused linear algebra ops
  core.Set("addmm", Napi::Function::New(env, Addmm, "addmm", &data));

  // Additional reduction ops
  core.Set("var", Napi::Function::New(env, Var, "var", &data));
  core.Set("variance", Napi::Function::New(env, Var, "variance", &data));
  core.Set("norm", Napi::Function::New(env, Norm, "norm", &data));
  core.Set("trace", Napi::Function::New(env, Trace, "trace", &data));

  // NN init functions
  Napi::Object nn_init = Napi::Object::New(env);
  nn_init.Set("sparse", Napi::Function::New(env, Sparse, "sparse", &data));

  // Random operations under core.random (matches mlx.core.random.*)
  Napi::Object random = Napi::Object::New(env);
  random.Set("uniform", Napi::Function::New(env, RandomUniform, "uniform", &data));
  random.Set("normal", Napi::Function::New(env, Normal, "normal", &data));
  random.Set("bernoulli", Napi::Function::New(env, Bernoulli, "bernoulli", &data));
  random.Set("seed", Napi::Function::New(env, RandomSeed, "seed", &data));
  random.Set("key", Napi::Function::New(env, RandomKey, "key", &data));
  random.Set("split", Napi::Function::New(env, RandomSplit, "split", &data));
  random.Set("randint", Napi::Function::New(env, RandomRandint, "randint", &data));
  random.Set("categorical", Napi::Function::New(env, RandomCategorical, "categorical", &data));
  random.Set("permutation", Napi::Function::New(env, RandomPermutation, "permutation", &data));
  random.Set("gumbel", Napi::Function::New(env, RandomGumbel, "gumbel", &data));
  random.Set("laplace", Napi::Function::New(env, RandomLaplace, "laplace", &data));
  random.Set("truncated_normal", Napi::Function::New(env, RandomTruncatedNormal, "truncated_normal", &data));
  random.Set("multivariate_normal", Napi::Function::New(env, RandomMultivariateNormal, "multivariate_normal", &data));
  core.Set("random", random);

  // Linear algebra operations under core.linalg (matches mlx.core.linalg.*)
  Napi::Object linalg = Napi::Object::New(env);
  linalg.Set("norm", Napi::Function::New(env, Norm, "norm", &data));
  linalg.Set("inv", Napi::Function::New(env, LinalgInv, "inv", &data));
  linalg.Set("pinv", Napi::Function::New(env, LinalgPinv, "pinv", &data));
  linalg.Set("solve", Napi::Function::New(env, LinalgSolve, "solve", &data));
  linalg.Set("solve_triangular", Napi::Function::New(env, LinalgSolveTriangular, "solve_triangular", &data));
  linalg.Set("cholesky", Napi::Function::New(env, LinalgCholesky, "cholesky", &data));
  linalg.Set("cholesky_inv", Napi::Function::New(env, LinalgCholeskyInv, "cholesky_inv", &data));
  linalg.Set("tri_inv", Napi::Function::New(env, LinalgTriInv, "tri_inv", &data));
  linalg.Set("svd", Napi::Function::New(env, LinalgSvd, "svd", &data));
  linalg.Set("qr", Napi::Function::New(env, LinalgQr, "qr", &data));
  linalg.Set("lu", Napi::Function::New(env, LinalgLu, "lu", &data));
  linalg.Set("lu_factor", Napi::Function::New(env, LinalgLuFactor, "lu_factor", &data));
  linalg.Set("eig", Napi::Function::New(env, LinalgEig, "eig", &data));
  linalg.Set("eigvals", Napi::Function::New(env, LinalgEigvals, "eigvals", &data));
  linalg.Set("eigh", Napi::Function::New(env, LinalgEigh, "eigh", &data));
  linalg.Set("eigvalsh", Napi::Function::New(env, LinalgEigvalsh, "eigvalsh", &data));
  linalg.Set("cross", Napi::Function::New(env, LinalgCross, "cross", &data));
  core.Set("linalg", linalg);

  // Device management
  core.Set("default_device", Napi::Function::New(env, DefaultDevice, "default_device", &data));
  core.Set("set_default_device", Napi::Function::New(env, SetDefaultDevice, "set_default_device", &data));
  core.Set("is_available", Napi::Function::New(env, IsAvailable, "is_available", &data));

  // Memory management
  core.Set("clear_cache", Napi::Function::New(env, ClearCache, "clear_cache", &data));
  core.Set("get_active_memory", Napi::Function::New(env, GetActiveMemory, "get_active_memory", &data));
  core.Set("get_cache_memory", Napi::Function::New(env, GetCacheMemory, "get_cache_memory", &data));
  core.Set("get_peak_memory", Napi::Function::New(env, GetPeakMemory, "get_peak_memory", &data));
  core.Set("reset_peak_memory", Napi::Function::New(env, ResetPeakMemory, "reset_peak_memory", &data));
  core.Set("set_cache_limit", Napi::Function::New(env, SetCacheLimit, "set_cache_limit", &data));
  core.Set("set_memory_limit", Napi::Function::New(env, SetMemoryLimit, "set_memory_limit", &data));
  core.Set("set_wired_limit", Napi::Function::New(env, SetWiredLimit, "set_wired_limit", &data));

  // FFT namespace
  Napi::Object fft = Napi::Object::New(env);
  fft.Set("fft", Napi::Function::New(env, FFT, "fft", &data));
  fft.Set("ifft", Napi::Function::New(env, IFFT, "ifft", &data));
  fft.Set("fft2", Napi::Function::New(env, FFT2, "fft2", &data));
  fft.Set("ifft2", Napi::Function::New(env, IFFT2, "ifft2", &data));
  fft.Set("fftn", Napi::Function::New(env, FFTn, "fftn", &data));
  fft.Set("ifftn", Napi::Function::New(env, IFFTn, "ifftn", &data));
  fft.Set("rfft", Napi::Function::New(env, RFFT, "rfft", &data));
  fft.Set("irfft", Napi::Function::New(env, IRFFT, "irfft", &data));
  fft.Set("rfft2", Napi::Function::New(env, RFFT2, "rfft2", &data));
  fft.Set("irfft2", Napi::Function::New(env, IRFFT2, "irfft2", &data));
  fft.Set("rfftn", Napi::Function::New(env, RFFTn, "rfftn", &data));
  fft.Set("irfftn", Napi::Function::New(env, IRFFTn, "irfftn", &data));
  fft.Set("fftshift", Napi::Function::New(env, FFTShift, "fftshift", &data));
  fft.Set("ifftshift", Napi::Function::New(env, IFFTShift, "ifftshift", &data));
  core.Set("fft", fft);

  // Fast ops (add to existing fast namespace)
  fast.Set("rms_norm", Napi::Function::New(env, FastRmsNorm, "rms_norm", &data));
  fast.Set("layer_norm", Napi::Function::New(env, FastLayerNorm, "layer_norm", &data));
  fast.Set("rope", Napi::Function::New(env, FastRope, "rope", &data));

  // Export/import operations
  core.Set("import_function", Napi::Function::New(env, ImportFunction, "import_function", &data));
  core.Set("export_function", Napi::Function::New(env, ExportFunction, "export_function", &data));
  core.Set("export_to_dot", Napi::Function::New(env, ExportToDot, "export_to_dot", &data));

  // Eval operations
  core.Set("eval", Napi::Function::New(env, Eval, "eval", &data));
  core.Set("async_eval", Napi::Function::New(env, AsyncEval, "async_eval", &data));

  // Transform operations
  core.Set("grad", Napi::Function::New(env, GradOp, "grad", &data));
  core.Set("value_and_grad", Napi::Function::New(env, ValueAndGradOp, "value_and_grad", &data));
  core.Set("vjp", Napi::Function::New(env, VjpOp, "vjp", &data));
  core.Set("jvp", Napi::Function::New(env, JvpOp, "jvp", &data));
  core.Set("vmap", Napi::Function::New(env, VmapOp, "vmap", &data));
  core.Set("compile", Napi::Function::New(env, CompileOp, "compile", &data));
  core.Set("enable_compile", Napi::Function::New(env, EnableCompile, "enable_compile", &data));
  core.Set("disable_compile", Napi::Function::New(env, DisableCompile, "disable_compile", &data));
  core.Set("checkpoint", Napi::Function::New(env, CheckpointOp, "checkpoint", &data));

  // IO operations
  core.Set("load", Napi::Function::New(env, Load, "load", &data));
  core.Set("save", Napi::Function::New(env, Save, "save", &data));
  core.Set("save_safetensors", Napi::Function::New(env, SaveSafetensors, "save_safetensors", &data));
  core.Set("save_gguf", Napi::Function::New(env, SaveGguf, "save_gguf", &data));

  mlx.Set("core", core);
  mlx.Set("nn_init", nn_init);

  // Return top-level mlx namespace (not nested under exports.mlx)
  return mlx;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, Init)
