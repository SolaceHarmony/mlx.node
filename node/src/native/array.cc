#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <napi.h>

#include "addon_data.h"
#include "mlx/array.h"
#include "mlx/dtype_utils.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/device.h"
#include "mlx_bridge.h"

#include "dtype.h"
#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"
#include "stream.h"
#include "runtime.h"

namespace {

Napi::Object WrapArray(Napi::Env env, std::shared_ptr<mlx::core::array> tensor);

// Forward declarations for helpers used by factories
bool IsDtypeArg(Napi::Env env, const Napi::Value& value);
mlx::core::Dtype MaybeParseDtype(
    Napi::Env env,
    const Napi::Value& value,
    mlx::core::Dtype fallback);

const std::unordered_map<std::string, mlx::core::Dtype>& DtypeLookup() {
  static const std::unordered_map<std::string, mlx::core::Dtype> mapping = {
      {"bool", mlx::core::bool_},       {"int8", mlx::core::int8},
      {"int16", mlx::core::int16},       {"int32", mlx::core::int32},
      {"int64", mlx::core::int64},       {"uint8", mlx::core::uint8},
      {"uint16", mlx::core::uint16},     {"uint32", mlx::core::uint32},
      {"uint64", mlx::core::uint64},     {"float16", mlx::core::float16},
      {"bfloat16", mlx::core::bfloat16}, {"float32", mlx::core::float32},
      {"float64", mlx::core::float64},   {"complex64", mlx::core::complex64},
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

    auto external = info[0].As<
        Napi::External<std::shared_ptr<mlx::bridge::ArrayHolder>>>();
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

  template <typename T>
  static std::vector<T> CopyTypedArray(const Napi::TypedArray& typed) {
    size_t length = typed.ElementLength();
    std::vector<T> output(length);
    std::memcpy(
        output.data(),
        static_cast<const uint8_t*>(typed.ArrayBuffer().Data()) +
            typed.ByteOffset(),
        length * sizeof(T));
    return output;
  }

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
          Napi::TypeError::New(
              env,
              "float32 dtype requires Float32Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::float64:
        if (typed.TypedArrayType() != napi_float64_array) {
          Napi::TypeError::New(
              env,
              "float64 dtype requires Float64Array input")
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
          Napi::TypeError::New(env, "uint64 dtype requires BigUint64Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        break;
      case mlx::core::complex64:
        if (typed.TypedArrayType() != napi_float32_array) {
          Napi::TypeError::New(env, "complex64 dtype requires Float32Array input")
              .ThrowAsJavaScriptException();
          return {};
        }
        if (elementCount * 2 != typedLength) {
          Napi::RangeError::New(
              env,
              "complex64 expects data length equal to element count * 2")
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

    switch (dtype) {
      case mlx::core::float32: {
        auto host = CopyTypedArray<float>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::float64: {
        auto host = CopyTypedArray<double>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::int8: {
        auto host = CopyTypedArray<int8_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::uint8: {
        auto host = CopyTypedArray<uint8_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::bool_: {
        std::vector<bool> host(typedLength);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(rawData);
        for (size_t i = 0; i < typedLength; ++i) {
          host[i] = src[i] != 0;
        }
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::int16: {
        auto host = CopyTypedArray<int16_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::uint16: {
        auto host = CopyTypedArray<uint16_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::float16: {
        std::vector<mlx::core::float16_t> host(typedLength);
        std::memcpy(host.data(), rawData, typedLength * sizeof(uint16_t));
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::bfloat16: {
        std::vector<mlx::core::bfloat16_t> host(typedLength);
        std::memcpy(host.data(), rawData, typedLength * sizeof(uint16_t));
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::int32: {
        auto host = CopyTypedArray<int32_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::uint32: {
        auto host = CopyTypedArray<uint32_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::int64: {
        auto host = CopyTypedArray<int64_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::uint64: {
        auto host = CopyTypedArray<uint64_t>(typed);
        return mlx::core::array(host.begin(), shape, dtype);
      }
      case mlx::core::complex64: {
        const float* src = reinterpret_cast<const float*>(rawData);
        std::vector<mlx::core::complex64_t> host(elementCount);
        for (size_t i = 0; i < elementCount; ++i) {
          host[i] = mlx::core::complex64_t(src[2 * i], src[2 * i + 1]);
        }
        return mlx::core::array(host.begin(), shape, dtype);
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
        env, typed.As<Napi::Float32Array>(), info[1].As<Napi::Array>(),
        mlx::core::float32);
    if (!tensor) {
      return env.Null();
    }
    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(*tensor)));
  }

  static Napi::Value FromTypedArray(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env, "Expected TypedArray and shape array")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    std::optional<mlx::core::Dtype> dtype;
    if (info.Length() >= 3 && !info[2].IsUndefined() && info[2].IsString()) {
      dtype = ParseDtype(env, info[2].As<Napi::String>().Utf8Value());
      if (env.IsExceptionPending()) {
        return env.Null();
      }
    }

    auto tensor = MakeArrayFromTyped(
        env,
        info[0].As<Napi::TypedArray>(),
        info[1].As<Napi::Array>(),
        dtype);
    if (!tensor) {
      return env.Null();
    }
    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(*tensor)));
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
    return Napi::String::New(env, DtypeToString(tensor().dtype()));
  }

  Napi::Value Eval(const Napi::CallbackInfo& info) {
    holder_->array.eval();
    return info.Env().Undefined();
  }

  Napi::Value ToFloat32Array(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    auto& arr = const_cast<mlx::core::array&>(tensor());
    arr.eval();

    const auto length = arr.size();
    Napi::ArrayBuffer buffer =
        Napi::ArrayBuffer::New(env, length * sizeof(float));
    Napi::Float32Array jsArray =
        Napi::Float32Array::New(env, length, buffer, 0);
    std::memcpy(buffer.Data(), arr.data<float>(), length * sizeof(float));
    return jsArray;
  }

  Napi::Value ToTypedArray(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    auto& arr = const_cast<mlx::core::array&>(tensor());
    arr.eval();

    const auto dtype = arr.dtype();
    const size_t length = arr.size();

    switch (dtype) {
      case mlx::core::float32:
        return ToFloat32Array(info);
      case mlx::core::float64: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(double));
        auto jsArray = Napi::Float64Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<double>(), length * sizeof(double));
        return jsArray;
      }
      case mlx::core::int8: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(int8_t));
        auto jsArray = Napi::Int8Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<int8_t>(), length * sizeof(int8_t));
        return jsArray;
      }
      case mlx::core::uint8:
      case mlx::core::bool_: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(uint8_t));
        auto jsArray = Napi::Uint8Array::New(env, length, buffer, 0);
        if (dtype == mlx::core::bool_) {
          auto src = arr.data<bool>();
          auto dest = static_cast<uint8_t*>(buffer.Data());
          for (size_t i = 0; i < length; ++i) {
            dest[i] = src[i] ? 1 : 0;
          }
        } else {
          std::memcpy(buffer.Data(), arr.data<uint8_t>(), length * sizeof(uint8_t));
        }
        return jsArray;
      }
      case mlx::core::int16: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(int16_t));
        auto jsArray = Napi::Int16Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<int16_t>(), length * sizeof(int16_t));
        return jsArray;
      }
      case mlx::core::uint16: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(uint16_t));
        auto jsArray = Napi::Uint16Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<uint16_t>(), length * sizeof(uint16_t));
        return jsArray;
      }
      case mlx::core::float16: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(uint16_t));
        auto jsArray = Napi::Uint16Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<mlx::core::float16_t>(),
                    length * sizeof(uint16_t));
        return jsArray;
      }
      case mlx::core::bfloat16: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(uint16_t));
        auto jsArray = Napi::Uint16Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<mlx::core::bfloat16_t>(),
                    length * sizeof(uint16_t));
        return jsArray;
      }
      case mlx::core::int32: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(int32_t));
        auto jsArray = Napi::Int32Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<int32_t>(), length * sizeof(int32_t));
        return jsArray;
      }
      case mlx::core::uint32: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(uint32_t));
        auto jsArray = Napi::Uint32Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<uint32_t>(), length * sizeof(uint32_t));
        return jsArray;
      }
      case mlx::core::int64: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(int64_t));
        auto jsArray = Napi::BigInt64Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<int64_t>(), length * sizeof(int64_t));
        return jsArray;
      }
      case mlx::core::uint64: {
        auto buffer = Napi::ArrayBuffer::New(env, length * sizeof(uint64_t));
        auto jsArray = Napi::BigUint64Array::New(env, length, buffer, 0);
        std::memcpy(buffer.Data(), arr.data<uint64_t>(), length * sizeof(uint64_t));
        return jsArray;
      }
      case mlx::core::complex64: {
        auto buffer = Napi::ArrayBuffer::New(env, length * 2 * sizeof(float));
        auto jsArray = Napi::Float32Array::New(env, length * 2, buffer, 0);
        auto src = arr.data<mlx::core::complex64_t>();
        auto dest = static_cast<float*>(buffer.Data());
        for (size_t i = 0; i < length; ++i) {
          dest[2 * i] = src[i].real();
          dest[2 * i + 1] = src[i].imag();
        }
        return jsArray;
      }
      default:
        Napi::TypeError::New(env, "Unsupported dtype for toTypedArray")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
  }

  std::shared_ptr<mlx::bridge::ArrayHolder> holder_;
};

// Static member definition placed after the class so it can call the private helper.
std::optional<mlx::core::array> ArrayWrapper::BuildFromTyped(
    Napi::Env env,
    const Napi::TypedArray& typed,
    const Napi::Array& shapeArray,
    std::optional<mlx::core::Dtype> requestedDtype) {
  return MakeArrayFromTyped(env, typed, shapeArray, requestedDtype);
}
mlx::core::Shape ParseShapeArgument(Napi::Env env, const Napi::Value& value) {
  if (!value.IsArray()) {
    Napi::TypeError::New(env, "Shape must be an array of integers")
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
    Napi::TypeError::New(env, "Device must be a string or object with type/index")
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

mlx::core::StreamOrDevice GetStreamArgument(
    const Napi::CallbackInfo& info,
    size_t index) {
  if (info.Length() <= index) {
    return {};
  }
  return ParseStreamOrDeviceValue(info.Env(), info[index]);
}

std::vector<int> ParseAxisVector(
    Napi::Env env,
    const Napi::Value& value,
    const char* name) {
  std::vector<int> axes;
  if (value.IsNumber()) {
    axes.push_back(value.As<Napi::Number>().Int32Value());
    return axes;
  }
  if (!value.IsArray()) {
    Napi::TypeError::New(env, std::string(name) + " must be an array of integers")
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
  auto external = Napi::External<std::shared_ptr<mlx::core::array>>::New(
      env,
      new std::shared_ptr<mlx::core::array>(std::move(tensor)),
      [](Napi::Env, std::shared_ptr<mlx::core::array>* data) {
        delete data;
      });
  return data.array_constructor.New({external});
}

Napi::Value ArrayFactory(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "array expects a TypedArray as first argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto typed = info[0].As<Napi::TypedArray>();
  if (info.Length() < 2 || !info[1].IsArray()) {
    Napi::TypeError::New(env, "array(typed, shape[, dtype][, stream]) expected")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto shapeArray = info[1].As<Napi::Array>();
  std::optional<mlx::core::Dtype> maybeDtype;
  size_t streamIndex = 2;
  if (info.Length() >= 3 && IsDtypeArg(env, info[2])) {
    maybeDtype = MaybeParseDtype(env, info[2], mlx::core::float32);
    if (env.IsExceptionPending()) return env.Null();
    streamIndex = 3;
  }
  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending()) return env.Null();

  auto arrOpt = ArrayWrapper::BuildFromTyped(env, typed, shapeArray, maybeDtype);
  if (!arrOpt.has_value()) return env.Null();
  std::shared_ptr<mlx::core::array> tensor;
  if (!std::holds_alternative<std::monostate>(streamArg)) {
    // If a stream/device was provided, move array to that stream's device
    tensor = std::make_shared<mlx::core::array>(mlx::core::copy(arrOpt.value(), streamArg));
  } else {
    tensor = std::make_shared<mlx::core::array>(std::move(arrOpt.value()));
  }
  return WrapArray(env, tensor);
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
    mlx::core::Dtype fallback) {
  if (value.IsUndefined() || value.IsNull()) {
    return fallback;
  }
  // Accept a real mlx.core.Dtype object first by reading its key()
  auto& data = mlx::node::GetAddonData(env);
  if (value.IsObject()) {
    auto obj = value.As<Napi::Object>();
    auto ctor = data.dtype_constructor.Value();
    if (!ctor.IsEmpty() && obj.InstanceOf(ctor)) {
      auto keyFn = obj.Get("key");
      if (keyFn.IsFunction()) {
        auto keyVal = keyFn.As<Napi::Function>().Call(obj, {});
        if (keyVal.IsString()) {
          return ParseDtypeKey(env, keyVal.As<Napi::String>().Utf8Value());
        }
      }
      // Fallback: toString contains 'mlx.core.<key>'
      auto toStr = obj.Get("toString");
      if (toStr.IsFunction()) {
        auto s = toStr.As<Napi::Function>().Call(obj, {});
        if (s.IsString()) {
          std::string full = s.As<Napi::String>().Utf8Value();
          auto pos = full.rfind('.');
          if (pos != std::string::npos && pos + 1 < full.size()) {
            return ParseDtypeKey(env, full.substr(pos + 1));
          }
        }
      }
    }
  }
  // Temporary convenience: allow string key
  if (value.IsString()) {
    return ParseDtypeKey(env, value.As<Napi::String>().Utf8Value());
  }
  Napi::TypeError::New(env, "dtype must be a mlx.core.Dtype")
      .ThrowAsJavaScriptException();
  return fallback;
}

bool IsDtypeArg(Napi::Env env, const Napi::Value& value) {
  if (value.IsUndefined() || value.IsNull()) return false;
  if (value.IsString()) return true; // temporary convenience
  if (!value.IsObject()) return false;
  auto& data = mlx::node::GetAddonData(env);
  auto obj = value.As<Napi::Object>();
  auto ctor = data.dtype_constructor.Value();
  return !ctor.IsEmpty() && obj.InstanceOf(ctor);
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

mlx::core::array MakeFilledArray(
    const mlx::core::Shape& shape,
    double fill,
    mlx::core::Dtype dtype) {
  return mlx::core::full(shape, fill, dtype);
}

Napi::Value Zeros(const Napi::CallbackInfo& info) {
  auto env = info.Env();
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
  // If arg1 is a dtype (preferred) or a string key -> dtype; else stream/device.
  size_t streamIndex = 1;
  if (info.Length() >= 2 && IsDtypeArg(env, info[1])) {
    dtype = MaybeParseDtype(env, info[1], dtype);
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
  if (info.Length() >= 2 && IsDtypeArg(env, info[1])) {
    dtype = MaybeParseDtype(env, info[1], dtype);
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
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "full expects shape and fill value")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto shape = ParseShapeArgument(env, info[0]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  double value = ParseScalarValue<double>(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto dtype = mlx::core::float32;
  size_t streamIndex = 2;
  if (info.Length() >= 3 && IsDtypeArg(env, info[2])) {
    dtype = MaybeParseDtype(env, info[2], dtype);
    if (env.IsExceptionPending()) {
      return env.Null();
    }
    streamIndex = 3;
  }
  auto streamArg = GetStreamArgument(info, streamIndex);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  return WrapArray(env, std::make_shared<mlx::core::array>(
                            mlx::core::full(shape, value, dtype, streamArg)));
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
  auto shape = ParseShapeArgument(env, info[1]);
  if (env.IsExceptionPending()) {
    return env.Null();
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
      Napi::RangeError::New(env, "source and destination axes must have the same length")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    mlx::core::array tensor = wrapper->tensor();
    for (size_t i = 0; i < srcVec.size(); ++i) {
      tensor = mlx::core::moveaxis(tensor, srcVec[i], dstVec[i], streamArg);
    }
    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(tensor)));
  }

  if (!info[1].IsNumber() || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "moveaxis expects numeric source/destination axes")
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

Napi::Value Add(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "add expects two arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* a = UnwrapArray(env, info[0]);
  auto* b = UnwrapArray(env, info[1]);
  if (env.IsExceptionPending() || a == nullptr || b == nullptr) {
    return env.Null();
  }
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::add(a->tensor(), b->tensor(), streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Multiply(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "multiply expects two arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* a = UnwrapArray(env, info[0]);
  auto* b = UnwrapArray(env, info[1]);
  if (env.IsExceptionPending() || a == nullptr || b == nullptr) {
    return env.Null();
  }
  auto streamArg = GetStreamArgument(info, 2);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::multiply(a->tensor(), b->tensor(), streamArg));
  return WrapArray(env, tensor);
}

Napi::Value Where(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 3) {
    Napi::TypeError::New(env, "where expects condition, x, y")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto* condition = UnwrapArray(env, info[0]);
  auto* x = UnwrapArray(env, info[1]);
  auto* y = UnwrapArray(env, info[2]);
  if (env.IsExceptionPending() || condition == nullptr || x == nullptr || y == nullptr) {
    return env.Null();
  }
  auto streamArg = GetStreamArgument(info, 3);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto tensor = std::make_shared<mlx::core::array>(
      mlx::core::where(condition->tensor(), x->tensor(), y->tensor(), streamArg));
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
        out.Set(key, Napi::Number::New(env, static_cast<double>(std::get<size_t>(v))));
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
    auto result = mlx::node::Runtime::Instance().WithMetalPool([&]() -> Napi::Value {
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

Napi::Value CPUSanity(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  try {
    // CPU stream demo: mirror the GPU example on CPU to prove the binding+streams.
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

} // namespace

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Defer GPU initialization to the first op to avoid loader-time hazards.
  auto& data = mlx::node::GetAddonData(env);

  // Build namespace: mlx.core.* with dtype constants, Array, ops.
  Napi::Object mlx = Napi::Object::New(env);
  Napi::Object core = Napi::Object::New(env);

  // Hello lives under core for quick diagnostics
  core.Set("hello", Napi::Function::New(env, Hello));

  // Classes and ops under core
  ArrayWrapper::Init(env, core);
  core.Set("array", Napi::Function::New(env, ArrayFactory, "array", &data));
  core.Set("zeros", Napi::Function::New(env, Zeros, "zeros", &data));
  core.Set("zeros_like", Napi::Function::New(env, ZerosLike, "zeros_like", &data));
  core.Set("ones", Napi::Function::New(env, Ones, "ones", &data));
  core.Set("ones_like", Napi::Function::New(env, OnesLike, "ones_like", &data));
  core.Set("full", Napi::Function::New(env, Full, "full", &data));
  core.Set("reshape", Napi::Function::New(env, Reshape, "reshape", &data));
  core.Set("transpose", Napi::Function::New(env, Transpose, "transpose", &data));
  core.Set("moveaxis", Napi::Function::New(env, MoveAxis, "moveaxis", &data));
  core.Set("swapaxes", Napi::Function::New(env, SwapAxes, "swapaxes", &data));
  core.Set("add", Napi::Function::New(env, Add, "add", &data));
  core.Set("multiply", Napi::Function::New(env, Multiply, "multiply", &data));
  core.Set("where", Napi::Function::New(env, Where, "where", &data));
  core.Set("gpu_info", Napi::Function::New(env, GPUInfo, "gpu_info", &data));
  core.Set("gpu_sanity", Napi::Function::New(env, GPUSanity, "gpu_sanity", &data));
  core.Set("cpu_sanity", Napi::Function::New(env, CPUSanity, "cpu_sanity", &data));

  // Dtype and stream bindings into both mlx.core and mlx (top-level) for parity
  mlx::node::InitDtype(env, core, data);
  mlx::node::InitDtype(env, mlx, data);
  mlx::node::InitStreamBindings(env, core, data);

  mlx.Set("core", core);

  // Return top-level mlx namespace (not nested under exports.mlx)
  return mlx;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, Init)
