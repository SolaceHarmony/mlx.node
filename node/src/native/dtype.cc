#include "dtype.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "addon_data.h"

#include "mlx/dtype.h"
#include "mlx/dtype_utils.h"

namespace mlx::node {
namespace {

using mlx::core::Dtype;

const char* CategoryKey(Dtype::Category category) {
  switch (category) {
    case Dtype::Category::complexfloating:
      return "complexfloating";
    case Dtype::Category::floating:
      return "floating";
    case Dtype::Category::inexact:
      return "inexact";
    case Dtype::Category::signedinteger:
      return "signedinteger";
    case Dtype::Category::unsignedinteger:
      return "unsignedinteger";
    case Dtype::Category::integer:
      return "integer";
    case Dtype::Category::number:
      return "number";
    case Dtype::Category::generic:
      return "generic";
  }
  return "generic";
}

class DtypeCategoryWrapper : public Napi::ObjectWrap<DtypeCategoryWrapper> {
 public:
  static void Define(Napi::Env env, AddonData& data) {
    if (!data.dtype_category_constructor.IsEmpty()) {
      return;
    }

    Napi::Function func = DefineClass(
        env,
        "DtypeCategory",
        {
            InstanceAccessor("name", &DtypeCategoryWrapper::Name, nullptr),
            InstanceAccessor("value", &DtypeCategoryWrapper::Value, nullptr),
            InstanceMethod("equals", &DtypeCategoryWrapper::Equals),
            InstanceMethod("toString", &DtypeCategoryWrapper::ToString),
        });

    data.dtype_category_constructor = Napi::Persistent(func);
  }

  static Napi::Object Create(
      Napi::Env env,
      Dtype::Category category,
      AddonData& data) {
    auto external = Napi::External<Dtype::Category>::New(
        env,
        new Dtype::Category(category),
        [](Napi::Env /*env*/, Dtype::Category* ptr) { delete ptr; });
    return data.dtype_category_constructor.New({external});
  }

  explicit DtypeCategoryWrapper(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<DtypeCategoryWrapper>(info),
        category_(Dtype::Category::generic) {
    auto env = info.Env();
    if (info.Length() != 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "mlx.core.DtypeCategory cannot be constructed directly")
          .ThrowAsJavaScriptException();
      return;
    }
    auto external = info[0].As<Napi::External<Dtype::Category>>();
    category_ = *external.Data();
  }

  Dtype::Category category() const {
    return category_;
  }

 private:
  Napi::Value Name(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), CategoryKey(category_));
  }

  Napi::Value Value(const Napi::CallbackInfo& info) {
    return Napi::Number::New(
        info.Env(), static_cast<uint32_t>(category_));
  }

  Napi::Value Equals(const Napi::CallbackInfo& info) {
    if (info.Length() < 1 || !info[0].IsObject()) {
      return Napi::Boolean::New(info.Env(), false);
    }
    auto other = info[0].As<Napi::Object>();
    auto* other_wrapper =
        Napi::ObjectWrap<DtypeCategoryWrapper>::Unwrap(other);
    if (other_wrapper == nullptr) {
      return Napi::Boolean::New(info.Env(), false);
    }
    return Napi::Boolean::New(
        info.Env(), category_ == other_wrapper->category_);
  }

  Napi::Value ToString(const Napi::CallbackInfo& info) {
    std::string repr = "DtypeCategory.";
    repr += CategoryKey(category_);
    return Napi::String::New(info.Env(), repr);
  }

  Dtype::Category category_;
};

class DtypeWrapper : public Napi::ObjectWrap<DtypeWrapper> {
 public:
  static void Define(Napi::Env env, AddonData& data) {
    if (!data.dtype_constructor.IsEmpty()) {
      return;
    }

    Napi::Function func = DefineClass(
        env,
        "Dtype",
        {
            InstanceAccessor("key", &DtypeWrapper::Key, nullptr),
            InstanceAccessor("size", &DtypeWrapper::Size, nullptr),
            InstanceAccessor("category", &DtypeWrapper::Category, nullptr),
            InstanceMethod("equals", &DtypeWrapper::Equals),
            InstanceMethod("toString", &DtypeWrapper::ToString),
            StaticMethod(
                "fromString",
                &DtypeWrapper::FromString,
                napi_default,
                &data),
        });

    data.dtype_constructor = Napi::Persistent(func);
  }

  static Napi::Object Create(
      Napi::Env env,
      const mlx::core::Dtype& dtype,
      AddonData& data) {
    auto external = Napi::External<mlx::core::Dtype>::New(
        env,
        new mlx::core::Dtype(dtype),
        [](Napi::Env /*env*/, mlx::core::Dtype* ptr) { delete ptr; });
    return data.dtype_constructor.New({external});
  }

  explicit DtypeWrapper(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<DtypeWrapper>(info), dtype_(mlx::core::float32) {
    auto env = info.Env();
    if (info.Length() != 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "mlx.core.Dtype cannot be constructed directly")
          .ThrowAsJavaScriptException();
      return;
    }
    auto external = info[0].As<Napi::External<mlx::core::Dtype>>();
    dtype_ = *external.Data();
  }

  Dtype dtype() const {
    return dtype_;
  }

  static Napi::Value FromString(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    auto* addon_data =
        static_cast<AddonData*>(info.Data());
    if (addon_data == nullptr) {
      Napi::Error::New(env, "AddonData missing for fromString")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    if (info.Length() < 1 || !info[0].IsString()) {
      Napi::TypeError::New(env, "fromString expects a dtype key")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    const auto key = info[0].As<Napi::String>().Utf8Value();
    auto maybe = Lookup(key);
    if (!maybe.has_value()) {
      Napi::TypeError::New(env, "Unknown dtype: " + key)
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    return Create(env, maybe.value(), *addon_data);
  }

 private:
  Napi::Value Key(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), mlx::core::dtype_to_string(dtype_));
  }

  Napi::Value Size(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), mlx::core::size_of(dtype_));
  }

  Napi::Value Category(const Napi::CallbackInfo& info) {
    using namespace mlx::core;
    if (issubdtype(dtype_, complexfloating)) {
      return Napi::String::New(info.Env(), "complexfloating");
    }
    if (issubdtype(dtype_, floating)) {
      return Napi::String::New(info.Env(), "floating");
    }
    if (issubdtype(dtype_, inexact)) {
      return Napi::String::New(info.Env(), "inexact");
    }
    if (issubdtype(dtype_, signedinteger)) {
      return Napi::String::New(info.Env(), "signedinteger");
    }
    if (issubdtype(dtype_, unsignedinteger)) {
      return Napi::String::New(info.Env(), "unsignedinteger");
    }
    if (issubdtype(dtype_, integer)) {
      return Napi::String::New(info.Env(), "integer");
    }
    if (issubdtype(dtype_, number)) {
      return Napi::String::New(info.Env(), "number");
    }
    return Napi::String::New(info.Env(), "generic");
  }

  Napi::Value Equals(const Napi::CallbackInfo& info) {
    if (info.Length() < 1 || !info[0].IsObject()) {
      return Napi::Boolean::New(info.Env(), false);
    }
    auto other = info[0].As<Napi::Object>();
    auto* other_wrapper = Napi::ObjectWrap<DtypeWrapper>::Unwrap(other);
    if (other_wrapper == nullptr) {
      return Napi::Boolean::New(info.Env(), false);
    }
    return Napi::Boolean::New(info.Env(), dtype_ == other_wrapper->dtype_);
  }

  Napi::Value ToString(const Napi::CallbackInfo& info) {
    std::string repr = "mlx.core.";
    repr += mlx::core::dtype_to_string(dtype_);
    return Napi::String::New(info.Env(), repr);
  }

  static std::optional<mlx::core::Dtype> Lookup(const std::string& key) {
    static const std::unordered_map<std::string, mlx::core::Dtype> kMap = {
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

    auto it = kMap.find(key);
    if (it != kMap.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  mlx::core::Dtype dtype_;
};

std::optional<mlx::core::Dtype> UnwrapDtype(
    const Napi::Value& value,
    AddonData& data) {
  if (!value.IsObject()) {
    return std::nullopt;
  }
  auto obj = value.As<Napi::Object>();
  auto ctor = data.dtype_constructor.Value();
  if (ctor.IsEmpty() || !obj.InstanceOf(ctor)) {
    return std::nullopt;
  }
  auto* wrapper = Napi::ObjectWrap<DtypeWrapper>::Unwrap(obj);
  if (wrapper == nullptr) {
    return std::nullopt;
  }
  return wrapper->dtype();
}

std::optional<mlx::core::Dtype::Category> UnwrapCategory(
    const Napi::Value& value,
    AddonData& data) {
  if (!value.IsObject()) {
    return std::nullopt;
  }
  auto obj = value.As<Napi::Object>();
  auto ctor = data.dtype_category_constructor.Value();
  if (ctor.IsEmpty() || !obj.InstanceOf(ctor)) {
    return std::nullopt;
  }
  auto* wrapper = Napi::ObjectWrap<DtypeCategoryWrapper>::Unwrap(obj);
  if (wrapper == nullptr) {
    return std::nullopt;
  }
  return wrapper->category();
}

Napi::Value Issubdtype(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon_data = static_cast<AddonData*>(info.Data());
  if (addon_data == nullptr) {
    Napi::Error::New(env, "AddonData missing for issubdtype")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "issubdtype expects two arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  const auto lhs_dtype = UnwrapDtype(info[0], *addon_data);
  const auto lhs_category = UnwrapCategory(info[0], *addon_data);
  const auto rhs_dtype = UnwrapDtype(info[1], *addon_data);
  const auto rhs_category = UnwrapCategory(info[1], *addon_data);

  if ((!lhs_dtype.has_value() && !lhs_category.has_value()) ||
      (!rhs_dtype.has_value() && !rhs_category.has_value())) {
    Napi::TypeError::New(
        env,
        "issubdtype arguments must be mlx.core Dtype or DtypeCategory")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  bool result = false;
  if (lhs_dtype.has_value() && rhs_dtype.has_value()) {
    result = mlx::core::issubdtype(lhs_dtype.value(), rhs_dtype.value());
  } else if (lhs_dtype.has_value() && rhs_category.has_value()) {
    result = mlx::core::issubdtype(
        lhs_dtype.value(), rhs_category.value());
  } else if (lhs_category.has_value() && rhs_dtype.has_value()) {
    result =
        mlx::core::issubdtype(lhs_category.value(), rhs_dtype.value());
  } else {
    result =
        mlx::core::issubdtype(lhs_category.value(), rhs_category.value());
  }
  return Napi::Boolean::New(env, result);
}

} // namespace

void InitDtype(Napi::Env env, Napi::Object exports, AddonData& data) {
  DtypeWrapper::Define(env, data);
  DtypeCategoryWrapper::Define(env, data);

  Napi::Object dtype_ns = Napi::Object::New(env);

  const std::vector<std::pair<std::string, mlx::core::Dtype>> entries = {
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

  for (const auto& entry : entries) {
    auto dtype_obj = DtypeWrapper::Create(env, entry.second, data);
    dtype_ns.Set(entry.first, dtype_obj);
    exports.Set(entry.first, dtype_obj);
  }

  dtype_ns.Set(
      "fromString",
      Napi::Function::New(env, DtypeWrapper::FromString, "fromString", &data));

  auto issubdtype_fn =
      Napi::Function::New(env, Issubdtype, "issubdtype", &data);
  dtype_ns.Set("issubdtype", issubdtype_fn);
  exports.Set("issubdtype", issubdtype_fn);

  Napi::Array keys = Napi::Array::New(env, entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    keys[i] = Napi::String::New(env, entries[i].first);
  }
  dtype_ns.Set("keys", keys);

  const std::vector<std::pair<std::string, mlx::core::Dtype::Category>>
      categories = {
          {"complexfloating", mlx::core::complexfloating},
          {"floating", mlx::core::floating},
          {"inexact", mlx::core::inexact},
          {"signedinteger", mlx::core::signedinteger},
          {"unsignedinteger", mlx::core::unsignedinteger},
          {"integer", mlx::core::integer},
          {"number", mlx::core::number},
          {"generic", mlx::core::generic},
      };

  for (const auto& entry : categories) {
    auto category_obj = DtypeCategoryWrapper::Create(env, entry.second, data);
    exports.Set(entry.first, category_obj);
    dtype_ns.Set(entry.first, category_obj);
  }

  exports.Set("Dtype", data.dtype_constructor.Value());
  exports.Set("DtypeCategory", data.dtype_category_constructor.Value());
  exports.Set("dtype", dtype_ns);
}

} // namespace mlx::node
