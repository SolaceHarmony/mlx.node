#include "stream.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "addon_data.h"

#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::node {
namespace {

using mlx::core::Device;
using mlx::core::Stream;
using mlx::core::StreamContext;
using mlx::core::StreamOrDevice;

class StreamWrapper : public Napi::ObjectWrap<StreamWrapper> {
 public:
  static void Define(Napi::Env env, Napi::Object exports, AddonData& data) {
    if (!data.stream_constructor.IsEmpty()) {
      return;
    }

    Napi::Function func = DefineClass(
        env,
        "Stream",
        {
            InstanceAccessor("index", &StreamWrapper::Index, nullptr),
            InstanceAccessor("device", &StreamWrapper::DeviceInfo, nullptr),
            InstanceMethod("equals", &StreamWrapper::Equals),
            InstanceMethod("toString", &StreamWrapper::ToString),
        },
        &data);

    data.stream_constructor = Napi::Persistent(func);
    exports.Set("Stream", func);
  }

  static Napi::Object Create(
      Napi::Env env,
      const Stream& stream,
      AddonData& data) {
    auto copy = new Stream(stream);
    auto external = Napi::External<Stream>::New(
        env,
        copy,
        [](Napi::Env /*env*/, Stream* ptr) { delete ptr; });
    return data.stream_constructor.New({external});
  }

  static bool IsInstance(const Napi::Value& value, AddonData& data) {
    if (!value.IsObject()) {
      return false;
    }
    auto ctor = data.stream_constructor.Value();
    return !ctor.IsEmpty() && value.As<Napi::Object>().InstanceOf(ctor);
  }

  static Stream Get(Napi::Env env, const Napi::Value& value, AddonData& data) {
    if (!IsInstance(value, data)) {
      Napi::TypeError::New(env, "Expected an mlx.core.Stream instance")
          .ThrowAsJavaScriptException();
      return Stream(0, Device(Device::cpu));
    }
    auto wrapper =
        Napi::ObjectWrap<StreamWrapper>::Unwrap(value.As<Napi::Object>());
    return wrapper->stream_;
  }

  explicit StreamWrapper(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<StreamWrapper>(info),
        addon_data_(static_cast<AddonData*>(info.Data())),
        stream_(0, Device(Device::cpu)) {
    auto env = info.Env();
    if (info.Length() != 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "mlx.core.Stream cannot be constructed directly")
          .ThrowAsJavaScriptException();
      return;
    }
    auto external = info[0].As<Napi::External<Stream>>();
    stream_ = *external.Data();
  }

 private:
  static Napi::Object WrapDevice(Napi::Env env, const Device& device) {
    auto obj = Napi::Object::New(env);
    std::string type = device.type == Device::DeviceType::cpu ? "cpu" : "gpu";
    obj.Set("type", type);
    obj.Set("index", Napi::Number::New(env, device.index));
    return obj;
  }

  Napi::Value Index(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), stream_.index);
  }

  Napi::Value DeviceInfo(const Napi::CallbackInfo& info) {
    return WrapDevice(info.Env(), stream_.device);
  }

  Napi::Value Equals(const Napi::CallbackInfo& info) {
    if (addon_data_ == nullptr || info.Length() < 1) {
      return Napi::Boolean::New(info.Env(), false);
    }
    if (!IsInstance(info[0], *addon_data_)) {
      return Napi::Boolean::New(info.Env(), false);
    }
    auto other = Get(info.Env(), info[0], *addon_data_);
    if (info.Env().IsExceptionPending()) {
      return Napi::Boolean::New(info.Env(), false);
    }
    return Napi::Boolean::New(info.Env(), other == stream_);
  }

  Napi::Value ToString(const Napi::CallbackInfo& info) {
    std::ostringstream os;
    os << stream_;
    return Napi::String::New(info.Env(), os.str());
  }

  AddonData* addon_data_;
  Stream stream_;
};

class StreamContextWrapper : public Napi::ObjectWrap<StreamContextWrapper> {
 public:
  static void Define(Napi::Env env, Napi::Object exports, AddonData& data) {
    if (!data.stream_context_constructor.IsEmpty()) {
      return;
    }

    Napi::Function func = DefineClass(
        env,
        "StreamContext",
        {
            InstanceMethod("enter", &StreamContextWrapper::Enter),
            InstanceMethod("exit", &StreamContextWrapper::Exit),
        });

    data.stream_context_constructor = Napi::Persistent(func);
    exports.Set("StreamContext", func);
  }

  static Napi::Object Create(
      Napi::Env env,
      StreamOrDevice stream_or_device,
      AddonData& data) {
    auto payload = new StreamOrDevice(std::move(stream_or_device));
    auto external = Napi::External<StreamOrDevice>::New(
        env,
        payload,
        [](Napi::Env /*env*/, StreamOrDevice* ptr) { delete ptr; });
    return data.stream_context_constructor.New({external});
  }

  explicit StreamContextWrapper(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<StreamContextWrapper>(info) {
    auto env = info.Env();
    if (info.Length() != 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "StreamContext cannot be constructed directly")
          .ThrowAsJavaScriptException();
      return;
    }
    auto external = info[0].As<Napi::External<StreamOrDevice>>();
    stream_or_device_ = *external.Data();
  }

  Napi::Value Enter(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    if (context_) {
      Napi::Error::New(env, "StreamContext already entered")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    try {
      context_ = std::make_unique<StreamContext>(stream_or_device_);
    } catch (const std::exception& error) {
      Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
  }

  Napi::Value Exit(const Napi::CallbackInfo& info) {
    context_.reset();
    return info.Env().Undefined();
  }

 private:
  StreamOrDevice stream_or_device_;
  std::unique_ptr<StreamContext> context_;
};

Device ParseDevice(Napi::Env env, const Napi::Value& value) {
  if (value.IsUndefined() || value.IsNull()) {
    return mlx::core::default_device();
  }

  std::string type;
  int index = 0;

  if (value.IsString()) {
    type = value.As<Napi::String>();
  } else if (value.IsObject()) {
    auto obj = value.As<Napi::Object>();
    if (obj.Has("type")) {
      type = obj.Get("type").As<Napi::String>();
    }
    if (obj.Has("index")) {
      index = obj.Get("index").As<Napi::Number>().Int32Value();
    }
  }

  Device::DeviceType device_type;
  if (type == "gpu") {
    device_type = Device::gpu;
  } else if (type == "cpu" || type.empty()) {
    device_type = Device::cpu;
  } else {
    Napi::TypeError::New(env, "Unsupported device type: " + type)
        .ThrowAsJavaScriptException();
    return Device(Device::cpu);
  }

  return Device(device_type, index);
}

Stream ParseStream(
    Napi::Env env,
    const Napi::Value& value,
    AddonData& data) {
  if (!StreamWrapper::IsInstance(value, data)) {
    Napi::TypeError::New(env, "Expected an mlx.core.Stream instance")
        .ThrowAsJavaScriptException();
    return Stream(0, Device(Device::cpu));
  }
  return StreamWrapper::Get(env, value, data);
}

StreamOrDevice ParseStreamOrDevice(
    Napi::Env env,
    const Napi::Value& value,
    AddonData& data) {
  if (StreamWrapper::IsInstance(value, data)) {
    return ParseStream(env, value, data);
  }
  return ParseDevice(env, value);
}

AddonData* RequireAddonData(const Napi::CallbackInfo& info, const char* label) {
  auto* addon_data = static_cast<AddonData*>(info.Data());
  if (addon_data == nullptr) {
    Napi::Error::New(info.Env(), std::string("AddonData missing for ") + label)
        .ThrowAsJavaScriptException();
  }
  return addon_data;
}

Napi::Value DefaultStream(const Napi::CallbackInfo& info) {
  auto* addon_data = RequireAddonData(info, "default_stream");
  if (addon_data == nullptr) {
    return info.Env().Null();
  }
  auto env = info.Env();
  Device device =
      info.Length() > 0 ? ParseDevice(env, info[0]) : mlx::core::default_device();
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto stream = mlx::core::default_stream(device);
  return StreamWrapper::Create(env, stream, *addon_data);
}

Napi::Value NewStream(const Napi::CallbackInfo& info) {
  auto* addon_data = RequireAddonData(info, "new_stream");
  if (addon_data == nullptr) {
    return info.Env().Null();
  }
  auto env = info.Env();
  Device device =
      info.Length() > 0 ? ParseDevice(env, info[0]) : mlx::core::default_device();
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto stream = mlx::core::new_stream(device);
  return StreamWrapper::Create(env, stream, *addon_data);
}

Napi::Value SetDefaultStream(const Napi::CallbackInfo& info) {
  auto* addon_data = RequireAddonData(info, "set_default_stream");
  auto env = info.Env();
  if (addon_data == nullptr) {
    return env.Undefined();
  }
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "set_default_stream expects a Stream argument")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  auto stream = ParseStream(env, info[0], *addon_data);
  if (env.IsExceptionPending()) {
    return env.Undefined();
  }
  mlx::core::set_default_stream(stream);
  return env.Undefined();
}

Napi::Value Synchronize(const Napi::CallbackInfo& info) {
  auto* addon_data = RequireAddonData(info, "synchronize");
  auto env = info.Env();
  if (addon_data == nullptr) {
    return env.Undefined();
  }
  if (info.Length() == 0 || info[0].IsUndefined() || info[0].IsNull()) {
    mlx::core::synchronize();
    return env.Undefined();
  }
  auto stream = ParseStream(env, info[0], *addon_data);
  if (env.IsExceptionPending()) {
    return env.Undefined();
  }
  mlx::core::synchronize(stream);
  return env.Undefined();
}

Napi::Value StreamFactory(const Napi::CallbackInfo& info) {
  auto* addon_data = RequireAddonData(info, "stream");
  auto env = info.Env();
  if (addon_data == nullptr) {
    return env.Null();
  }
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "stream expects a stream or device argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto stream_or_device = ParseStreamOrDevice(env, info[0], *addon_data);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  return StreamContextWrapper::Create(env, stream_or_device, *addon_data);
}

} // namespace

void InitStreamBindings(Napi::Env env, Napi::Object exports, AddonData& data) {
  StreamWrapper::Define(env, exports, data);
  StreamContextWrapper::Define(env, exports, data);
  exports.Set(
      "default_stream",
      Napi::Function::New(env, DefaultStream, "default_stream", &data));
  exports.Set(
      "new_stream",
      Napi::Function::New(env, NewStream, "new_stream", &data));
  exports.Set(
      "set_default_stream",
      Napi::Function::New(env, SetDefaultStream, "set_default_stream", &data));
  exports.Set(
      "synchronize",
      Napi::Function::New(env, Synchronize, "synchronize", &data));
  exports.Set(
      "stream",
      Napi::Function::New(env, StreamFactory, "stream", &data));
}

} // namespace mlx::node
