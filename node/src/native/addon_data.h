#pragma once

#include <napi.h>

namespace mlx::node {

struct AddonData {
  Napi::FunctionReference array_constructor;
  Napi::FunctionReference dtype_constructor;
  Napi::FunctionReference dtype_category_constructor;
  Napi::FunctionReference stream_constructor;
  Napi::FunctionReference stream_context_constructor;

  ~AddonData();
};

AddonData& GetAddonData(Napi::Env env);

} // namespace mlx::node
