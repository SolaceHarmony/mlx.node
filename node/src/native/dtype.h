#pragma once

#include <napi.h>

#include "addon_data.h"
#include "mlx/dtype.h"

namespace mlx::node {

void InitDtype(Napi::Env env, Napi::Object exports, AddonData& data);

// Returns true and fills `out` if `value` is a mlx.core.Dtype object.
bool TryUnwrapDtype(const Napi::Value& value, AddonData& data, mlx::core::Dtype& out);

} // namespace mlx::node
