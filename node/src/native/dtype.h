#pragma once

#include <napi.h>

#include "addon_data.h"

namespace mlx::node {

void InitDtype(Napi::Env env, Napi::Object exports, AddonData& data);

} // namespace mlx::node
