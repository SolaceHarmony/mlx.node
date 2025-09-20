#pragma once

#include <napi.h>

#include "addon_data.h"

namespace mlx::node {

void InitStreamBindings(Napi::Env env, Napi::Object exports, AddonData& data);

}
