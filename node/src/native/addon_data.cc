#include "addon_data.h"

namespace mlx::node {

AddonData::~AddonData() {
  array_constructor.Reset();
  dtype_constructor.Reset();
  dtype_category_constructor.Reset();
  stream_constructor.Reset();
  stream_context_constructor.Reset();
}

AddonData& GetAddonData(Napi::Env env) {
  auto* data = env.GetInstanceData<AddonData>();
  if (data == nullptr) {
    data = new AddonData();
    env.SetInstanceData<AddonData>(data);
    env.AddCleanupHook(
        [](void* ptr) { delete static_cast<AddonData*>(ptr); }, data);
  }
  return *data;
}

} // namespace mlx::node
