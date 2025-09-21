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
    // Attach instance data with the default finalizer provided by node-addon-api,
    // which will delete the pointer exactly once at env teardown.
    env.SetInstanceData<AddonData>(data);
  }
  return *data;
}

} // namespace mlx::node
