#include "runtime.h"

namespace mlx::node {

Runtime& Runtime::Instance() {
  static Runtime rt;
  return rt;
}

void Runtime::EnsureMetalInit() {
  std::call_once(metal_once_, [] {
    if (!mlx::core::gpu::is_available()) {
      throw std::runtime_error("MLX GPU backend is unavailable");
    }
    auto pool = mlx::core::metal::new_scoped_memory_pool();
    // Initialize GPU device and set GPU as default device so downstream ops
    // run on Metal without having to probe device_info() here.
    (void)mlx::core::metal::device(mlx::core::Device::gpu);
    mlx::core::set_default_device(mlx::core::Device(mlx::core::Device::gpu));
  });
}

} // namespace mlx::node

