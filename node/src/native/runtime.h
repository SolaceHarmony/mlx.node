#pragma once

#include <mutex>
#include <utility>

#include "mlx/device.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/backend/metal/device.h"

namespace mlx::node {

class Runtime {
 public:
  static Runtime& Instance();

  void EnsureMetalInit();

  template <typename F>
  auto WithMetalPool(F&& f) -> decltype(f()) {
    auto pool = mlx::core::metal::new_scoped_memory_pool();
    return std::forward<F>(f)();
  }

 private:
  Runtime() = default;
  std::once_flag metal_once_{};
};

} // namespace mlx::node

