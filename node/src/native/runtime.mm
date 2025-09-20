#include "runtime.h"
#include <objc/objc.h>
#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <Metal/Metal.h>
#endif
#endif

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

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
    struct Pool {
      void* p{nullptr};
      Pool() { p = objc_autoreleasePoolPush(); }
      ~Pool() { if (p) objc_autoreleasePoolPop(p); }
    } pool_guard;
    // Initialize GPU device and set GPU as default device so downstream ops
    // run on Metal without having to probe device_info() here.
#if defined(__APPLE__) && TARGET_OS_OSX
    {
      id dev = MTLCreateSystemDefaultDevice();
      if (!dev) {
        throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
      }
      // dev is autoreleased by the pool
    }
#endif
    (void)mlx::core::metal::device(mlx::core::Device::gpu);
    mlx::core::set_default_device(mlx::core::Device(mlx::core::Device::gpu));
  });
}

} // namespace mlx::node
