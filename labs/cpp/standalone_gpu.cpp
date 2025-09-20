// Minimal standalone C++ sanity for MLX (Metal GPU streams)
// Build via labs/cpp/build_standalone.sh

#include <iostream>
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  try {
    // Force GPU default device to make intent explicit
    mx::set_default_device(mx::Device(mx::Device::gpu));

    auto s2 = mx::new_stream(mx::Device::gpu);
    auto s3 = mx::new_stream(mx::Device::gpu);

    auto a = mx::arange(1.f, 10.f, 1.f, mx::float32, s2);
    auto b = mx::arange(1.f, 10.f, 1.f, mx::float32, s3);
    auto x = mx::add(a, a, s2);
    auto y = mx::add(b, b, s3);
    auto z = mx::multiply(x, y);
    mx::eval(z);
    std::cout << z << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 2;
  }
}

