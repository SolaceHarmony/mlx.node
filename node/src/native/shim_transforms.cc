#include <functional>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::detail {

int InTracing::grad_counter = 0;

std::vector<std::pair<char, char>>& InTracing::trace_stack() {
  thread_local std::vector<std::pair<char, char>> stack;
  return stack;
}

int RetainGraph::tracing_counter = 0;

} // namespace mlx::core::detail
