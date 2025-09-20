#pragma once

#include <memory>

#include "mlx/array.h"

namespace mlx::bridge {

struct ArrayHolder {
  using Array = mlx::core::array;

  ArrayHolder() = default;
  explicit ArrayHolder(Array value) : array(std::move(value)) {}

  ArrayHolder(const ArrayHolder&) = default;
  ArrayHolder(ArrayHolder&&) noexcept = default;
  ArrayHolder& operator=(const ArrayHolder&) = default;
  ArrayHolder& operator=(ArrayHolder&&) noexcept = default;

  Array array;
};

using ArrayHolderPtr = std::shared_ptr<ArrayHolder>;

} // namespace mlx::bridge
