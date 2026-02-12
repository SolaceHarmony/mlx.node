#include <cmath>
#include <memory>
#include <vector>

#include <napi.h>

#include "addon_data.h"
#include "mlx/array.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx_bridge.h"
#include "runtime.h"
#include "stream.h"

namespace {

// Forward declarations
Napi::Object WrapArray(Napi::Env env, std::shared_ptr<mlx::core::array> tensor);
mlx::core::array* UnwrapArray(Napi::Env env, const Napi::Value& value);

// Helper to unwrap array from JS
mlx::core::array* UnwrapArray(Napi::Env env, const Napi::Value& value) {
  if (!value.IsObject()) {
    Napi::TypeError::New(env, "Expected Array object")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
  auto obj = value.As<Napi::Object>();
  auto* wrapper = Napi::ObjectWrap<mlx::node::ArrayWrapper>::Unwrap(obj);
  if (!wrapper) {
    Napi::TypeError::New(env, "Expected Array object")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
  return &wrapper->tensor();
}

// Helper to wrap array to JS
Napi::Object WrapArray(Napi::Env env, std::shared_ptr<mlx::core::array> tensor) {
  auto& data = mlx::node::GetAddonData(env);
  auto constructor = data.array_constructor.Value();
  auto instance = constructor.New({});
  auto* wrapper = Napi::ObjectWrap<mlx::node::ArrayWrapper>::Unwrap(instance);
  wrapper->set_tensor(std::move(*tensor));
  return instance;
}

/**
 * Sparse initializer
 * 
 * Args:
 *   - a: Input array (must be 2D)
 *   - sparsity: Fraction of elements in each column to set to zero
 *   - mean: Mean of normal distribution (default: 0.0)
 *   - std: Standard deviation (default: 1.0)
 *   - stream (optional)
 * 
 * Returns: Array with same shape as input, sparsified normal distribution
 */
Napi::Value Sparse(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto* addon = static_cast<mlx::node::AddonData*>(info.Data());

  // Ensure Metal is initialized
  try {
    mlx::node::Runtime::Instance().EnsureMetalInit();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }

  // Validate argument count (at least 2: array and sparsity)
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "sparse requires at least 2 arguments: array and sparsity")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse input array
  auto* a_ptr = UnwrapArray(env, info[0]);
  if (!a_ptr) return env.Null();
  const auto& a = *a_ptr;

  // Validate 2D array
  if (a.ndim() != 2) {
    Napi::TypeError::New(env, "Only tensors with 2 dimensions are supported")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Parse sparsity
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "sparsity must be a number")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  float sparsity = info[1].As<Napi::Number>().FloatValue();

  // Parse optional parameters
  float mean = 0.0f;
  float std = 1.0f;

  int next_arg = 2;
  
  // Check for mean parameter
  if (info.Length() > next_arg && info[next_arg].IsNumber()) {
    mean = info[next_arg].As<Napi::Number>().FloatValue();
    next_arg++;
  }

  // Check for std parameter
  if (info.Length() > next_arg && info[next_arg].IsNumber()) {
    std = info[next_arg].As<Napi::Number>().FloatValue();
    next_arg++;
  }

  // Parse stream (last argument if present and not a number)
  auto stream = mlx::core::default_stream(mlx::core::default_device());
  if (info.Length() > next_arg) {
    stream = mlx::node::ParseStreamOrDevice(env, info[next_arg], *addon);
    if (env.IsExceptionPending()) return env.Null();
  }

  try {
    // Get array dimensions
    int rows = a.shape(0);
    int cols = a.shape(1);
    
    // Calculate number of zeros per column
    int num_zeros = static_cast<int>(std::ceil(sparsity * static_cast<float>(cols)));

    // Generate random order for each element (argsort of uniform random values)
    auto order = mlx::core::argsort(
        mlx::core::random::uniform(a.shape(), mlx::core::float32, std::nullopt, stream),
        1,  // axis=1 (sort along columns for each row)
        stream
    );

    // Generate normal distribution with specified mean and std
    auto result = mlx::core::random::normal(
        a.shape(),
        a.dtype(),
        mlx::core::array(mean),
        mlx::core::array(std),
        std::nullopt,
        stream
    );

    // Create a mask: 1 for elements to keep, 0 for elements to zero out
    // Start with all ones
    auto mask = mlx::core::ones(a.shape(), mlx::core::float32, stream);

    // We need to set mask to 0 for the first num_zeros elements in each row
    // according to the sorted order
    // Create a range [0, 1, 2, ..., cols-1] and reshape to (1, cols)
    auto col_range = mlx::core::arange(0, cols, 1, mlx::core::int32, stream);
    col_range = mlx::core::reshape(col_range, {1, cols}, stream);
    
    // Broadcast to (rows, cols)
    col_range = mlx::core::broadcast_to(col_range, {rows, cols}, stream);
    
    // Get the sorted indices for the first num_zeros elements
    // Slice order to get order[:, :num_zeros]
    std::vector<int> starts = {0, 0};
    std::vector<int> stops = {rows, num_zeros};
    std::vector<int> strides = {1, 1};
    auto order_slice = mlx::core::slice(order, starts, stops, strides, stream);
    
    // Create mask: elements where col_range appears in order_slice should be 0
    // This is complex with just MLX ops. Let's use a different approach.
    
    // Alternative: Since we need to set specific (row, col) pairs to zero,
    // and MLX doesn't have easy advanced indexing in C++, we'll work with
    // the evaluated arrays in memory
    
    // Evaluate the arrays we need
    order = mlx::core::eval(order);
    result = mlx::core::eval(result);
    
    // Get pointers to the data
    // Note: We need to modify result, so we'll create a new array
    std::vector<float> result_vec;
    result_vec.reserve(rows * cols);
    
    // Copy result data
    auto result_ptr = result.data<float>();
    for (int i = 0; i < rows * cols; i++) {
      result_vec.push_back(result_ptr[i]);
    }
    
    // Get order data
    auto order_ptr = order.data<int>();
    
    // Zero out the selected elements
    for (int row = 0; row < rows; row++) {
      for (int j = 0; j < num_zeros; j++) {
        int col = order_ptr[row * cols + j];
        result_vec[row * cols + col] = 0.0f;
      }
    }
    
    // Create new array from modified data
    auto final_result = mlx::core::array(
        result_vec.data(),
        {rows, cols},
        a.dtype()
    );

    return WrapArray(env, std::make_shared<mlx::core::array>(std::move(final_result)));
  } catch (const std::exception& e) {
    Napi::Error::New(env, std::string("sparse failed: ") + e.what())
        .ThrowAsJavaScriptException();
    return env.Null();
  }
}

}  // namespace

// Module initialization
Napi::Object InitNN(Napi::Env env, Napi::Object exports) {
  auto& data = mlx::node::GetAddonData(env);
  
  exports.Set("sparse", Napi::Function::New(env, Sparse, "sparse", &data));
  
  return exports;
}

NODE_API_MODULE(nn, InitNN)
