{
  "targets": [
    {
      "target_name": "mlx_array",
      "sources": [
        "src/native/array.cc",
        "vendor/mlx/version.cpp",
        "vendor/mlx/allocator.cpp",
        "vendor/mlx/array.cpp",
        "vendor/mlx/dtype.cpp",
        "vendor/mlx/dtype_utils.cpp",
        "vendor/mlx/backend/no_gpu/allocator.cpp",
        "src/native/dtype.cc",
        "src/native/shim_transforms.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../",
        "vendor"
      ],
      "dependencies": [
        "<!@(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags": [ "-fexceptions" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-std=c++17", "-fexceptions", "-frtti" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [ "NAPI_CPP_EXCEPTIONS", "MLX_VERSION=\"0.29.0\"" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++"
      }
    }
  ]
}
