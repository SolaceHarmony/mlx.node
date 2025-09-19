{
  "includes": ["scripts/mlx_sources.gypi"],
  "targets": [
    {
      "target_name": "mlx_core",
      "type": "static_library",
      "sources": ["<@(mlx_sources)"],
      "include_dirs": [
        "../",
        "vendor",
        "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
      ],
      "cflags": [ "-fexceptions" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-std=gnu++17", "-fexceptions", "-frtti" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": ["MLX_VERSION=\"0.29.0\"", "MLX_USE_ACCELERATE"],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "GCC_ENABLE_CPP_RTTI": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "CLANG_CXX_LANGUAGE_STANDARD": "gnu++17",
        "HEADER_SEARCH_PATHS": [
          "$(HEADER_SEARCH_PATHS)",
          "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
        ]
      }
    },
    {
      "target_name": "mlx_array",
      "sources": [
        "src/native/array.cc",
        "src/native/dtype.cc",
        "src/native/shim_transforms.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../",
        "vendor",
        "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
      ],
      "dependencies": [
        "<!@(node -p \"require('node-addon-api').gyp\")",
        "mlx_core"
      ],
      "cflags": [ "-fexceptions" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-std=gnu++17", "-fexceptions", "-frtti" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [ "NAPI_CPP_EXCEPTIONS", "MLX_VERSION=\"0.29.0\"", "MLX_USE_ACCELERATE" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "GCC_ENABLE_CPP_RTTI": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "CLANG_CXX_LANGUAGE_STANDARD": "gnu++17",
        "HEADER_SEARCH_PATHS": [
          "$(HEADER_SEARCH_PATHS)",
          "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
        ],
        "OTHER_LDFLAGS": [
          "-framework", "Metal",
          "-framework", "Accelerate",
          "-framework", "Foundation",
          "-framework", "QuartzCore"
        ]
      }
    }
  ]
}
