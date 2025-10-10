{
  "includes": ["scripts/mlx_sources.gypi"],
  "targets": [
    
    {
      "target_name": "mlx_core",
      "type": "static_library",
      "sources": [
        "<@(mlx_sources)",
        "generated/jit/utils.cpp",
        "generated/jit/unary_ops.cpp",
        "generated/jit/binary_ops.cpp",
        "generated/jit/ternary_ops.cpp",
        "generated/jit/reduce_utils.cpp",
        "generated/jit/gather.cpp",
        "generated/jit/gather_front.cpp",
        "generated/jit/gather_axis.cpp",
        "generated/jit/scatter_axis.cpp",
        "generated/jit/hadamard.cpp",
        "generated/jit/arange.cpp",
        "generated/jit/copy.cpp",
        "generated/jit/unary.cpp",
        "generated/jit/binary.cpp",
        "generated/jit/binary_two.cpp",
        "generated/jit/fft.cpp",
        "generated/jit/logsumexp.cpp",
        "generated/jit/ternary.cpp",
        "generated/jit/softmax.cpp",
        "generated/jit/scan.cpp",
        "generated/jit/sort.cpp",
        "generated/jit/reduce.cpp",
        "generated/jit/gemm.cpp",
        "generated/jit/steel_gemm_fused.cpp",
        "generated/jit/steel_gemm_masked.cpp",
        "generated/jit/steel_gemm_gather.cpp",
        "generated/jit/steel_gemm_splitk.cpp",
        "generated/jit/steel_gemm_segmented.cpp",
        "generated/jit/conv.cpp",
        "generated/jit/steel_conv.cpp",
        "generated/jit/steel_conv_general.cpp",
        "generated/jit/quantized_utils.cpp",
        "generated/jit/quantized.cpp",
        "generated/jit/fp4_quantized.cpp",
        "generated/jit/gemv_masked.cpp"
      ],
      "include_dirs": [
        "vendor/metal-cpp",
        "vendor",
        "../",
        "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
      ],
      "cflags": [ "-fexceptions" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-std=gnu++17", "-fexceptions", "-frtti" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [
        "FMT_HEADER_ONLY",
        "MLX_VERSION=\"0.29.0\""
      ],
      "conditions": [
        ['OS=="mac"', {
          "defines": [
            "MLX_USE_ACCELERATE",
            "METAL_PATH=\"<(module_root_dir)/vendor/mlx/backend/metal/kernels/mlx.metallib\""
          ],
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
        }],
        ['OS=="linux"', {
          "defines": [
            "MLX_USE_BLAS"
          ],
          "cflags": [
            "-ULAPACK_GLOBAL",
            "-ULAPACK_NAME"
          ],
          "include_dirs": [
            "/usr/include"
          ],
          "libraries": [
            "-lblas",
            "-llapack"
          ]
        }]
      ],
      "actions": [
        {
          "action_name": "generate_metal_jit",
          "inputs": [ "scripts/gen_mlx_metal_jit.sh", "../mlx/backend/metal/make_compiled_preamble.sh" ],
          "outputs": [
            "<(module_root_dir)/generated/jit/utils.cpp",
            "<(module_root_dir)/generated/jit/unary_ops.cpp",
            "<(module_root_dir)/generated/jit/binary_ops.cpp",
            "<(module_root_dir)/generated/jit/ternary_ops.cpp",
            "<(module_root_dir)/generated/jit/reduce_utils.cpp",
            "<(module_root_dir)/generated/jit/indexing.cpp",
            "<(module_root_dir)/generated/jit/gather.cpp",
            "<(module_root_dir)/generated/jit/gather_front.cpp",
            "<(module_root_dir)/generated/jit/gather_axis.cpp",
            "<(module_root_dir)/generated/jit/scatter_axis.cpp",
            "<(module_root_dir)/generated/jit/hadamard.cpp",
            "<(module_root_dir)/generated/jit/arange.cpp",
            "<(module_root_dir)/generated/jit/copy.cpp",
            "<(module_root_dir)/generated/jit/unary.cpp",
            "<(module_root_dir)/generated/jit/binary.cpp",
            "<(module_root_dir)/generated/jit/binary_two.cpp",
            "<(module_root_dir)/generated/jit/fft.cpp",
            "<(module_root_dir)/generated/jit/logsumexp.cpp",
            "<(module_root_dir)/generated/jit/ternary.cpp",
            "<(module_root_dir)/generated/jit/softmax.cpp",
            "<(module_root_dir)/generated/jit/scan.cpp",
            "<(module_root_dir)/generated/jit/sort.cpp",
            "<(module_root_dir)/generated/jit/reduce.cpp",
            "<(module_root_dir)/generated/jit/gemm.cpp",
            "<(module_root_dir)/generated/jit/steel_gemm_fused.cpp",
            "<(module_root_dir)/generated/jit/steel_gemm_masked.cpp",
            "<(module_root_dir)/generated/jit/steel_gemm_gather.cpp",
            "<(module_root_dir)/generated/jit/steel_gemm_splitk.cpp",
            "<(module_root_dir)/generated/jit/steel_gemm_segmented.cpp",
            "<(module_root_dir)/generated/jit/conv.cpp",
            "<(module_root_dir)/generated/jit/steel_conv.cpp",
            "<(module_root_dir)/generated/jit/steel_conv_general.cpp",
            "<(module_root_dir)/generated/jit/quantized_utils.cpp",
            "<(module_root_dir)/generated/jit/quantized.cpp",
            "<(module_root_dir)/generated/jit/fp4_quantized.cpp",
            "<(module_root_dir)/generated/jit/gemv_masked.cpp"
          ],
          "action": [
            "bash", "-lc",
            "./scripts/gen_mlx_metal_jit.sh '<(module_root_dir)/..' '<(module_root_dir)/generated/jit' \"$CC\""
          ]
        }
      ]
    },
    {
      "target_name": "mlx",
      "sources": [
        "src/native/addon_data.cc",
        "src/native/array.cc",
        "src/native/dtype.cc",
        "src/native/stream.cc",
        "src/native/runtime.mm"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<!@(node -p \"require('nan')\")",
        "vendor",
        "../"
      ],
      "dependencies": [
        "<!@(node -p \"require('node-addon-api').gyp\")",
        "mlx_core"
      ],
      "cflags": [ "-fexceptions" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-std=gnu++17", "-fexceptions", "-frtti" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [
        "NAPI_CPP_EXCEPTIONS",
        "MLX_VERSION=\"0.29.0\""
      ],
      "conditions": [
        ['OS=="mac"', {
          "defines": [
            "MLX_USE_ACCELERATE",
            "METAL_PATH=\"<(module_root_dir)/vendor/mlx/backend/metal/kernels/mlx.metallib\""
          ],
          "include_dirs": [
            "vendor/metal-cpp",
            "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
          ],
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "GCC_ENABLE_CPP_RTTI": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "CLANG_CXX_LANGUAGE_STANDARD": "gnu++17",
            "OTHER_LDFLAGS": [
              "-framework", "Metal",
              "-framework", "Accelerate",
              "-framework", "Foundation",
              "-framework", "QuartzCore"
            ]
          }
        }],
        ['OS=="linux"', {
          "defines": [
            "MLX_USE_BLAS"
          ]
        }]
      ]
    }
  ]
}