cmake_minimum_required(VERSION 3.22)
project(llama_android C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp build options
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE       OFF CACHE BOOL "" FORCE)

# Architecture-specific optimizations
if(ANDROID_ABI STREQUAL "arm64-v8a")
    # Modern ARMv8.4+ / ARMv9 flagships (Snapdragon 8 Gen 1 Kryo Cortex-X2 + A710)
    # Enable dotprod and i8mm for accelerated matrix math kernels (yields ~7.70 sustained tok/s on Qwen3.5-2B Q4_K_M)
    add_compile_options(-O3 -march=armv8.4-a+dotprod+i8mm)
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    # 32-bit ARM legacy support with NEON vector instructions
    add_compile_options(-O3 -mfpu=neon)
elseif(ANDROID_ABI STREQUAL "x86_64")
    # Android Studio Emulator / x86_64 development environments
    add_compile_options(-O3)
endif()

# Vulkan GPU compute backend (OFF by default: Qualcomm Adreno vendor drivers on Samsung One UI
# exhibit heap allocation crashes in ggml_backend_buft_alloc_buffer with large tensor memory pools)
option(GGML_VULKAN "Enable Vulkan backend for ggml" OFF)

if(GGML_VULKAN)
    find_library(Vulkan_LIBRARY vulkan)
    set(VULKAN_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-headers/include")
    if(EXISTS "${VULKAN_HEADERS_DIR}")
        set(Vulkan_INCLUDE_DIRS "${VULKAN_HEADERS_DIR};${ANDROID_SYSROOT}/usr/include" CACHE STRING "" FORCE)
        set(Vulkan_INCLUDE_DIR  "${VULKAN_HEADERS_DIR}" CACHE STRING "" FORCE)
    endif()

    find_program(Vulkan_GLSLC_EXECUTABLE glslc
        HINTS
            "${ANDROID_NDK}/shader-tools/linux-x86_64"
            "$ENV{ANDROID_NDK_ROOT}/shader-tools/linux-x86_64"
            "$ENV{ANDROID_NDK_HOME}/shader-tools/linux-x86_64"
    )
    if(Vulkan_GLSLC_EXECUTABLE)
        set(Vulkan_GLSLC_FOUND TRUE CACHE BOOL "" FORCE)
    endif()

    if(Vulkan_LIBRARY AND NOT TARGET Vulkan::Vulkan)
        add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${Vulkan_LIBRARY}"
        )
        set(Vulkan_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
endif()

add_subdirectory(llama.cpp)

# JNI bridge shared library
add_library(llama_jni SHARED
    llama_jni.cpp
)

target_include_directories(llama_jni PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
)

target_link_libraries(llama_jni
    llama
    ggml
    android
    log
    atomic
)

if(GGML_VULKAN AND TARGET Vulkan::Vulkan)
    target_link_libraries(llama_jni Vulkan::Vulkan)
endif()
