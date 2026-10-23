cmake_minimum_required(VERSION 3.22)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)

# llama.cpp build options
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# ARM NEON
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_NEON ON CACHE BOOL "" FORCE)
    add_compile_options(-march=armv8.4-a+dotprod+i8mm -O3)
endif()

# Vulkan
set(VULKAN_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-headers/include")
set(NDK_VULKAN_DIR "${ANDROID_SYSROOT}/usr/include")

set(Vulkan_INCLUDE_DIRS "${VULKAN_HEADERS_DIR};${NDK_VULKAN_DIR}" CACHE STRING "" FORCE)
set(Vulkan_INCLUDE_DIR  "${VULKAN_HEADERS_DIR}"                    CACHE STRING "" FORCE)

set(Vulkan_LIBRARY "/opt/android-sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/28/libvulkan.so" CACHE STRING "" FORCE)
set(Vulkan_FOUND TRUE CACHE BOOL "" FORCE)

set(Vulkan_GLSLC_EXECUTABLE
    "/opt/android-sdk/ndk/27.2.12479018/shader-tools/linux-x86_64/glslc"
    CACHE STRING "" FORCE)
set(Vulkan_GLSLC_FOUND TRUE CACHE BOOL "" FORCE)

if (NOT TARGET Vulkan::Vulkan)
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${Vulkan_LIBRARY}"
    )
endif()

set(GGML_VULKAN OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

# JNI bridge
add_library(llama_jni SHARED llama_jni.cpp)

target_include_directories(llama_jni PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
)

target_link_libraries(llama_jni
    llama
    ggml
    android
    log
    Vulkan::Vulkan
)
