cmake_minimum_required(VERSION 3.18)
project(tejas LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(SKIP_CUDA "Build without CUDA support" OFF)

include_directories(include)

if(NOT SKIP_CUDA)
    # Enable CUDA only if the flag is OFF (Local machine)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    # Compile the library with the GPU backend
    add_library(tejas_lib src/tensor.cpp src/cuda_backend.cu src/nn.cpp)
    target_link_libraries(tejas_lib PUBLIC CUDA::cudart)
    
    #Define USE_CUDA macro for our C++ files
    target_compile_definitions(tejas_lib PUBLIC USE_CUDA)
else()
    # CPU-Only Mode (GitHub Actions)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    
    # Compile the library WITHOUT the GPU backend
    add_library(tejas_lib src/tensor.cpp src/nn.cpp)
endif()

# ==========================================
# Universal Executables (Always Built)
# ==========================================
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor tejas_lib)

add_executable(xor examples/xor.cpp)
target_link_libraries(xor tejas_lib)

add_executable(test_gradcheck tests/test_gradcheck.cpp)
target_link_libraries(test_gradcheck tejas_lib)

add_executable(attention examples/attention.cpp)
target_link_libraries(attention tejas_lib)

add_executable(transformer_block examples/transformer_block.cpp)
target_link_libraries(transformer_block tejas_lib)
# ==========================================
# Hardware-Specific Executables (Skipped in CI)
# ==========================================
if(NOT SKIP_CUDA)
    add_executable(test_dispatch tests/test_dispatch.cpp)
    target_link_libraries(test_dispatch tejas_lib)

    add_executable(test_parity tests/test_parity.cpp)
    target_link_libraries(test_parity tejas_lib)
endif()