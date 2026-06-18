cmake_minimum_required(VERSION 3.18)
project(tejas LANGUAGES CXX CUDA)

# Force modern C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Explicitly find the CUDA Toolkit so g++ can see its headers
find_package(CUDAToolkit REQUIRED)

# Add compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# 2. Tell CMake to include BOTH your include/ folder AND the CUDA Toolkit headers
include_directories(include ${CUDAToolkit_INCLUDE_DIRS})

# Compile the core library
add_library(tejas_lib 
    src/tensor.cpp 
    src/cuda_backend.cu
)

# Link the CUDA runtime to your library
target_link_libraries(tejas_lib PUBLIC CUDA::cudart)

# Build the test executable
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor tejas_lib)

# Build the XOR demo executable
add_executable(xor examples/xor.cpp)
target_link_libraries(xor tejas_lib)

# Build the Dispatcher Test
add_executable(test_dispatch tests/test_dispatch.cpp)
target_link_libraries(test_dispatch tejas_lib)

# Build the Parity Test
add_executable(test_parity tests/test_parity.cpp)
target_link_libraries(test_parity tejas_lib)