cmake_minimum_required(VERSION 3.18)
project(tejas LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_executable(test_tensor tests/test_tensor.cpp src/tensor.cpp)
