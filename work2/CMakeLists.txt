
cmake_minimum_required(VERSION 3.26)
project(work2 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(work2_1 work2_1.cu)
add_executable(work2_2 work2_2.cu)

set_target_properties(work2_1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(work2_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
