cmake_minimum_required(VERSION 3.26)
project(work3 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(work3 work3.cu)

set_target_properties(work3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
