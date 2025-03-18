cmake_minimum_required(VERSION 3.17)
project(work1 CUDA)
include_directories(./include)
find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(work1 work1.cu)

set_target_properties(
        work1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
