cmake_minimum_required(VERSION 3.6)
project(parallel-loop-1)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(parallel-loop-1 ${SOURCE_FILES})

# for openmp
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
