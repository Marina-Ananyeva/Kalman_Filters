cmake_minimum_required(VERSION 3.14)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(ukf_exercise)

add_executable (ukf_exercise main.cpp ukf.h ukf.cpp)