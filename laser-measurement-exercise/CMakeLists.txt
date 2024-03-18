cmake_minimum_required(VERSION 3.14)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(laser_measurement_exercise)

add_executable (laser_measurement_exercise main.cpp kalman_filter.h kalman_filter.cpp
                measurement_package.h tracking.h tracking.cpp)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                            ${CMAKE_SOURCE_DIR}/media/obj_pose-laser-radar-synthetic-input.txt
                            ${CMAKE_CURRENT_BINARY_DIR}/media/obj_pose-laser-radar-synthetic-input.txt)
