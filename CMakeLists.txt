cmake_minimum_required(VERSION 3.18)
project(FastTrackerCUDA VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type (Release for performance)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Project-specific variables
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# Find dependencies
find_package(OpenCV REQUIRED)

# Enable CUDA
enable_language(CUDA)

# Include directories
include_directories(${INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# CUDA configuration for GTX 1060 (compute capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Source files
set(CPU_SOURCES
    ${SOURCE_DIR}/cpu/main.cpp
    ${SOURCE_DIR}/cpu/fast_tracker.cpp
)

set(CUDA_SOURCES
    ${SOURCE_DIR}/cuda/fast_sad.cu
)

# Create executable
add_executable(FastTrackerCUDA
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(FastTrackerCUDA
    ${OpenCV_LIBS}
)

# Link CUDA libraries (modern way)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(FastTrackerCUDA
    CUDA::cudart
)

# Set target properties
set_target_properties(FastTrackerCUDA PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration summary
message(STATUS "==========================================")
message(STATUS "FastTrackerCUDA Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV Found: ${OpenCV_VERSION}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  Output Directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "==========================================")
