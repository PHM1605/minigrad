cmake_minimum_required(VERSION 3.18)
project(minigrad LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA enable
find_package(CUDA)
if (CUDA_FOUND)
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA)
  message(STATUS ">>> CUDA backend enabled")
endif()

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CPU_SRC "")
file(GLOB CPU_SRC src/*.cpp)

set(CUDA_SRC "")
if (CUDA_FOUND)
  file(GLOB CUDA_SRC src/cuda/*.cu)
endif()

# Build static library
add_library(minigrad STATIC
  ${CPU_SRC}
  ${CUDA_SRC}
)
# Link outside library
if (CUDA_FOUND)
  target_link_libraries(minigrad PUBLIC cuda cudart)
endif()

# Build examples/*.cpp
file(GLOB EXAMPLE_SOURCES examples/*.cpp)
# NAME_WE=NameWithoutExtension => program_name='mnist'
foreach(EG_SRC ${EXAMPLE_SOURCES})
  get_filename_component(program_name ${EG_SRC} NAME_WE)
  add_executable(${program_name} ${EG_SRC})
  target_link_libraries(${program_name} PRIVATE minigrad)
endforeach()
