cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(tsdf_cuda SHARED
  tsdf.cpp
  tsdf.cu)

CUDA_ADD_CUBLAS_TO_TARGET( tsdf_cuda )

target_link_libraries(tsdf_cuda
  ${PYTHON_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES})

set_target_properties(tsdf_cuda PROPERTIES SUFFIX .pyd)
set_target_properties(tsdf_cuda PROPERTIES PREFIX "")