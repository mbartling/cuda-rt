cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)
#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -std=c++11 -gencode arch=compute_35,code=sm_35
#    )
#list(APPEND CUDA_NVCC_FLAGS "-arch=compute_50;-code=sm_50;-std=c++11;-O3;-lcudadevrt")
#list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35;-code=sm_35;-gencode=arch=compute_50;-code=sm_50;-std=c++11;-O0;-g;-G;-lcudadevrt")
list(APPEND CUDA_NVCC_FLAGS "-arch=compute_35;-code=sm_35;-std=c++11;-O3;-lcudadevrt")

#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O3;-DVERBOSE; ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -std=c++11")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD ON)



# Specify include directories
include_directories(inc)
CUDA_INCLUDE_DIRECTORIES(
    kernels
    inc
    )
file(GLOB SOURCES src/*.cpp src/*.cc)
file(GLOB CUDA_SOURCES kernels/*.cu)

cuda_add_executable(cuda_bvh ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(cuda_bvh ${CUDA_LIBRARIES})
target_link_libraries(cuda_bvh ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudadevrt.a)
message(STATUS ${CUDA_LIBRARIES})
message(STATUS ${CUDA_TOOLKIT_ROOT_DIR})
#leave this out for now
#add_executable(cpu_bvh ${SOURCES})
