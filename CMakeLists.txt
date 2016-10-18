cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION TRUE)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIR} )

include_directories(inc)
#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_22,code=sm_22
#    )

# Specify include directories

file(GLOB SOURCES src/*/*.cpp src/*/*.cc)
file(GLOB CUDA_SOURCES src/kernels/*.cu)

cuda_add_executable(raytrace ${SOURCES} ${CUDA_SOURCES} OPTIONS -arch=sm_35 -rdc=true -lcudadevrt)

target_link_libraries(raytrace ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${CUDA_LIBRARIES} )

