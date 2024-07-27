cmake_minimum_required(VERSION 3.10)
project(CUDANormals)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD) 
endif()
# 设置 CUDA 的最低版本
find_package(CUDA REQUIRED)

# 设置编译选项
set(CMAKE_CXX_STANDARD 11)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找当前目录下的所有源文件
file(GLOB SRC_LIST "src/*.cpp" "src/*.cu")

# 添加可执行文件
cuda_add_executable(CUDANormals ${SRC_LIST})

# 链接必要的库
target_link_libraries(CUDANormals ${CUDA_LIBRARIES})

