cmake_minimum_required(VERSION 2.8)
project(UFAST)

set(INC_DIR /home/d/tool/ncnn/build/install/include)
set(LINK_DIR /home/d/tool/ncnn/build/install/lib/libncnn.a)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${INC_DIR} ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(LaneDet ./src/LaneDet.cpp)
target_link_libraries(LaneDet ${LINK_DIR} ${OpenCV_LIBS})
