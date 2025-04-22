cmake_minimum_required(VERSION 3.10)
project(sam2_mnn)

option(SAM2_USE_STATIC_MNN "Use static MNN library" ON)
option(MNN_DIR "MNN library directory" "")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV
#find_package(OpenCV REQUIRED)
#if(OpenCV_FOUND)
#    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
#    include_directories(${OpenCV_INCLUDE_DIRS})
#else()
#    message(FATAL_ERROR "OpenCV not found")
#endif()

# 设置MNN路径
#set(MNN_INCLUDE_DIR /data/learnzone/MNN/build2/install/include)
#set(MNN_DIR /data/learnzone/MNN/build2/install/lib)

message(STATUS "MNN_DIR: ${MNN_DIR}")

# 如果MNN_INCLUDE_DIR和MNN_LIB_DIR没有设置，则从MNN_DIR读取拼接
if (NOT DEFINED MNN_INCLUDE_DIR OR NOT DEFINED MNN_LIB_DIR)
    if (MNN_DIR)
        message(STATUS "MNN_DIR is set, using it to set MNN_INCLUDE_DIR and MNN_LIB_DIR")
        set(MNN_INCLUDE_DIR ${MNN_DIR}/include)
        set(MNN_LIB_DIR ${MNN_DIR}/lib)
    else()
        message(WARNING "MNN_INCLUDE_DIR and MNN_LIB_DIR are not set, and MNN_DIR is not set")
    endif()
endif()

message(STATUS "MNN_INCLUDE_DIR: ${MNN_INCLUDE_DIR}")
message(STATUS "MNN_LIB_DIR: ${MNN_LIB_DIR}")

# 添加头文件路径
include_directories(${MNN_INCLUDE_DIR})

# Add spdlog
add_subdirectory(third_party/spdlog)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)

# Set header files
set(HEADER_FILES
    src/sam2_common.h
    src/sam2_base.h
    src/sam2_image.h
    src/sam2_video.h
    src/expected.hpp
)

set(SOURCE_FILES
    src/tar.cpp
    src/sam2_base.cpp
    src/sam2_image.cpp
    src/sam2_video.cpp
)

# Create library
add_library(sam2 SHARED ${SOURCE_FILES})
if(SAM2_USE_STATIC_MNN)
    target_link_libraries(sam2 
        PRIVATE
        -Wl,--whole-archive
        ${MNN_LIB_DIR}/libMNN.a
        spdlog::spdlog
        -Wl,--no-whole-archive
    )
else()
    file(GLOB MNN_LIBS ${MNN_LIB_DIR}/*.so)
    foreach(MNN_LIB ${MNN_LIBS})
        message(STATUS "MNN_LIB: ${MNN_LIB}")
    endforeach()

    target_link_libraries(sam2 
        PRIVATE
        ${MNN_LIBS}
        spdlog::spdlog
    )
endif()

# Set output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Installation
install(TARGETS sam2
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADER_FILES}
    DESTINATION include
)

# Install MNN library
#install(FILES ${MNN_LIB_DIR}/libMNN.a
#    DESTINATION lib
#)

# Install OpenCV dependencies
#install(FILES ${OpenCV_LIBS}
#    DESTINATION lib
#) 