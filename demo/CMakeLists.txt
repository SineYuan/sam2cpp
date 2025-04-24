cmake_minimum_required(VERSION 3.10)
project(sam2_mnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

include_directories(${SAM2CPP_DIR}/include)
link_directories(${SAM2CPP_DIR}/lib)


add_executable(sam2_image_demo sam2_image_demo.cpp )
target_link_libraries(sam2_image_demo
    PRIVATE
    sam2
    ${OpenCV_LIBS}
)

add_executable(sam2_video_demo sam2_video_demo.cpp )
target_link_libraries(sam2_video_demo
    PRIVATE
    sam2
    ${OpenCV_LIBS}
)