cmake_minimum_required(VERSION 3.12)
project(test_onnx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNX_ROOT ${CMAKE_SOURCE_DIR}/..)
set(ONNX_INCLUDE ${ONNX_ROOT}/include)
set(ONNX_LIB ${ONNX_ROOT}/lib/libonnxruntime.so)

if(NOT EXISTS ${ONNX_INCLUDE})
    message(FATAL_ERROR "Include not found: ${ONNX_INCLUDE}")
endif()

if(NOT EXISTS ${ONNX_LIB})
    message(FATAL_ERROR "Library not found: ${ONNX_LIB}")
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(OPENCV REQUIRED opencv4)

if(NOT OPENCV_FOUND)
    pkg_check_modules(OPENCV REQUIRED opencv)
endif()

include_directories(${ONNX_INCLUDE})
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

set(DNN_TEST_CODE "
#include <opencv2/dnn.hpp>
int main() {
    cv::dnn::Net net;
    return 0;
}
")

file(WRITE "${CMAKE_BINARY_DIR}/test_dnn.cpp" "${DNN_TEST_CODE}")

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${OPENCV_CFLAGS} -c "${CMAKE_BINARY_DIR}/test_dnn.cpp" -o "${CMAKE_BINARY_DIR}/test_dnn.o"
    RESULT_VARIABLE DNN_COMPILE_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(DNN_COMPILE_RESULT EQUAL 0)
    add_definitions(-DHAVE_DNN=1)
    message(STATUS "DNN module found")
else()
    add_definitions(-DHAVE_DNN=0)
    message(WARNING "DNN module not found")
endif()

add_executable(test_onnx
    main.cpp
    test.cpp
)

target_link_libraries(test_onnx 
    ${ONNX_LIB} 
    ${OPENCV_LIBRARIES}
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(test_onnx stdc++fs)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test_onnx PRIVATE -O3)
endif()

set_target_properties(test_onnx PROPERTIES
    BUILD_RPATH "${ONNX_ROOT}/lib"
    INSTALL_RPATH "${ONNX_ROOT}/lib"
)

message(STATUS "OpenCV version: ${OPENCV_VERSION}")
message(STATUS "OpenCV cflags: ${OPENCV_CFLAGS}")
message(STATUS "OpenCV libs: ${OPENCV_LIBRARIES}")
message(STATUS "ONNX Runtime path: ${ONNX_LIB}")

