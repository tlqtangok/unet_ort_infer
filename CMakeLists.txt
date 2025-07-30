cmake_minimum_required(VERSION 3.12)
project(test_onnx)

set(CMAKE_CXX_STANDARD 17)

set(ONNX_ROOT ${CMAKE_SOURCE_DIR}/..)
set(ONNX_INCLUDE ${ONNX_ROOT}/include)
set(ONNX_LIB ${ONNX_ROOT}/lib/libonnxruntime.so)

if(NOT EXISTS ${ONNX_INCLUDE})
    message(FATAL_ERROR "Include not found: ${ONNX_INCLUDE}")
endif()

if(NOT EXISTS ${ONNX_LIB})
    message(FATAL_ERROR "Library not found: ${ONNX_LIB}")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(${ONNX_INCLUDE})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_onnx
    main.cpp
    test.cpp
)

target_link_libraries(test_onnx ${ONNX_LIB} ${OpenCV_LIBS})

set_target_properties(test_onnx PROPERTIES
    BUILD_RPATH "${ONNX_ROOT}/lib"
    INSTALL_RPATH "${ONNX_ROOT}/lib"
)