#!/bin/bash

echo "Building ONNX Runtime test..."

#rm -rf build

mkdir -p build
cd build

echo "Checking installation paths..."

if [ ! -d "../../include" ]; then
    echo "ERROR: Include directory not found"
    echo "Contents of parent directory:"
    ls -la ..
    exit 1
fi

if [ ! -f "../../lib/libonnxruntime.so" ]; then
    echo "ERROR: Library file not found"
    echo "Contents of lib directory:"
    ls -la ../lib
    exit 1
fi

echo "All required files found. Building..."

cmake ..

cmake --build . --config Release -v

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./test_onnx"
else
    echo "Build failed!"
    exit 1
fi

./test_onnx
