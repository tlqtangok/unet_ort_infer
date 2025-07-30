# unet_ort_infer
use onnx runtime to inference unet model 

only support onnxruntime cpu right now. but easy to extend to gpu

## the project structure
```
/home/jd/t/git/onnxruntime-linux-x64-1.22.0/unet_ort_infer

root@838e9d354ef9:~/t/git/onnxruntime-linux-x64-1.22.0/unet_ort_infer# cd ..
root@838e9d354ef9:~/t/git/onnxruntime-linux-x64-1.22.0# tree -L 2
.
├── GIT_COMMIT_ID
├── LICENSE
├── Privacy.md
├── README.md
├── ThirdPartyNotices.txt
├── VERSION_NUMBER
├── include
│   ├── core
│   ├── cpu_provider_factory.h
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   ├── onnxruntime_cxx_inline.h
│   ├── onnxruntime_float16.h
│   ├── onnxruntime_lite_custom_op.h
│   ├── onnxruntime_run_options_config_keys.h
│   ├── onnxruntime_session_options_config_keys.h
│   └── provider_options.h
├── lib
│   ├── cmake
│   ├── libonnxruntime.so -> libonnxruntime.so.1
│   ├── libonnxruntime.so.1 -> libonnxruntime.so.1.22.0
│   ├── libonnxruntime.so.1.22.0
│   ├── libonnxruntime_providers_shared.so
│   └── pkgconfig
└── unet_ort_infer 
    ├── 1.linux.txt
    ├── 1.tgz
    ├── 1.txt
    ├── CMakeLists.txt
    ├── bak.1.tgz.202507301707
    ├── build
    ├── build.sh
    ├── check.sh
    ├── files.zip
    ├── main.cpp
    ├── run.sh
    ├── tags
    ├── test.cpp
    └── test.h

7 directories, 32 files
```

## ref train with multiple-gpu project 
![https://github.com/tlqtangok/unet_multiple_gpu](https://github.com/tlqtangok/unet_multiple_gpu)


## to build and run the project
```
# go to the project root
bash build.sh

# to run 
./build/test_onnx --model $t/model.onnx --input $m/data/infer/imgs --output $m/data/infer/imgs --batch 2
```

## author 
jd tang from wuhan, china
