#include "test.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

OnnxTest::OnnxTest()
    : _use_gpu(false), _batch_size(1)
{
}

OnnxTest::~OnnxTest()
{
}

void OnnxTest::set_gpu_mode(bool use_gpu)
{
    _use_gpu = use_gpu;
}

void OnnxTest::set_batch_size(int batch_size)
{
    _batch_size = batch_size;
}

void OnnxTest::print_test_status(const std::string& name, bool result)
{
    std::cout << "[" << (result ? "PASS" : "FAIL") << "] " << name << std::endl;
}

bool OnnxTest::test_api()
{
    try 
    {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (api != nullptr) 
        {
            std::cout << "ONNX Runtime API loaded successfully" << std::endl;
            std::cout << "API Version: " << ORT_API_VERSION << std::endl;
            return true;
        }
        return false;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "API test error: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxTest::test_environment()
{
    try 
    {
        _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test_env");
        if (_env != nullptr) 
        {
            std::cout << "Environment created successfully" << std::endl;
            return true;
        }
        return false;
    }
    catch (const Ort::Exception& e) 
    {
        std::cerr << "Environment error: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxTest::check_file_exists(const std::string& path)
{
    std::ifstream file(path);
    bool exists = file.good();
    if (exists) 
    {
        std::cout << "Model file found: " << path << std::endl;
    }
    else 
    {
        std::cout << "Model file not found: " << path << std::endl;
    }
    return exists;
}

void OnnxTest::setup_execution_provider()
{
    if (_use_gpu) 
    {
        try 
        {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            _options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA execution provider enabled" << std::endl;
        }
        catch (const Ort::Exception& e) 
        {
            std::cerr << "CUDA setup failed: " << e.what() << std::endl;
            std::cout << "Falling back to CPU" << std::endl;
        }
    }
    else 
    {
        std::cout << "Using CPU execution provider" << std::endl;
    }
}

void OnnxTest::setup_input_output_names()
{
    if (!_session) 
    {
        return;
    }
    
    try 
    {
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t input_count = _session->GetInputCount();
        size_t output_count = _session->GetOutputCount();
        
        _input_names.clear();
        _output_names.clear();
        
        for (size_t i = 0; i < input_count; i++) 
        {
            auto name = _session->GetInputNameAllocated(i, allocator);
            _input_names.push_back(std::string(name.get()));
            
            if (i == 0) 
            {
                auto type_info = _session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                _input_shape = tensor_info.GetShape();
            }
        }
        
        for (size_t i = 0; i < output_count; i++) 
        {
            auto name = _session->GetOutputNameAllocated(i, allocator);
            _output_names.push_back(std::string(name.get()));
            
            if (i == 0) 
            {
                auto type_info = _session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                _output_shape = tensor_info.GetShape();
            }
        }
    }
    catch (const Ort::Exception& e) 
    {
        std::cerr << "Error setting up names: " << e.what() << std::endl;
    }
}

void OnnxTest::print_model_info()
{
    if (!_session) 
    {
        return;
    }
    
    try 
    {
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t input_count = _session->GetInputCount();
        size_t output_count = _session->GetOutputCount();
        
        std::cout << "Model Info:" << std::endl;
        std::cout << "  Input count: " << input_count << std::endl;
        std::cout << "  Output count: " << output_count << std::endl;
        std::cout << "  Batch size: " << _batch_size << std::endl;
        std::cout << "  GPU mode: " << (_use_gpu ? "enabled" : "disabled") << std::endl;
        
        for (size_t i = 0; i < input_count; i++) 
        {
            auto name = _session->GetInputNameAllocated(i, allocator);
            auto type_info = _session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "  Input " << i << ": " << name.get() << " [";
            for (size_t j = 0; j < shape.size(); j++) 
            {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        for (size_t i = 0; i < output_count; i++) 
        {
            auto name = _session->GetOutputNameAllocated(i, allocator);
            auto type_info = _session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "  Output " << i << ": " << name.get() << " [";
            for (size_t j = 0; j < shape.size(); j++) 
            {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    catch (const Ort::Exception& e) 
    {
        std::cerr << "Error getting model info: " << e.what() << std::endl;
    }
}

bool OnnxTest::test_model_load(const std::string& path)
{
    try 
    {
        if (!_env) 
        {
            std::cerr << "Environment not initialized" << std::endl;
            return false;
        }
        
        _options.SetInterOpNumThreads(1);
        _options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        
        setup_execution_provider();
        
        _session = std::make_unique<Ort::Session>(*_env, path.c_str(), _options);
        
        std::cout << "Model loaded successfully from: " << path << std::endl;
        print_model_info();
        setup_input_output_names();
        return true;
    }
    catch (const Ort::Exception& e) 
    {
        std::cerr << "Model load error: " << e.what() << std::endl;
        return false;
    }
}


bool ends_with_out_pattern(const std::string& file_path)
{
	size_t dot_pos = file_path.find_last_of('.');
	if (dot_pos == std::string::npos)
	{
		return false;
	}

	std::string name_part = file_path.substr(0, dot_pos);
	return name_part.size() >= 4 && name_part.substr(name_part.size() - 4) == "_OUT";
}


std::vector<std::string> OnnxTest::get_image_files(const std::string& folder_path)
{
    std::vector<std::string> image_files;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    
    try 
    {
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) 
        {
            if (entry.is_regular_file()) 
            {
                std::string file_path = entry.path().string();
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) 
                {
					if (!ends_with_out_pattern(file_path))
					{
						image_files.push_back(file_path);
					}
                }
            }
        }
        
        std::sort(image_files.begin(), image_files.end());
    }
    catch (const std::filesystem::filesystem_error& e) 
    {
        std::cerr << "Error reading folder: " << e.what() << std::endl;
    }
    
    return image_files;
}

std::string OnnxTest::get_output_filename(const std::string& input_filename)
{
    std::filesystem::path input_path(input_filename);
    std::string stem = input_path.stem().string();
    return stem + "_OUT.png";
}

cv::Mat OnnxTest::preprocess_image(const cv::Mat& image)
{
    cv::Mat processed;
    
    int target_height = static_cast<int>(_input_shape[2]);
    int target_width = static_cast<int>(_input_shape[3]);
    cv::resize(image, processed, cv::Size(target_width, target_height));
    
    if (processed.channels() == 3) 
    {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    
    return processed;
}

std::vector<float> OnnxTest::mat_to_chw_vector(const cv::Mat& mat)
{
    std::vector<float> vec;
    int channels = mat.channels();
    int height = mat.rows;
    int width = mat.cols;
    
    vec.resize(channels * height * width);



    std::vector<cv::Mat> v_img; 
    cv::split(mat, v_img);


    size_t offset = 0;
    for (const auto& eimg : v_img)
    {
        if (eimg.isContinuous())
        {
            const float* data = eimg.ptr<float>();
            std::copy(data, data + eimg.total(), vec.begin() + offset);
        }
        else
        {
            size_t current_offset = offset;
            for (int h = 0; h < eimg.rows; ++h)
            {
                const float* row_data = eimg.ptr<float>(h);
                std::copy(row_data, row_data + eimg.cols, vec.begin() + current_offset);
                current_offset += eimg.cols;
            }
        }
        offset += eimg.total();
    }


#if 0
    for (int c = 0; c < channels; ++c) 
    {
        for (int h = 0; h < height; ++h) 
        {
            for (int w = 0; w < width; ++w) 
            {
                vec[c * height * width + h * width + w] = mat.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
#endif 
    
    return vec;
}

cv::Mat OnnxTest::extract_mask_from_output(const std::vector<float>& output_data)
{
    if (_output_shape.size() != 4 || _output_shape[1] != 2) 
    {
        std::cerr << "Invalid output format" << std::endl;
        return cv::Mat();
    }
    
    int height = static_cast<int>(_output_shape[2]);
    int width = static_cast<int>(_output_shape[3]);
    
    const float* data = output_data.data();
    
    cv::Mat class0(height, width, CV_32F);
    cv::Mat class1(height, width, CV_32F);
    
    int channel_size = height * width;
    
    for (int i = 0; i < channel_size; i++) 
    {
        class0.at<float>(i / width, i % width) = data[i];
        class1.at<float>(i / width, i % width) = data[i + channel_size];
    }
    
    cv::Mat mask;
    cv::compare(class1, class0, mask, cv::CMP_GT);
    
    return mask;
}

bool OnnxTest::save_result(const cv::Mat& result, const std::string& output_path)
{
    try 
    {
        return cv::imwrite(output_path, result);
    }
    catch (const cv::Exception& e) 
    {
        std::cerr << "Error saving result: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxTest::single_inference(const std::string& image_path, const std::string& output_folder)
{
    if (!_session) 
    {
        std::cerr << "Model not loaded" << std::endl;
        return false;
    }
    
    try 
    {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) 
        {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return false;
        }
        
        cv::Mat processed = preprocess_image(image);
        std::vector<float> input_data = mat_to_chw_vector(processed);
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), _input_shape.data(), _input_shape.size());
        
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        
        for (const auto& name : _input_names) 
        {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : _output_names) 
        {
            output_names_cstr.push_back(name.c_str());
        }
        
        auto output_tensors = _session->Run(Ort::RunOptions{nullptr}, 
                                          input_names_cstr.data(), &input_tensor, 1,
                                          output_names_cstr.data(), output_names_cstr.size());
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> output_vec(output_data, output_data + output_size);
        
        cv::Mat mask = extract_mask_from_output(output_vec);
        if (mask.empty()) 
        {
            std::cerr << "Failed to extract mask" << std::endl;
            return false;
        }
        
        std::string output_filename = get_output_filename(image_path);
        std::string output_path = std::filesystem::path(output_folder) / output_filename;
        
        std::filesystem::create_directories(output_folder);
        
        bool success = save_result(mask, output_path);
        if (success) 
        {
            std::cout << "Mask saved to: " << output_path << std::endl;
        }
        
        return success;
    }
    catch (const Ort::Exception& e) 
    {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxTest::batch_inference(const std::vector<std::string>& image_paths, const std::string& output_folder)
{
    if (!_session) 
    {
        std::cerr << "Model not loaded" << std::endl;
        return false;
    }
    
    bool all_success = true;
    size_t total_images = image_paths.size();
    
    for (size_t i = 0; i < total_images; i += _batch_size) 
    {
        size_t current_batch_size = std::min(static_cast<size_t>(_batch_size), total_images - i);
        
        std::cout << "Processing batch " << (i / _batch_size + 1) 
                  << " (" << current_batch_size << " images)" << std::endl;
        
        for (size_t j = 0; j < current_batch_size; ++j) 
        {
            std::string image_path = image_paths[i + j];
            std::cout << "  Processing: " << std::filesystem::path(image_path).filename() << std::endl;
            
            bool success = single_inference(image_path, output_folder);
            if (!success) 
            {
                std::cerr << "  Failed to process: " << image_path << std::endl;
                all_success = false;
            }
        }
    }
    
    return all_success;
}

bool OnnxTest::process_folder(const std::string& input_folder, const std::string& output_folder)
{
    std::vector<std::string> image_files = get_image_files(input_folder);
    
    if (image_files.empty()) 
    {
        std::cerr << "No image files found in folder: " << input_folder << std::endl;
        return false;
    }
    
    std::cout << "Found " << image_files.size() << " image files" << std::endl;
    
    if (_batch_size > 1) 
    {
        return batch_inference(image_files, output_folder);
    }
    else 
    {
        bool all_success = true;
        for (const auto& image_path : image_files) 
        {
            std::cout << "Processing: " << std::filesystem::path(image_path).filename() << std::endl;
            bool success = single_inference(image_path, output_folder);
            if (!success) 
            {
                all_success = false;
            }
        }
        return all_success;
    }
}

void OnnxTest::run_all_tests(const std::string &model_path)
{
    std::cout << "=== ONNX Runtime Test Suite ===" << std::endl;
    
    bool api_test = test_api();
    print_test_status("API Loading", api_test);
    
    bool env_test = test_environment();
    print_test_status("Environment", env_test);
    
    bool file_exists = check_file_exists(model_path);
    print_test_status("Model File Check", file_exists);
    
    bool model_load = false;
    if (file_exists && env_test) 
    {
        model_load = test_model_load(model_path);
    }
    print_test_status("Model Loading", model_load);
    
    std::cout << "===============================" << std::endl;
    
    if (api_test && env_test && file_exists && model_load) 
    {
        std::cout << "SUCCESS: All tests passed!" << std::endl;
        std::cout << "Your ONNX Runtime installation is working correctly." << std::endl;
    } 
    else 
    {
        std::cout << "Results summary:" << std::endl;
        if (!api_test) std::cout << "  - API loading failed" << std::endl;
        if (!env_test) std::cout << "  - Environment creation failed" << std::endl;
        if (!file_exists) std::cout << "  - Model file not accessible" << std::endl;
        if (!model_load) std::cout << "  - Model loading failed" << std::endl;
    }
}
