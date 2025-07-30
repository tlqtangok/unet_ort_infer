#include "test.h"
#include <iostream>
#include <string>
#include <filesystem>

struct Config 
{
    std::string model_path;
    std::string input_folder;
    std::string output_folder;
    bool use_gpu = false;
    int batch_size = 1;
    bool test_mode = false;
};

void print_usage()
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  Test mode: ./test_onnx model.onnx" << std::endl;
    std::cout << "  Inference: ./test_onnx --model <model_path> --input <input_folder> --output <output_folder> [--gpu] [--batch <size>]" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./test_onnx --model checkpoint_epoch1.onnx --input ./imgs --output ./results" << std::endl;
    std::cout << "  ./test_onnx --model checkpoint_epoch1.onnx --gpu --input ./imgs --output ./results --batch 4" << std::endl;
}

bool parse_arguments(int argc, char** argv, Config& config)
{
    for (int i = 1; i < argc; i++) 
    {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) 
        {
            config.model_path = argv[++i];
        }
        else if (arg == "--input" && i + 1 < argc) 
        {
            config.input_folder = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) 
        {
            config.output_folder = argv[++i];
        }
        else if (arg == "--gpu") 
        {
            config.use_gpu = true;
        }
        else if (arg == "--batch" && i + 1 < argc) 
        {
            config.batch_size = std::stoi(argv[++i]);
            if (config.batch_size <= 0) 
            {
                std::cerr << "Error: Batch size must be positive" << std::endl;
                return false;
            }
        }
        else if (arg[0] != '-') 
        {
            config.model_path = arg;
            config.test_mode = true;
        }
    }
    
    return true;
}

bool validate_config(const Config& config)
{
    if (config.model_path.empty()) 
    {
        std::cerr << "Error: Model path is required" << std::endl;
        return false;
    }
    
    if (!config.test_mode) 
    {
        if (config.input_folder.empty() || config.output_folder.empty()) 
        {
            std::cerr << "Error: --input and --output folders are required for inference" << std::endl;
            return false;
        }
        
        if (!std::filesystem::exists(config.input_folder)) 
        {
            std::cerr << "Error: Input folder does not exist: " << config.input_folder << std::endl;
            return false;
        }
        
        if (!std::filesystem::is_directory(config.input_folder)) 
        {
            std::cerr << "Error: Input path is not a directory: " << config.input_folder << std::endl;
            return false;
        }
    }
    
    return true;
}

bool initialize_onnx_test(OnnxTest& tester, const Config& config)
{
    if (!tester.check_file_exists(config.model_path)) 
    {
        return false;
    }
    
    if (!tester.test_environment()) 
    {
        std::cerr << "Failed to initialize ONNX Runtime environment" << std::endl;
        return false;
    }
    
    tester.set_gpu_mode(config.use_gpu);
    tester.set_batch_size(config.batch_size);
    
    if (!tester.test_model_load(config.model_path)) 
    {
        std::cerr << "Failed to load model: " << config.model_path << std::endl;
        return false;
    }
    
    return true;
}

int run_inference(OnnxTest& tester, const Config& config)
{
    std::cout << "Starting inference..." << std::endl;
    std::cout << "Input folder: " << config.input_folder << std::endl;
    std::cout << "Output folder: " << config.output_folder << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "GPU mode: " << (config.use_gpu ? "enabled" : "disabled") << std::endl;
    
    bool success = tester.process_folder(config.input_folder, config.output_folder);
    
    if (success) 
    {
        std::cout << "Inference completed successfully" << std::endl;
        return 0;
    }
    else 
    {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }
}

// main_
int main(int argc, char** argv)
{
    std::cout << "ONNX Runtime UNet Inference Tool" << std::endl;
    
    if (argc <= 1) 
    {
        print_usage();
        return -1;
    }
    
    Config config;
    
    if (!parse_arguments(argc, argv, config)) 
    {
        print_usage();
        return -1;
    }
    
    if (!validate_config(config)) 
    {
        print_usage();
        return -1;
    }
    
    OnnxTest tester;
    
    if (config.test_mode) 
    {
        tester.run_all_tests(config.model_path);
        return 0;
    }
    
    if (!initialize_onnx_test(tester, config)) 
    {
        return -1;
    }
    
    return run_inference(tester, config);
}
