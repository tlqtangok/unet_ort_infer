#ifndef TEST_H
#define TEST_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

class OnnxTest 
{
private:
    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _session;
    Ort::SessionOptions _options;
    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;
    std::vector<int64_t> _input_shape;
    std::vector<int64_t> _output_shape;
    bool _use_gpu;
    int _batch_size;

public:
    OnnxTest();
    ~OnnxTest();
    
    bool test_api();
    bool test_environment();
    bool test_model_load(const std::string& path);
    bool check_file_exists(const std::string& path);
    void run_all_tests(const std::string & model_path);
    
    bool process_folder(const std::string& input_folder, const std::string& output_folder);
    bool batch_inference(const std::vector<std::string>& image_paths, const std::string& output_folder);
    bool single_inference(const std::string& image_path, const std::string& output_folder);
    
    void set_gpu_mode(bool use_gpu);
    void set_batch_size(int batch_size);
    
private:
    void print_test_status(const std::string& name, bool result);
    void print_model_info();
    void setup_input_output_names();
    void setup_execution_provider();
    
    cv::Mat preprocess_image(const cv::Mat& image);
    cv::Mat extract_mask_from_output(const std::vector<float>& output_data);
    bool save_result(const cv::Mat& result, const std::string& output_path);
    std::vector<float> mat_to_chw_vector(const cv::Mat& mat);
    std::vector<std::string> get_image_files(const std::string& folder_path);
    std::string get_output_filename(const std::string& input_filename);
};

#endif