#include "test.h"
#include <iostream>
#include <string>
#include <filesystem>

#define LOG_DEBUG std::cout 
using namespace std; 

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



	if (config.model_path.find("yolo") != std::string::npos)
	{

		std::cout << "yolo.onnx" << std::endl ;

		
		return 1;
	}
	else
	{	
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
}

#if 1
// hpp_ utils_ start
#include <opencv2/dnn.hpp>

struct Detection {
    float confidence;
    cv::Rect bbox;
    int class_id;
    std::string class_name;
};

namespace utils
{
    std::wstring string_to_wstring(const std::string& str);

    size_t vectorProduct(const std::vector<int64_t>& vector);
    // std::wstring charToWstring(const char* str);
    // std::vector<std::string> loadNames(const std::string& path);
    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string>& classNames);

    void letterbox(const cv::Mat& image, cv::Mat& outImage,
                   const cv::Size& newShape,
                   const cv::Scalar& color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

    // template <typename T>
    // T clip(const T& n, const T& lower, const T& upper);
}

// hpp_ utils_ end



// hpp_ yolo_ start
class YOLODetector
{
public:
	YOLODetector() {};
    explicit YOLODetector(std::nullptr_t) {};
    YOLODetector(const std::string& modelPath,
                 const bool& isGPU,
                 const cv::Size& inputSize);

    std::vector<Detection> detect(cv::Mat &image, float confThreshold = 0.25, float iouThreshold = 0.45);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    void preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          float confThreshold, float iouThreshold);

    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                 float& bestConf, int& bestClassId);

    // std::vector<std::string> inputNames;
    std::string inputName;
    // std::vector<std::string> outputNames;
    std::string outputName;
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

};

// hpp_ yolo_ end

// cpp_ utils_ start
//#include "common/utils_onnx.h"

std::wstring utils::string_to_wstring(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
}

size_t utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

// std::wstring utils::charToWstring(const char* str)
// {
//     typedef std::codecvt_utf8<wchar_t> convert_type;
//     std::wstring_convert<convert_type, wchar_t> converter;

//     return converter.from_bytes(str);
// }

// std::vector<std::string> utils::loadNames(const std::string& path)
// {
//     // load class names
//     std::vector<std::string> classNames;
//     std::ifstream infile(path);
//     if (infile.good())
//     {
//         std::string line;
//         while (getline (infile, line))
//         {
//             if (line.back() == '\r')
//                 line.pop_back();
//             classNames.emplace_back(line);
//         }
//         infile.close();
//     }
//     else
//     {
//         std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
//     }

//     return classNames;
// }


void utils::visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
                               const std::vector<std::string>& classNames)
{
    for (const Detection& detection : detections)
    {
        cv::rectangle(image, detection.bbox, cv::Scalar(229, 160, 21), 2);

        int x = detection.bbox.x;
        int y = detection.bbox.y;

        int conf = (int)std::round(detection.confidence * 100);
        int classId = detection.class_id;
        std::string label = classNames[classId] + " 0." + std::to_string(conf);

        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
        cv::rectangle(image,
                      cv::Point(x, y - 25), cv::Point(x + size.width, y),
                      cv::Scalar(229, 160, 21), -1);

        cv::putText(image, label,
                    cv::Point(x, y - 3), cv::FONT_ITALIC,
                    0.8, cv::Scalar(255, 255, 255), 2);
    }
}

void utils::letterbox(const cv::Mat& image, cv::Mat& outImage,
                      const cv::Size& newShape = cv::Size(640, 640),
                      const cv::Scalar& color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2] {r, r};
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int) std::round(((float)coords.width / gain));
    coords.height = (int) std::round(((float)coords.height / gain));
}

// template <typename T>
// T utils::clip(const T& n, const T& lower, const T& upper)
// {
//     return std::max(lower, std::min(n, upper));
// }

// cpp_ utils_ end

// cpp_ yolo_ start
YOLODetector::YOLODetector(const std::string& modelPath,
                           const bool& isGPU = true,
                           const cv::Size& inputSize = cv::Size(640, 640))
{

    LOG_DEBUG << "ONNX Runtime Version: " << Ort::GetVersionString() << std::endl;

    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        LOG_DEBUG << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        LOG_DEBUG << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        LOG_DEBUG << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        LOG_DEBUG << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::string_to_wstring(modelPath);
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        LOG_DEBUG << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    // for (auto shape : inputTensorShape)
    //     std::cout << "Input shape: " << shape << std::endl;

    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();

    this->inputImageShape = cv::Size2f(inputSize);
    LOG_DEBUG << "Input name: " << inputName << std::endl;
    LOG_DEBUG << "Output name: " << outputName << std::endl;
    LOG_DEBUG << "Input shape:" << inputImageShape << endl;
}

void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void YOLODetector::preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape, cv::Scalar(114, 114, 114), this->isDynamicInputShape, false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YOLODetector::postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    float confThreshold, float iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int) (it[0]);
            int centerY = (int) (it[1]);
            int width = (int) (it[2]);
            int height = (int) (it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.bbox = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.bbox, originalImageShape);

        det.confidence = confs[idx];
        det.class_id = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat &image, float confThreshold, float iouThreshold)

{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));

    // 准备输入输出名称数组
    const char* input_names[] = { inputName.c_str() };  // 转换为C风格字符串数组
    const char* output_names[] = { outputName.c_str() };

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              input_names,
                                                              inputTensors.data(),
                                                              1,
                                                              output_names,
                                                              1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    delete[] blob;

    return result;
}

// cpp_ yolo_ end

#endif 

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

	int flag_yolo = 0;

	if (config.model_path.find("yolo") != std::string::npos)
	{

		flag_yolo = 1;
		std::cout << "yolo.onnx" << std::endl ;

		auto _cellCutModel = std::make_shared<YOLODetector>(config.model_path, config.use_gpu, cv::Size(640, 640));

		const float confThreshold = 0.1f;
		const float iouThreshold = 0.2f;


		auto v_fn_imgs = tester.get_image_files(config.input_folder);
		for(auto & eimg: v_fn_imgs)
		{

			cv::Mat image = cv::imread(eimg);
			std::vector<Detection> v_detects = _cellCutModel->detect(image, confThreshold, iouThreshold);

			for(auto & e_detect : v_detects)
			{
		

                // print all e_detect
                std::cout << "Detection: " << e_detect.class_name << " "
                          << "Confidence: " << e_detect.confidence << " "
                          << "BBox: (" << e_detect.bbox.x << ", " << e_detect.bbox.y << ", "
                          << e_detect.bbox.width << ", " << e_detect.bbox.height << ") "
                          << "Class ID: " << e_detect.class_id << std::endl;

			}

		}


	}





	if (config.test_mode) 
	{
		tester.run_all_tests(config.model_path);
		return 0;
	}

	if (!initialize_onnx_test(tester, config)) 
	{
		return -1;
	}


	if (flag_yolo == 1)
	{
	 return 1; 
	}

	return run_inference(tester, config);
}
