#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <map>
#include <sstream>
#include <random>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
};

struct BatchResult {
    std::string image_path;
    std::vector<Detection> detections;
    double inference_time_ms;
    bool success;
    std::string error_msg;
};

class BatchInference {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool model_loaded_ = false;
    float confidence_threshold_ = 0.5f;
    int input_size_ = 960;
    int num_threads_ = 1;

    torch::Tensor preprocessImage(const cv::Mat& image) {
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        
        cv::Mat resized_image;
        cv::resize(rgb_image, resized_image, cv::Size(input_size_, input_size_));
        
        torch::Tensor tensor_image = torch::from_blob(
            resized_image.data, 
            {1, resized_image.rows, resized_image.cols, 3}, 
            torch::kUInt8
        );
        
        tensor_image = tensor_image.permute({0, 3, 1, 2}).to(torch::kFloat32) / 255.0;
        return tensor_image;
    }

    torch::Tensor extractDetectionsTensor(const torch::jit::IValue& value) {
        if (value.isTensor()) {
            return value.toTensor();
        } else if (value.isTuple()) {
            auto elements = value.toTuple()->elements();
            for (const auto& el : elements) {
                if (el.isTensor()) {
                    auto t = el.toTensor();
                    if (t.dim() >= 2 && t.size(-1) >= 5) {
                        return t;
                    }
                }
            }
        } else if (value.isList()) {
            auto list = value.toList();
            for (size_t i = 0; i < list.size(); ++i) {
                auto el = list.get(i);
                if (el.isTensor()) {
                    auto t = el.toTensor();
                    if (t.dim() >= 2 && t.size(-1) >= 5) {
                        return t;
                    }
                }
            }
        }
        throw std::runtime_error("Could not extract detection tensor from model output");
    }

    std::vector<Detection> postprocessOutput(const torch::Tensor& output, const cv::Mat& original_image) {
        std::vector<Detection> detections;
        
        try {
            auto out = output.contiguous().to(torch::kCPU);
            
            if (out.dim() == 3 && out.size(2) >= 6) {
                auto accessor = out.accessor<float, 3>();
                int num_detections = out.size(1);
                
                float scale_x = static_cast<float>(original_image.cols) / static_cast<float>(input_size_);
                float scale_y = static_cast<float>(original_image.rows) / static_cast<float>(input_size_);
                
                for (int i = 0; i < num_detections; ++i) {
                    float confidence = accessor[0][i][4];
                    if (confidence < confidence_threshold_) continue;
                    
                    Detection det;
                    det.x1 = accessor[0][i][0];
                    det.y1 = accessor[0][i][1]; 
                    det.x2 = accessor[0][i][2];
                    det.y2 = accessor[0][i][3];
                    
                    // Scale coordinates
                    if (det.x2 <= 2.0f && det.y2 <= 2.0f) {
                        det.x1 *= original_image.cols;
                        det.y1 *= original_image.rows;
                        det.x2 *= original_image.cols;
                        det.y2 *= original_image.rows;
                    } else if (det.x2 <= input_size_ && det.y2 <= input_size_) {
                        det.x1 *= scale_x;
                        det.y1 *= scale_y;
                        det.x2 *= scale_x;
                        det.y2 *= scale_y;
                    }
                    
                    det.confidence = confidence;
                    det.class_id = static_cast<int>(accessor[0][i][5]);
                    det.class_name = "class_" + std::to_string(det.class_id);
                    
                    detections.push_back(det);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in postprocessing: " << e.what() << std::endl;
        }
        
        return detections;
    }

    BatchResult processImage(const std::string& image_path) {
        BatchResult result;
        result.image_path = image_path;
        result.success = false;
        
        try {
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                result.error_msg = "Failed to load image";
                return result;
            }

            torch::Tensor input_tensor = preprocessImage(image);
            input_tensor = input_tensor.to(device_);
            
            torch::InferenceMode infer_guard;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto output_ivalue = model_.forward(inputs);
            auto end = std::chrono::high_resolution_clock::now();
            
            result.inference_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            
            auto det_tensor = extractDetectionsTensor(output_ivalue);
            result.detections = postprocessOutput(det_tensor, image);
            result.success = true;
            
        } catch (const std::exception& e) {
            result.error_msg = e.what();
        }
        
        return result;
    }

public:
    BatchInference() : device_(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }
        num_threads_ = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    bool loadModel(const std::string& model_path) {
        try {
            std::cout << "Loading model: " << model_path << std::endl;
            model_ = torch::jit::load(model_path, device_);
            model_.eval();
            model_ = torch::jit::optimize_for_inference(model_);
            model_loaded_ = true;
            std::cout << "Model loaded successfully" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

    void setConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }
    void setInputSize(int size) { input_size_ = size; }
    void setNumThreads(int threads) { num_threads_ = threads; }

    std::vector<BatchResult> processImageList(const std::vector<std::string>& image_paths) {
        std::vector<BatchResult> results;
        
        if (!model_loaded_) {
            std::cerr << "Model not loaded!" << std::endl;
            return results;
        }

        std::cout << "Processing " << image_paths.size() << " images..." << std::endl;
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // For now, process sequentially (CUDA context issues with threading)
        for (size_t i = 0; i < image_paths.size(); ++i) {
            std::cout << "Processing [" << i+1 << "/" << image_paths.size() << "]: " 
                      << std::filesystem::path(image_paths[i]).filename().string() << std::endl;
            
            auto result = processImage(image_paths[i]);
            results.push_back(result);
            
            if (result.success) {
                std::cout << "  -> " << result.detections.size() << " detections in " 
                          << std::fixed << std::setprecision(1) << result.inference_time_ms << "ms" << std::endl;
            } else {
                std::cout << "  -> ERROR: " << result.error_msg << std::endl;
            }
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
        
        std::cout << "\nBatch processing complete:" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_time_ms << "ms" << std::endl;
        std::cout << "Average per image: " << total_time_ms / image_paths.size() << "ms" << std::endl;
        
        return results;
    }

    std::vector<BatchResult> processDirectory(const std::string& dir_path, const std::vector<std::string>& extensions = {".jpg", ".jpeg", ".png", ".bmp"}) {
        std::vector<std::string> image_paths;
        
        try {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    
                    if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                        image_paths.push_back(entry.path().string());
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error scanning directory: " << e.what() << std::endl;
        }
        
        std::sort(image_paths.begin(), image_paths.end());
        std::cout << "Found " << image_paths.size() << " images in " << dir_path << std::endl;
        
        return processImageList(image_paths);
    }

    void saveResults(const std::vector<BatchResult>& results, const std::string& output_path) {
        std::ofstream json_out(output_path);
        json_out << "{\n";
        json_out << "  \"total_images\": " << results.size() << ",\n";
        
        // Summary stats
        int successful = 0;
        int total_detections = 0;
        double total_time = 0;
        
        for (const auto& result : results) {
            if (result.success) {
                successful++;
                total_detections += result.detections.size();
                total_time += result.inference_time_ms;
            }
        }
        
        json_out << "  \"successful_images\": " << successful << ",\n";
        json_out << "  \"total_detections\": " << total_detections << ",\n";
        json_out << "  \"average_inference_time_ms\": " << (successful > 0 ? total_time / successful : 0) << ",\n";
        
        // Individual results
        json_out << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            json_out << "    {\n";
            json_out << "      \"image_path\": \"" << result.image_path << "\",\n";
            json_out << "      \"success\": " << (result.success ? "true" : "false") << ",\n";
            json_out << "      \"inference_time_ms\": " << result.inference_time_ms << ",\n";
            
            if (!result.success) {
                json_out << "      \"error\": \"" << result.error_msg << "\",\n";
            }
            
            json_out << "      \"detections\": [\n";
            for (size_t j = 0; j < result.detections.size(); ++j) {
                const auto& det = result.detections[j];
                json_out << "        {\n";
                json_out << "          \"class_id\": " << det.class_id << ",\n";
                json_out << "          \"class_name\": \"" << det.class_name << "\",\n";
                json_out << "          \"confidence\": " << det.confidence << ",\n";
                json_out << "          \"bbox\": [" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]\n";
                json_out << "        }" << (j + 1 < result.detections.size() ? "," : "") << "\n";
            }
            json_out << "      ]\n";
            json_out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        json_out << "  ]\n";
        json_out << "}";
        json_out.close();
        
        std::cout << "Results saved to " << output_path << std::endl;
    }
};

// Function to draw bounding boxes on image
void drawBoundingBoxes(cv::Mat& image, const std::vector<Detection>& detections) {
    // Generate distinct colors for different classes
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for consistent colors
    std::uniform_int_distribution<> color_dist(0, 255);
    
    // Color map for classes
    std::map<int, cv::Scalar> class_colors;
    
    for (const auto& det : detections) {
        // Generate or get color for this class
        if (class_colors.find(det.class_id) == class_colors.end()) {
            class_colors[det.class_id] = cv::Scalar(
                color_dist(gen), color_dist(gen), color_dist(gen)
            );
        }
        cv::Scalar color = class_colors[det.class_id];
        
        // Draw bounding box
        cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
        cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
        
        // Draw filled rectangle with transparency
        cv::Mat overlay = image.clone();
        cv::rectangle(overlay, pt1, pt2, color, -1);
        cv::addWeighted(image, 0.7, overlay, 0.3, 0, image);
        
        // Draw border
        cv::rectangle(image, pt1, pt2, color, 2);
        
        // Prepare label text
        std::ostringstream label;
        label << "C" << det.class_id << ": " << std::fixed << std::setprecision(1) << det.confidence;
        std::string label_text = label.str();
        
        // Get label size
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, font, font_scale, thickness, &baseline);
        
        // Draw label background
        cv::Point label_bg_pt1(pt1.x, pt1.y - text_size.height - 8);
        cv::Point label_bg_pt2(pt1.x + text_size.width + 6, pt1.y);
        cv::rectangle(image, label_bg_pt1, label_bg_pt2, color, -1);
        
        // Draw label text
        cv::Point text_pt(pt1.x + 3, pt1.y - 3);
        cv::putText(image, label_text, text_pt, font, font_scale, cv::Scalar(255, 255, 255), thickness);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " <model.pt> <image_directory> [confidence] [output.json]" << std::endl;
        std::cout << "  " << argv[0] << " <model.pt> -list <images.txt> [confidence] [output.json]" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " model.torchscript.pt ./test_images/ 0.5 results.json" << std::endl;
        std::cout << "  " << argv[0] << " model.torchscript.pt -list image_list.txt 0.3" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    float confidence = 0.5f;
    std::string output_path = "batch_results.json";
    
    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find(".json") != std::string::npos) {
            output_path = arg;
        } else {
            try {
                confidence = std::stof(arg);
            } catch (...) {
                std::cerr << "Invalid confidence value: " << arg << std::endl;
                return 1;
            }
        }
    }

    BatchInference engine;
    engine.setConfidenceThreshold(confidence);

    if (!engine.loadModel(model_path)) {
        return 1;
    }

    std::vector<BatchResult> results;
    
    if (input_path == "-list") {
        if (argc < 4) {
            std::cerr << "Image list file required when using -list" << std::endl;
            return 1;
        }
        
        std::string list_file = argv[3];
        std::vector<std::string> image_paths;
        std::ifstream file(list_file);
        std::string line;
        
        while (std::getline(file, line)) {
            if (!line.empty()) {
                image_paths.push_back(line);
            }
        }
        
        results = engine.processImageList(image_paths);
    } else {
        results = engine.processDirectory(input_path);
    }

    engine.saveResults(results, output_path);
    
    return 0;
}