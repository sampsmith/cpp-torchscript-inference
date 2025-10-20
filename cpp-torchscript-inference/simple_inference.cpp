#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <map>
#include <sstream>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
};

class SimpleInference {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool model_loaded_ = false;
    float confidence_threshold_ = 0.5f;
    int input_size_ = 960;

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
            
            float scale_x = static_cast<float>(original_image.cols) / static_cast<float>(input_size_);
            float scale_y = static_cast<float>(original_image.rows) / static_cast<float>(input_size_);
            
            // Case A: YOLOv5 Hub (channels-first): [1, 5, N] -> [x,y,w,h,conf]
            if (out.dim() == 3 && out.size(0) == 1 && out.size(1) == 5) {
                const int64_t num_detections = out.size(2);
                auto accessor = out.accessor<float, 3>();
                
                for (int64_t i = 0; i < num_detections; ++i) {
                    float confidence = accessor[0][4][i];
                    if (confidence < confidence_threshold_) continue;
                    
                    // Get center coordinates and size in model input space
                    float center_x = accessor[0][0][i];
                    float center_y = accessor[0][1][i];
                    float width = accessor[0][2][i];
                    float height = accessor[0][3][i];
                    
                    // Convert center+size to corner coordinates in model space
                    float x1 = center_x - width * 0.5f;
                    float y1 = center_y - height * 0.5f;
                    float x2 = center_x + width * 0.5f;
                    float y2 = center_y + height * 0.5f;
                    
                    // Scale to image coordinates
                    Detection det;
                    det.x1 = x1 * scale_x;
                    det.y1 = y1 * scale_y;
                    det.x2 = x2 * scale_x;
                    det.y2 = y2 * scale_y;
                    
                    // Clamp to image bounds
                    det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(original_image.cols)));
                    det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(original_image.rows)));
                    det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(original_image.cols)));
                    det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(original_image.rows)));
                    
                    det.confidence = confidence;
                    det.class_id = 0; // YOLOv5 Hub format doesn't include class in this tensor
                    det.class_name = "class_" + std::to_string(det.class_id);
                    
                    detections.push_back(det);
                }
            }
            // Case B: Generic [1, N, 6] -> [x1,y1,x2,y2,conf,class]
            else if (out.dim() == 3 && out.size(2) >= 6) {
                auto accessor = out.accessor<float, 3>();
                int num_detections = out.size(1);
                
                for (int i = 0; i < num_detections; ++i) {
                    float confidence = accessor[0][i][4];
                    if (confidence < confidence_threshold_) continue;
                    
                    Detection det;
                    det.x1 = accessor[0][i][0];
                    det.y1 = accessor[0][i][1]; 
                    det.x2 = accessor[0][i][2];
                    det.y2 = accessor[0][i][3];
                    
                    // Scale coordinates if they appear to be normalized or model-sized
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

public:
    SimpleInference() : device_(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }
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

    void setConfidenceThreshold(float threshold) {
        confidence_threshold_ = threshold;
    }

    void setInputSize(int size) {
        input_size_ = size;
    }

    std::vector<Detection> runInference(const std::string& image_path) {
        std::vector<Detection> detections;
        
        if (!model_loaded_) {
            std::cerr << "Model not loaded!" << std::endl;
            return detections;
        }

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return detections;
        }

        return runInference(image);
    }

    std::vector<Detection> runInference(const cv::Mat& image) {
        std::vector<Detection> detections;
        
        if (!model_loaded_) {
            std::cerr << "Model not loaded!" << std::endl;
            return detections;
        }

        try {
            // Preprocess
            torch::Tensor input_tensor = preprocessImage(image);
            input_tensor = input_tensor.to(device_);
            
            // Inference
            torch::InferenceMode infer_guard;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto output_ivalue = model_.forward(inputs);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
            
            // Postprocess
            auto det_tensor = extractDetectionsTensor(output_ivalue);
            detections = postprocessOutput(det_tensor, image);
            
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
        }
        
        return detections;
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
        label << "Class " << det.class_id << ": " << std::fixed << std::setprecision(2) << det.confidence;
        std::string label_text = label.str();
        
        // Get label size
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.6;
        int thickness = 2;
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, font, font_scale, thickness, &baseline);
        
        // Draw label background
        cv::Point label_bg_pt1(pt1.x, pt1.y - text_size.height - 10);
        cv::Point label_bg_pt2(pt1.x + text_size.width + 10, pt1.y);
        cv::rectangle(image, label_bg_pt1, label_bg_pt2, color, -1);
        
        // Draw label text
        cv::Point text_pt(pt1.x + 5, pt1.y - 5);
        cv::putText(image, label_text, text_pt, font, font_scale, cv::Scalar(255, 255, 255), thickness);
    }
    
    // Add summary text at the bottom
    std::ostringstream summary;
    summary << "Found " << detections.size() << " detections";
    std::string summary_text = summary.str();
    
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    int thickness = 2;
    cv::Size text_size = cv::getTextSize(summary_text, font, font_scale, thickness, nullptr);
    
    cv::Point summary_bg_pt1(10, image.rows - text_size.height - 20);
    cv::Point summary_bg_pt2(text_size.width + 30, image.rows - 5);
    cv::rectangle(image, summary_bg_pt1, summary_bg_pt2, cv::Scalar(0, 0, 0), -1);
    
    cv::Point summary_text_pt(20, image.rows - 15);
    cv::putText(image, summary_text, summary_text_pt, font, font_scale, cv::Scalar(255, 255, 255), thickness);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.pt> <image.jpg> [confidence_threshold] [input_size]" << std::endl;
        std::cout << "Example: " << argv[0] << " model.torchscript.pt test.jpg 0.5 960" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    float confidence = (argc > 3) ? std::stof(argv[3]) : 0.5f;
    int input_size = (argc > 4) ? std::stoi(argv[4]) : 960;

    SimpleInference engine;
    engine.setConfidenceThreshold(confidence);
    engine.setInputSize(input_size);

    if (!engine.loadModel(model_path)) {
        return 1;
    }

    auto detections = engine.runInference(image_path);

    std::cout << "\nFound " << detections.size() << " detections:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        std::cout << "[" << i << "] "
                  << det.class_name << " "
                  << "conf=" << det.confidence << " "
                  << "bbox=[" << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << "]"
                  << std::endl;
    }

    // Save results to JSON
    std::ofstream json_out("detections.json");
    json_out << "{\n  \"detections\": [\n";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        json_out << "    {\n";
        json_out << "      \"class_id\": " << det.class_id << ",\n";
        json_out << "      \"class_name\": \"" << det.class_name << "\",\n";
        json_out << "      \"confidence\": " << det.confidence << ",\n";
        json_out << "      \"bbox\": [" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]\n";
        json_out << "    }" << (i + 1 < detections.size() ? "," : "") << "\n";
    }
    json_out << "  ]\n}";
    json_out.close();
    
    std::cout << "Results saved to detections.json" << std::endl;

    // Create visualization image with bounding boxes
    cv::Mat vis_image = cv::imread(image_path);
    if (!vis_image.empty() && !detections.empty()) {
        drawBoundingBoxes(vis_image, detections);
        std::string output_image = "result_" + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
        if (cv::imwrite(output_image, vis_image)) {
            std::cout << "Visualization saved to " << output_image << std::endl;
        }
    }

    return 0;
}