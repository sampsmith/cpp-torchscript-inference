#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

class BenchmarkInference {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool model_loaded_ = false;
    int input_size_ = 960;
    
    torch::Tensor createDummyInput() {
        // Create a dummy RGB tensor
        torch::Tensor tensor = torch::rand({1, 3, input_size_, input_size_}, torch::kFloat32);
        if (device_.is_cuda()) {
            tensor = tensor.to(device_);
        }
        return tensor;
    }

    torch::Tensor createRealInput(const cv::Mat& image) {
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
        return tensor_image.to(device_);
    }

public:
    BenchmarkInference() : device_(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }
        std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    }

    bool loadModel(const std::string& model_path) {
        try {
            std::cout << "\nLoading model: " << model_path << std::endl;
            model_ = torch::jit::load(model_path, device_);
            model_.eval();
            
            // Apply optimizations
            model_ = torch::jit::optimize_for_inference(model_);
            
            model_loaded_ = true;
            std::cout << "Model loaded and optimized successfully" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

    void setInputSize(int size) {
        input_size_ = size;
        std::cout << "Input size set to: " << size << "x" << size << std::endl;
    }

    struct BenchmarkResults {
        std::vector<double> inference_times_ms;
        double mean_ms;
        double std_ms;
        double min_ms;
        double max_ms;
        double median_ms;
        double p95_ms;
        double p99_ms;
        double throughput_fps;
        int warmup_iterations;
        int benchmark_iterations;
        bool use_real_image;
        std::string device_name;
    };

    BenchmarkResults runBenchmark(int warmup_iterations = 50, int benchmark_iterations = 100, const std::string& image_path = "") {
        BenchmarkResults results;
        results.warmup_iterations = warmup_iterations;
        results.benchmark_iterations = benchmark_iterations;
        results.use_real_image = !image_path.empty();
        results.device_name = device_.is_cuda() ? "CUDA" : "CPU";
        
        if (!model_loaded_) {
            std::cerr << "Model not loaded!" << std::endl;
            return results;
        }

        torch::Tensor input_tensor;
        cv::Mat image;
        
        if (!image_path.empty()) {
            image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                return results;
            }
            std::cout << "Using real image: " << image_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;
        } else {
            std::cout << "Using synthetic random input" << std::endl;
        }

        std::cout << "\nWarmup phase: " << warmup_iterations << " iterations..." << std::endl;
        
        // Warmup phase
        {
            torch::InferenceMode infer_guard;
            for (int i = 0; i < warmup_iterations; ++i) {
                if (!image_path.empty()) {
                    input_tensor = createRealInput(image);
                } else {
                    input_tensor = createDummyInput();
                }
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                auto output = model_.forward(inputs);
                
                if (device_.is_cuda()) {
                    torch::cuda::synchronize();
                }
                
                if (i % 10 == 0) {
                    std::cout << "." << std::flush;
                }
            }
        }
        std::cout << " done\n" << std::endl;

        std::cout << "Benchmark phase: " << benchmark_iterations << " iterations..." << std::endl;
        
        // Benchmark phase
        results.inference_times_ms.reserve(benchmark_iterations);
        
        {
            torch::InferenceMode infer_guard;
            for (int i = 0; i < benchmark_iterations; ++i) {
                if (!image_path.empty()) {
                    input_tensor = createRealInput(image);
                } else {
                    input_tensor = createDummyInput();
                }
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Precise timing
                auto start = std::chrono::high_resolution_clock::now();
                auto output = model_.forward(inputs);
                
                if (device_.is_cuda()) {
                    torch::cuda::synchronize();
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                
                double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                results.inference_times_ms.push_back(time_ms);
                
                if (i % 10 == 0) {
                    std::cout << "." << std::flush;
                }
            }
        }
        std::cout << " done\n" << std::endl;

        // Calculate statistics
        auto times = results.inference_times_ms;
        std::sort(times.begin(), times.end());
        
        results.min_ms = times.front();
        results.max_ms = times.back();
        results.median_ms = times[times.size() / 2];
        results.p95_ms = times[static_cast<size_t>(times.size() * 0.95)];
        results.p99_ms = times[static_cast<size_t>(times.size() * 0.99)];
        
        results.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        double variance = 0;
        for (double time : times) {
            variance += (time - results.mean_ms) * (time - results.mean_ms);
        }
        results.std_ms = std::sqrt(variance / times.size());
        
        results.throughput_fps = 1000.0 / results.mean_ms;
        
        return results;
    }

    void printResults(const BenchmarkResults& results) {
        std::cout << "=== BENCHMARK RESULTS ===" << std::endl;
        std::cout << "Device: " << results.device_name << std::endl;
        std::cout << "Input size: " << input_size_ << "x" << input_size_ << std::endl;
        std::cout << "Warmup iterations: " << results.warmup_iterations << std::endl;
        std::cout << "Benchmark iterations: " << results.benchmark_iterations << std::endl;
        std::cout << "Using real image: " << (results.use_real_image ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Inference Time Statistics (ms):" << std::endl;
        std::cout << "  Mean:   " << results.mean_ms << std::endl;
        std::cout << "  Std:    " << results.std_ms << std::endl;
        std::cout << "  Min:    " << results.min_ms << std::endl;
        std::cout << "  Max:    " << results.max_ms << std::endl;
        std::cout << "  Median: " << results.median_ms << std::endl;
        std::cout << "  P95:    " << results.p95_ms << std::endl;
        std::cout << "  P99:    " << results.p99_ms << std::endl;
        std::cout << std::endl;
        
        std::cout << std::setprecision(1);
        std::cout << "Throughput: " << results.throughput_fps << " FPS" << std::endl;
        std::cout << std::endl;
        
        // Performance analysis
        double cv = results.std_ms / results.mean_ms * 100; // Coefficient of variation
        std::cout << "Performance Analysis:" << std::endl;
        std::cout << "  Stability (CV): " << std::setprecision(2) << cv << "%" << std::endl;
        
        if (cv < 5) {
            std::cout << "  Assessment: Excellent stability" << std::endl;
        } else if (cv < 10) {
            std::cout << "  Assessment: Good stability" << std::endl;
        } else if (cv < 20) {
            std::cout << "  Assessment: Moderate stability" << std::endl;
        } else {
            std::cout << "  Assessment: High variance - check system load" << std::endl;
        }
        
        if (results.device_name == "CUDA") {
            if (results.throughput_fps > 100) {
                std::cout << "  GPU Performance: Excellent for real-time applications" << std::endl;
            } else if (results.throughput_fps > 30) {
                std::cout << "  GPU Performance: Good for most applications" << std::endl;
            } else {
                std::cout << "  GPU Performance: Consider model optimization" << std::endl;
            }
        }
        
        std::cout << "=========================" << std::endl;
    }

    void saveResultsCSV(const BenchmarkResults& results, const std::string& filename) {
        std::ofstream csv_file(filename);
        csv_file << "iteration,inference_time_ms" << std::endl;
        
        for (size_t i = 0; i < results.inference_times_ms.size(); ++i) {
            csv_file << i << "," << results.inference_times_ms[i] << std::endl;
        }
        
        csv_file.close();
        std::cout << "Detailed results saved to: " << filename << std::endl;
    }

    void compareInputSizes(const std::vector<int>& sizes, int iterations = 50, const std::string& image_path = "") {
        std::cout << "\n=== INPUT SIZE COMPARISON ===" << std::endl;
        std::cout << "Sizes to test: ";
        for (int size : sizes) {
            std::cout << size << "x" << size << " ";
        }
        std::cout << std::endl << std::endl;
        
        std::vector<BenchmarkResults> all_results;
        
        for (int size : sizes) {
            setInputSize(size);
            std::cout << "Testing " << size << "x" << size << "..." << std::endl;
            
            auto result = runBenchmark(20, iterations, image_path);
            all_results.push_back(result);
        }
        
        // Print comparison table
        std::cout << "\nComparison Summary:" << std::endl;
        std::cout << std::left << std::setw(8) << "Size" 
                  << std::setw(12) << "Mean (ms)"
                  << std::setw(12) << "Throughput" 
                  << std::setw(12) << "Memory"
                  << std::endl;
        std::cout << std::string(44, '-') << std::endl;
        
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto& result = all_results[i];
            double memory_factor = (sizes[i] * sizes[i]) / double(sizes[0] * sizes[0]);
            
            std::cout << std::left << std::setw(8) << (std::to_string(sizes[i]) + "x" + std::to_string(sizes[i]))
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.mean_ms
                      << std::setw(12) << std::setprecision(1) << result.throughput_fps << " FPS"
                      << std::setw(12) << std::setprecision(2) << memory_factor << "x mem"
                      << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " <model.pt> [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --image <path>         Use real image instead of synthetic data" << std::endl;
        std::cout << "  --size <width>         Input size (default: 960)" << std::endl;
        std::cout << "  --warmup <num>         Warmup iterations (default: 50)" << std::endl;
        std::cout << "  --iterations <num>     Benchmark iterations (default: 100)" << std::endl;
        std::cout << "  --compare-sizes        Test multiple input sizes" << std::endl;
        std::cout << "  --csv <filename>       Save detailed results to CSV" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " model.torchscript.pt" << std::endl;
        std::cout << "  " << argv[0] << " model.torchscript.pt --image test.jpg --iterations 200" << std::endl;
        std::cout << "  " << argv[0] << " model.torchscript.pt --compare-sizes --csv results.csv" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = "";
    int input_size = 960;
    int warmup_iterations = 50;
    int benchmark_iterations = 100;
    bool compare_sizes = false;
    std::string csv_output = "";

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            input_size = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            benchmark_iterations = std::stoi(argv[++i]);
        } else if (arg == "--compare-sizes") {
            compare_sizes = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_output = argv[++i];
        }
    }

    BenchmarkInference engine;

    if (!engine.loadModel(model_path)) {
        return 1;
    }

    if (compare_sizes) {
        std::vector<int> sizes = {416, 640, 960, 1280};
        engine.compareInputSizes(sizes, benchmark_iterations / 2, image_path);
    } else {
        engine.setInputSize(input_size);
        auto results = engine.runBenchmark(warmup_iterations, benchmark_iterations, image_path);
        engine.printResults(results);
        
        if (!csv_output.empty()) {
            engine.saveResultsCSV(results, csv_output);
        }
    }

    return 0;
}