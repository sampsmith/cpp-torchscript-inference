# YOLO TorchScript C++ Inference Engines

High-performance C++ command-line tools for running YOLO TorchScript model inference. Built for production computer vision workflows with GPU acceleration support.

## Overview

This project provides three specialized inference engines designed for YOLO model deployment:

- **Simple Inference**: Single image processing with detailed detection output
- **Batch Processing**: High-throughput directory and file list processing  
- **Performance Benchmarking**: Comprehensive model performance analysis

The engines are built with modern C++17, support both CPU and CUDA execution, and provide JSON output for easy integration into larger systems.

## Features

**Performance & Hardware**
- CUDA GPU acceleration with automatic CPU fallback
- Optimized memory management with pinned memory transfers
- Model warmup and JIT optimization for consistent performance
- Support for various input sizes (416×416 to 1280×1280)

**Flexibility & Integration**
- Multiple image formats (JPEG, PNG, BMP, TIFF)
- Configurable confidence thresholds and preprocessing
- JSON output format for structured results
- Automatic image visualization with bounding box overlays
- Cross-platform build system (Linux, macOS, Windows)

**Production Ready**
- Comprehensive error handling and logging
- Progress tracking for batch operations
- Self-contained executables with minimal dependencies
- Extensive documentation and examples

## Quick Start

### Prerequisites

- CMake 3.18 or later
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- OpenCV 4.x
- PyTorch C++ (LibTorch) - downloaded automatically by build script
- CUDA Toolkit (optional, for GPU acceleration)

### Building

```bash
git clone https://github.com/sampsmith/cpp-torchscript-inference.git
cd cpp-torchscript-inference
chmod +x build.sh
./build.sh
```

The build script will automatically download LibTorch and configure the build environment.

### Basic Usage

```bash
# Single image inference
./run_inference.sh simple sample_data/models/your_model.pt sample_data/images/test.jpg

# Batch processing
./run_inference.sh batch sample_data/models/your_model.pt sample_data/images/

# Performance testing
./run_inference.sh benchmark sample_data/models/your_model.pt --iterations 100
```

## Usage Examples

### Single Image Processing

Process individual images with customizable parameters:

```bash
# Basic inference with default settings
./run_inference.sh simple model.pt image.jpg

# Custom confidence threshold and input size
./run_inference.sh simple model.pt image.jpg 0.3 640

# Results are saved to detections.json
```

Output example:
```
Using CUDA device
Loading model: model.pt
Model loaded successfully
Inference time: 12ms

Found 3 detections:
[0] class_0 conf=0.876 bbox=[145.2,203.1,287.4,356.8]
[1] class_1 conf=0.734 bbox=[401.5,125.7,523.2,289.3]
[2] class_2 conf=0.652 bbox=[789.1,456.2,912.7,578.9]

Results saved to detections.json
Visualization saved to result_1234567890.jpg
```

**Automatic Visualization**: The engines automatically generate annotated images showing bounding boxes overlaid on the input image, making it easy to visually verify detection results.

### Batch Processing

Process multiple images efficiently:

```bash
# Process all images in a directory
./run_inference.sh batch model.pt ./images/ 0.5 results.json

# Process images from a file list
echo "image1.jpg" > image_list.txt
echo "image2.jpg" >> image_list.txt
./run_inference.sh batch model.pt -list image_list.txt 0.5 batch_results.json
```

### Performance Benchmarking

Analyze model performance with detailed statistics:

```bash
# Basic benchmark with synthetic data
./run_inference.sh benchmark model.pt

# Benchmark with real images
./run_inference.sh benchmark model.pt --image test.jpg --iterations 200

# Compare different input sizes
./run_inference.sh benchmark model.pt --compare-sizes --csv results.csv
```

## Output Format

All engines output structured JSON results:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.876,
      "bbox": [145.2, 203.1, 287.4, 356.8]
    }
  ]
}
```

Batch processing includes additional metadata:
```json
{
  "total_images": 10,
  "successful_images": 9,
  "total_detections": 15,
  "average_inference_time_ms": 8.5,
  "results": [...]
}
```

## Model Requirements

The engines work with YOLO TorchScript models exported from PyTorch. Your YOLO model should:

- Be exported as a `.pt` file using `torch.jit.script()` or `torch.jit.trace()`
- Accept input tensors in NCHW format (batch, channels, height, width)
- Output detection results in standard YOLO format (bounding boxes, confidences, classes)

Example YOLO model export:
```python
import torch

# Load your trained YOLO model
model = YourYOLOModel()
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('your_yolo_model.torchscript.pt')
```

## Integration

### Command Line Integration

```bash
#!/bin/bash
# Process workflow pipeline
for image in input_images/*.jpg; do
    ./run_inference.sh simple model.pt "$image" 0.5
    # Process results...
done
```

### Python Integration

```python
import subprocess
import json

def run_inference(model_path, image_path):
    result = subprocess.run([
        "./build/bin/simple_inference", 
        model_path, 
        image_path
    ], capture_output=True, text=True)
    
    with open("detections.json") as f:
        return json.load(f)

detections = run_inference("model.pt", "image.jpg")
print(f"Found {len(detections['detections'])} objects")
```

### C++ Library Usage

The inference engines can be integrated as libraries in larger C++ applications:

```cpp
#include "simple_inference.cpp"

int main() {
    SimpleInference engine;
    engine.loadModel("model.pt");
    engine.setConfidenceThreshold(0.5f);
    
    auto detections = engine.runInference("image.jpg");
    
    std::cout << "Found " << detections.size() << " detections" << std::endl;
    return 0;
}
```

## Configuration & Optimization

### Input Sizes
The engines support various input resolutions for speed/accuracy trade-offs:

- **416×416**: Fastest inference, lower accuracy
- **640×640**: Balanced performance  
- **960×960**: Default, good accuracy/speed ratio
- **1280×1280**: Highest accuracy, slower inference

### GPU Optimization
For best GPU performance:

- Ensure CUDA drivers and toolkit are properly installed
- Use batch processing for multiple images
- Enable model warmup for consistent timing
- Consider FP16 precision for supported models

### Memory Management
The engines implement several optimizations:

- Pinned memory for faster CPU-GPU transfers
- Model optimization with `optimize_for_inference()`
- Automatic memory cleanup and error handling
- Configurable batch sizes to fit available memory

## Troubleshooting

### Common Build Issues

**CMake not found**: Install CMake 3.18+
```bash
sudo apt install cmake  # Ubuntu/Debian
brew install cmake       # macOS
```

**OpenCV missing**: Install development headers
```bash
sudo apt install libopencv-dev pkg-config  # Ubuntu/Debian
brew install opencv                         # macOS
```

**LibTorch download fails**: The build script will retry automatically, or download manually from pytorch.org

### Runtime Issues

**Model loading fails**: Verify the model is a valid TorchScript file exported with compatible PyTorch version

**CUDA errors**: 
- Check GPU compatibility: `nvidia-smi`
- Verify CUDA toolkit installation: `nvcc --version`
- Try CPU-only mode: `./build.sh --cpu-only`

**Poor performance**: 
- Enable GPU acceleration
- Use appropriate input sizes for your hardware
- Ensure model is optimized for inference
- Check system load and available memory

### Debugging

Build in debug mode for detailed error information:
```bash
./build.sh --debug --verbose
```

Enable detailed logging by setting environment variables:
```bash
export TORCH_LOGS=all
./run_inference.sh simple model.pt image.jpg
```

## Performance

Typical performance on modern hardware:

**NVIDIA RTX 4060 (CUDA)**
- Single image: 8-15ms
- Batch processing: 6-12ms per image
- Model loading: 100-500ms (one-time)

**Intel i7-12700K (CPU)**
- Single image: 50-150ms  
- Batch processing: 40-120ms per image
- Model loading: 200-800ms (one-time)

Performance varies significantly based on model complexity, input size, and hardware configuration.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

When contributing:
1. Follow the existing code style (modern C++17)
2. Add tests for new functionality  
3. Update documentation as needed
4. Ensure cross-platform compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Built with:
- [PyTorch C++ (LibTorch)](https://pytorch.org/cppdocs/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [CMake](https://cmake.org/) - Build system

Designed for YOLO object detection workflows and optimized for modern hardware.
