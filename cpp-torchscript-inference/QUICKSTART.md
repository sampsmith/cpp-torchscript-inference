# Quick Start Guide

Get up and running with YOLO TorchScript C++ inference engines in minutes. This guide covers everything you need for your first successful YOLO inference.

## Prerequisites Check

Before building, ensure you have the essential tools:

```bash
# Check your system
cmake --version    # Need 3.18+
g++ --version      # Need C++17 support
pkg-config --exists opencv4 && echo "OpenCV found" || echo "OpenCV missing"
```

If anything is missing:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake build-essential libopencv-dev pkg-config

# macOS
brew install cmake opencv pkg-config
```

For GPU acceleration, install the CUDA toolkit:
```bash
# Ubuntu/Debian  
sudo apt install nvidia-cuda-toolkit

# Check installation
nvidia-smi && nvcc --version
```

## Build in 3 Steps

1. **Clone and enter directory**
   ```bash
   git clone https://github.com/sampsmith/cpp-torchscript-inference.git
   cd cpp-torchscript-inference
   ```

2. **Build everything**
   ```bash
   chmod +x build.sh
   ./build.sh
   ```
   
   The build script automatically downloads LibTorch and sets up the environment. This may take 5-10 minutes on first run.

3. **Verify the build**
   ```bash
   ./run_inference.sh help
   ```

You should see the help menu with available engines.

## Your First Inference

### Option 1: Use the provided sample
```bash
# Single image inference
./run_inference.sh simple sample_data/models/your_yolo_model.pt sample_data/images/test.jpg
```

### Option 2: Use your own YOLO model
```bash
# Put your YOLO TorchScript model in the project directory
./run_inference.sh simple your_yolo_model.pt your_image.jpg
```

### Expected Output
```
Running simple inference...
Using CUDA device
Loading model: your_yolo_model.pt  
Model loaded successfully
Inference time: 15ms

Found 2 detections:
[0] class_0 conf=0.876 bbox=[145.2,203.1,287.4,356.8]
[1] class_1 conf=0.734 bbox=[401.5,125.7,523.2,289.3]

Results saved to detections.json
```

## Try Batch Processing

Process multiple images at once:

```bash
# Create a test directory with images
mkdir my_test_images
cp your_image1.jpg your_image2.jpg my_test_images/

# Run batch inference
./run_inference.sh batch your_yolo_model.pt my_test_images/ 0.5 my_results.json
```

## Performance Testing

Check how fast your model runs:

```bash
# Basic performance test
./run_inference.sh benchmark your_yolo_model.pt --iterations 50

# Test with real image data
./run_inference.sh benchmark your_yolo_model.pt --image your_image.jpg
```

## Common Usage Patterns

### Production Pipeline
```bash
#!/bin/bash
for image in production_images/*.jpg; do
    ./run_inference.sh simple yolo_model.pt "$image" 0.5
    # Process the results from detections.json
done
```

### Different Confidence Thresholds
```bash
# Conservative (fewer false positives)
./run_inference.sh simple yolo_model.pt image.jpg 0.7

# Sensitive (catch more objects)  
./run_inference.sh simple yolo_model.pt image.jpg 0.3
```

### Different Input Sizes
```bash
# Fast inference
./run_inference.sh simple yolo_model.pt image.jpg 0.5 416

# High accuracy
./run_inference.sh simple yolo_model.pt image.jpg 0.5 1280
```

## Troubleshooting

### Build Issues

**"CMake not found"**
```bash
sudo apt install cmake
```

**"OpenCV not found"**  
```bash
sudo apt install libopencv-dev pkg-config
```

**"Build failed"**
```bash
# Try clean rebuild
./build.sh --clean --verbose
```

### Runtime Issues

**"Model loading failed"**
- Check that your `.pt` file is a valid YOLO TorchScript model
- Ensure the file path is correct

**"CUDA errors"**  
- Verify GPU setup: `nvidia-smi`
- Try CPU-only build: `./build.sh --cpu-only`

**"No detections found"**
- Try lowering confidence threshold: `0.3` instead of `0.5`
- Check if your model expects different input sizes
- Verify your image loads correctly with standard tools

### Getting Help

**Check logs**
```bash
# Run with verbose output
./run_inference.sh simple yolo_model.pt image.jpg 2>&1 | tee debug.log
```

**Test with provided sample**
```bash
# This should always work if the build succeeded
./run_inference.sh simple sample_data/models/your_yolo_model.pt sample_data/images/test.jpg
```

## Integration Examples

### Python Script
```python
import subprocess
import json

result = subprocess.run([
    "./run_inference.sh", "simple", 
    "yolo_model.pt", "image.jpg"
], capture_output=True, text=True)

with open("detections.json") as f:
    detections = json.load(f)
    print(f"Found {len(detections['detections'])} objects")
```

### Bash Pipeline
```bash
# Process and filter results
./run_inference.sh simple yolo_model.pt image.jpg
cat detections.json | jq '.detections[] | select(.confidence > 0.8)'
```

## Next Steps

Once you have basic inference working:

1. **Read the full [README.md](README.md)** for comprehensive documentation
2. **Explore [examples/](examples/)** for integration patterns
3. **Optimize performance** by trying different input sizes and batch processing
4. **Integrate into your workflow** using the JSON output format

## Performance Expectations

On modern hardware, expect:

- **Model loading**: 200-800ms (one-time cost)
- **Single inference**: 8-50ms depending on hardware and model
- **Batch processing**: 5-10% faster per image than single processing
- **GPU vs CPU**: 3-10x speedup with proper CUDA setup

Your actual performance will vary based on model complexity, input resolution, and hardware specifications.

---

**Ready to run inference!** For detailed documentation, integration examples, and advanced configuration, see the main [README.md](README.md).