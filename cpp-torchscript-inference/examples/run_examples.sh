#!/bin/bash

# TorchScript Inference Engines - Example Usage Scripts
# This script demonstrates various ways to use the inference engines

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TorchScript Inference Engines - Examples${NC}"
echo -e "${BLUE}============================================${NC}"
echo

# Check if executables exist
BIN_DIR="../build/bin"
if [ ! -d "$BIN_DIR" ]; then
    BIN_DIR="../build"
fi

SIMPLE_INFERENCE="$BIN_DIR/simple_inference"
BATCH_INFERENCE="$BIN_DIR/batch_inference"
BENCHMARK_INFERENCE="$BIN_DIR/benchmark_inference"

for exe in "$SIMPLE_INFERENCE" "$BATCH_INFERENCE" "$BENCHMARK_INFERENCE"; do
    if [ ! -f "$exe" ]; then
        echo -e "${RED}Error: Executable not found: $exe${NC}"
        echo -e "${YELLOW}Please build the project first:${NC}"
        echo "  cd .. && ./build.sh"
        exit 1
    fi
done

# Configuration
MODEL_PATH="${MODEL_PATH:-../models/example_model.pt}"
TEST_IMAGE="${TEST_IMAGE:-test_image.jpg}"
TEST_DIR="${TEST_DIR:-test_images}"

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Model path: ${GREEN}$MODEL_PATH${NC}"
echo -e "  Test image: ${GREEN}$TEST_IMAGE${NC}"
echo -e "  Test directory: ${GREEN}$TEST_DIR${NC}"
echo

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model file not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Using placeholder - some examples may fail${NC}"
    echo
fi

# Create test data if it doesn't exist
create_test_data() {
    echo -e "${BLUE}Creating test data...${NC}"
    
    # Create test image if it doesn't exist
    if [ ! -f "$TEST_IMAGE" ]; then
        echo -e "${YELLOW}Creating placeholder test image...${NC}"
        # Create a simple test image using ImageMagick or Python
        if command -v convert &> /dev/null; then
            convert -size 640x480 xc:skyblue -fill white -gravity center -pointsize 72 -annotate +0+0 "TEST" "$TEST_IMAGE"
        elif command -v python3 &> /dev/null; then
            python3 -c "
import cv2
import numpy as np
img = np.full((480, 640, 3), (135, 206, 235), dtype=np.uint8)
cv2.putText(img, 'TEST IMAGE', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.imwrite('$TEST_IMAGE', img)
print('Created test image')
"
        else
            echo -e "${YELLOW}Cannot create test image - please provide $TEST_IMAGE${NC}"
        fi
    fi
    
    # Create test directory with multiple images
    if [ ! -d "$TEST_DIR" ]; then
        mkdir -p "$TEST_DIR"
        echo -e "${YELLOW}Creating test images directory...${NC}"
        
        # Copy test image multiple times with different names
        for i in {1..5}; do
            if [ -f "$TEST_IMAGE" ]; then
                cp "$TEST_IMAGE" "$TEST_DIR/test_$(printf "%03d" $i).jpg"
            fi
        done
        
        echo "Created $TEST_DIR with test images"
    fi
}

# Example 1: Simple single image inference
example_simple_inference() {
    echo -e "${GREEN}=== Example 1: Simple Inference ===${NC}"
    echo -e "${BLUE}Processing single image with default settings${NC}"
    echo
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$SIMPLE_INFERENCE $MODEL_PATH $TEST_IMAGE"
    echo
    
    if [ -f "$MODEL_PATH" ] && [ -f "$TEST_IMAGE" ]; then
        $SIMPLE_INFERENCE "$MODEL_PATH" "$TEST_IMAGE"
        echo
        
        if [ -f "detections.json" ]; then
            echo -e "${GREEN}✓ Results saved to detections.json${NC}"
            echo -e "${BLUE}First few lines:${NC}"
            head -n 10 detections.json
        fi
    else
        echo -e "${RED}Skipping - missing model or image file${NC}"
    fi
    
    echo
}

# Example 2: Custom confidence threshold and input size
example_custom_parameters() {
    echo -e "${GREEN}=== Example 2: Custom Parameters ===${NC}"
    echo -e "${BLUE}Using custom confidence threshold and input size${NC}"
    echo
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$SIMPLE_INFERENCE $MODEL_PATH $TEST_IMAGE 0.3 416"
    echo
    
    if [ -f "$MODEL_PATH" ] && [ -f "$TEST_IMAGE" ]; then
        $SIMPLE_INFERENCE "$MODEL_PATH" "$TEST_IMAGE" 0.3 416
        echo
    else
        echo -e "${RED}Skipping - missing model or image file${NC}"
    fi
    
    echo
}

# Example 3: Batch processing directory
example_batch_directory() {
    echo -e "${GREEN}=== Example 3: Batch Processing (Directory) ===${NC}"
    echo -e "${BLUE}Processing all images in directory${NC}"
    echo
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$BATCH_INFERENCE $MODEL_PATH $TEST_DIR 0.5 batch_results.json"
    echo
    
    if [ -f "$MODEL_PATH" ] && [ -d "$TEST_DIR" ]; then
        $BATCH_INFERENCE "$MODEL_PATH" "$TEST_DIR" 0.5 batch_results.json
        echo
        
        if [ -f "batch_results.json" ]; then
            echo -e "${GREEN}✓ Batch results saved to batch_results.json${NC}"
            echo -e "${BLUE}Summary:${NC}"
            python3 -c "
import json
try:
    with open('batch_results.json', 'r') as f:
        data = json.load(f)
    print(f'  Total images: {data[\"total_images\"]}')
    print(f'  Successful: {data[\"successful_images\"]}')
    print(f'  Total detections: {data[\"total_detections\"]}')
    print(f'  Average time: {data[\"average_inference_time_ms\"]:.1f}ms')
except:
    print('Could not parse results')
"
        fi
    else
        echo -e "${RED}Skipping - missing model or test directory${NC}"
    fi
    
    echo
}

# Example 4: Batch processing from file list
example_batch_list() {
    echo -e "${GREEN}=== Example 4: Batch Processing (File List) ===${NC}"
    echo -e "${BLUE}Processing images from file list${NC}"
    echo
    
    # Create image list file
    IMAGE_LIST="image_list.txt"
    if [ -d "$TEST_DIR" ]; then
        find "$TEST_DIR" -name "*.jpg" | head -3 > "$IMAGE_LIST"
        echo -e "${BLUE}Created image list:${NC}"
        cat "$IMAGE_LIST"
        echo
    else
        echo -e "${YELLOW}Creating simple image list...${NC}"
        echo "$TEST_IMAGE" > "$IMAGE_LIST"
    fi
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$BATCH_INFERENCE $MODEL_PATH -list $IMAGE_LIST 0.5 list_results.json"
    echo
    
    if [ -f "$MODEL_PATH" ] && [ -f "$IMAGE_LIST" ]; then
        $BATCH_INFERENCE "$MODEL_PATH" -list "$IMAGE_LIST" 0.5 list_results.json
        echo
        
        if [ -f "list_results.json" ]; then
            echo -e "${GREEN}✓ List results saved to list_results.json${NC}"
        fi
    else
        echo -e "${RED}Skipping - missing model or image list${NC}"
    fi
    
    echo
}

# Example 5: Basic benchmarking
example_basic_benchmark() {
    echo -e "${GREEN}=== Example 5: Basic Benchmarking ===${NC}"
    echo -e "${BLUE}Performance testing with synthetic data${NC}"
    echo
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$BENCHMARK_INFERENCE $MODEL_PATH --warmup 10 --iterations 20"
    echo
    
    if [ -f "$MODEL_PATH" ]; then
        $BENCHMARK_INFERENCE "$MODEL_PATH" --warmup 10 --iterations 20
        echo
    else
        echo -e "${RED}Skipping - missing model file${NC}"
    fi
    
    echo
}

# Example 6: Benchmarking with real image
example_image_benchmark() {
    echo -e "${GREEN}=== Example 6: Benchmarking with Real Image ===${NC}"
    echo -e "${BLUE}Performance testing with real image data${NC}"
    echo
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$BENCHMARK_INFERENCE $MODEL_PATH --image $TEST_IMAGE --iterations 30 --csv benchmark_results.csv"
    echo
    
    if [ -f "$MODEL_PATH" ] && [ -f "$TEST_IMAGE" ]; then
        $BENCHMARK_INFERENCE "$MODEL_PATH" --image "$TEST_IMAGE" --iterations 30 --csv benchmark_results.csv
        echo
        
        if [ -f "benchmark_results.csv" ]; then
            echo -e "${GREEN}✓ Detailed results saved to benchmark_results.csv${NC}"
            echo -e "${BLUE}First few measurements:${NC}"
            head -n 6 benchmark_results.csv
        fi
    else
        echo -e "${RED}Skipping - missing model or image file${NC}"
    fi
    
    echo
}

# Example 7: Input size comparison
example_size_comparison() {
    echo -e "${GREEN}=== Example 7: Input Size Comparison ===${NC}"
    echo -e "${BLUE}Comparing performance across different input sizes${NC}"
    echo
    
    echo -e "${YELLOW}Command:${NC}"
    echo "$BENCHMARK_INFERENCE $MODEL_PATH --compare-sizes --csv size_comparison.csv"
    echo
    
    if [ -f "$MODEL_PATH" ]; then
        $BENCHMARK_INFERENCE "$MODEL_PATH" --compare-sizes --csv size_comparison.csv
        echo
        
        if [ -f "size_comparison.csv" ]; then
            echo -e "${GREEN}✓ Size comparison saved to size_comparison.csv${NC}"
        fi
    else
        echo -e "${RED}Skipping - missing model file${NC}"
    fi
    
    echo
}

# Production workflow example
example_production_workflow() {
    echo -e "${GREEN}=== Example 8: Production Workflow ===${NC}"
    echo -e "${BLUE}Simulating a production image processing pipeline${NC}"
    echo
    
    # Create workflow script
    cat > production_workflow.sh << 'EOF'
#!/bin/bash

# Production Image Processing Workflow
MODEL_PATH="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"

echo "Starting production workflow..."
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process images in batches
BATCH_SIZE=10
BATCH_NUM=0

for batch_dir in $(find "$INPUT_DIR" -name "*.jpg" | head -n $BATCH_SIZE | xargs -n $BATCH_SIZE dirname | sort -u); do
    BATCH_NUM=$((BATCH_NUM + 1))
    echo "Processing batch $BATCH_NUM..."
    
    # Run batch inference
    ./batch_inference "$MODEL_PATH" "$batch_dir" 0.5 "$OUTPUT_DIR/batch_${BATCH_NUM}_results.json"
    
    # Process results (example: count detections)
    python3 -c "
import json
try:
    with open('$OUTPUT_DIR/batch_${BATCH_NUM}_results.json', 'r') as f:
        data = json.load(f)
    total_detections = data['total_detections']
    print(f'Batch {$BATCH_NUM}: {total_detections} total detections')
except Exception as e:
    print(f'Error processing batch {$BATCH_NUM}: {e}')
"
done

echo "Production workflow complete!"
EOF
    
    chmod +x production_workflow.sh
    
    echo -e "${BLUE}Created production_workflow.sh${NC}"
    echo -e "${YELLOW}Usage:${NC}"
    echo "./production_workflow.sh MODEL_PATH INPUT_DIR OUTPUT_DIR"
    echo
    
    if [ -f "$MODEL_PATH" ] && [ -d "$TEST_DIR" ]; then
        echo -e "${BLUE}Running example workflow:${NC}"
        ./production_workflow.sh "$MODEL_PATH" "$TEST_DIR" "output_results"
        echo
    else
        echo -e "${YELLOW}Workflow script created but not executed (missing model or test data)${NC}"
    fi
    
    echo
}

# Integration examples
show_integration_examples() {
    echo -e "${GREEN}=== Integration Examples ===${NC}"
    echo
    
    # Python integration example
    cat > python_integration_example.py << 'EOF'
#!/usr/bin/env python3
"""
Python integration example for TorchScript inference engines
"""

import subprocess
import json
import sys
from pathlib import Path

class TorchScriptInference:
    def __init__(self, model_path, executable_path="./simple_inference"):
        self.model_path = model_path
        self.executable_path = executable_path
    
    def process_single_image(self, image_path, confidence=0.5, input_size=960):
        """Process single image and return detections"""
        cmd = [
            self.executable_path,
            self.model_path,
            image_path,
            str(confidence),
            str(input_size)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read results from JSON file
            with open("detections.json", "r") as f:
                detections = json.load(f)
            
            return {
                "success": True,
                "detections": detections["detections"],
                "stdout": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": e.stderr,
                "stdout": e.stdout
            }
    
    def process_batch(self, image_directory, confidence=0.5, output_file="batch_results.json"):
        """Process batch of images"""
        cmd = [
            self.executable_path.replace("simple_inference", "batch_inference"),
            self.model_path,
            image_directory,
            str(confidence),
            output_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            with open(output_file, "r") as f:
                batch_results = json.load(f)
            
            return {
                "success": True,
                "results": batch_results,
                "stdout": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": e.stderr,
                "stdout": e.stdout
            }

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 python_integration_example.py MODEL_PATH IMAGE_PATH")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Initialize inference engine
    engine = TorchScriptInference(model_path)
    
    # Process single image
    print("Processing single image...")
    result = engine.process_single_image(image_path)
    
    if result["success"]:
        print(f"Found {len(result['detections'])} detections")
        for i, det in enumerate(result["detections"]):
            print(f"  [{i}] Class {det['class_id']}: {det['confidence']:.3f}")
    else:
        print(f"Error: {result['error']}")
EOF
    
    chmod +x python_integration_example.py
    
    echo -e "${BLUE}Created Python integration example:${NC}"
    echo -e "${YELLOW}python_integration_example.py${NC}"
    echo
    
    # C++ integration example
    cat > cpp_integration_example.cpp << 'EOF'
/*
C++ integration example for TorchScript inference engines
Compile with: g++ -std=c++17 cpp_integration_example.cpp -o cpp_integration_example
*/

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>

class TorchScriptWrapper {
private:
    std::string model_path_;
    std::string executable_path_;
    
public:
    TorchScriptWrapper(const std::string& model_path, const std::string& executable_path = "./simple_inference")
        : model_path_(model_path), executable_path_(executable_path) {}
    
    bool processImage(const std::string& image_path, float confidence = 0.5f) {
        // Build command
        std::string command = executable_path_ + " " + model_path_ + " " + image_path + " " + std::to_string(confidence);
        
        // Execute command
        int result = system(command.c_str());
        
        if (result == 0) {
            // Read results from JSON file
            std::ifstream json_file("detections.json");
            if (json_file.is_open()) {
                std::string line, json_content;
                while (std::getline(json_file, line)) {
                    json_content += line + "\n";
                }
                json_file.close();
                
                std::cout << "Detection results:\n" << json_content << std::endl;
                return true;
            }
        }
        
        return false;
    }
    
    bool benchmark(int iterations = 100) {
        std::string command = executable_path_;
        command.replace(command.find("simple_inference"), 16, "benchmark_inference");
        command += " " + model_path_ + " --iterations " + std::to_string(iterations);
        
        return system(command.c_str()) == 0;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " MODEL_PATH IMAGE_PATH" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    
    TorchScriptWrapper wrapper(model_path);
    
    std::cout << "Processing image: " << image_path << std::endl;
    if (wrapper.processImage(image_path)) {
        std::cout << "Processing successful!" << std::endl;
    } else {
        std::cout << "Processing failed!" << std::endl;
        return 1;
    }
    
    return 0;
}
EOF
    
    echo -e "${BLUE}Created C++ integration example:${NC}"
    echo -e "${YELLOW}cpp_integration_example.cpp${NC}"
    echo
    
    # Shell script integration
    cat > shell_integration_example.sh << 'EOF'
#!/bin/bash

# Shell script integration example
# Process multiple images with error handling and logging

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
INPUT_DIR="$2"
OUTPUT_DIR="${3:-results}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 MODEL_PATH INPUT_DIR [OUTPUT_DIR]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/processing.log"
echo "Starting batch processing: $(date)" > "$LOG_FILE"

# Counters
TOTAL=0
SUCCESS=0
FAILED=0

# Process each image
for image in "$INPUT_DIR"/*.{jpg,jpeg,png,bmp}; do
    [ -f "$image" ] || continue
    
    TOTAL=$((TOTAL + 1))
    filename=$(basename "$image")
    name="${filename%.*}"
    
    echo "Processing: $filename"
    echo "Processing: $filename" >> "$LOG_FILE"
    
    # Run inference
    if "$SCRIPT_DIR/simple_inference" "$MODEL_PATH" "$image" 0.5 > "$OUTPUT_DIR/${name}_output.log" 2>&1; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✓ Success" | tee -a "$LOG_FILE"
        
        # Move results to output directory
        if [ -f "detections.json" ]; then
            mv "detections.json" "$OUTPUT_DIR/${name}_detections.json"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed" | tee -a "$LOG_FILE"
    fi
done

# Summary
echo ""
echo "Processing Summary:"
echo "  Total images: $TOTAL"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"

echo "Summary: Total=$TOTAL, Success=$SUCCESS, Failed=$FAILED" >> "$LOG_FILE"
echo "Completed: $(date)" >> "$LOG_FILE"

echo "Results saved to: $OUTPUT_DIR"
EOF
    
    chmod +x shell_integration_example.sh
    
    echo -e "${BLUE}Created Shell integration example:${NC}"
    echo -e "${YELLOW}shell_integration_example.sh${NC}"
    echo
}

# Main execution
main() {
    echo -e "${BLUE}Select examples to run:${NC}"
    echo "1) Create test data"
    echo "2) Simple inference"
    echo "3) Custom parameters"
    echo "4) Batch processing (directory)"
    echo "5) Batch processing (file list)"
    echo "6) Basic benchmarking"
    echo "7) Benchmarking with image"
    echo "8) Input size comparison"
    echo "9) Production workflow"
    echo "10) Show integration examples"
    echo "11) Run all examples"
    echo "0) Exit"
    echo
    
    read -p "Enter choice (0-11): " choice
    
    case $choice in
        1)
            create_test_data
            ;;
        2)
            create_test_data
            example_simple_inference
            ;;
        3)
            create_test_data
            example_custom_parameters
            ;;
        4)
            create_test_data
            example_batch_directory
            ;;
        5)
            create_test_data
            example_batch_list
            ;;
        6)
            example_basic_benchmark
            ;;
        7)
            create_test_data
            example_image_benchmark
            ;;
        8)
            example_size_comparison
            ;;
        9)
            create_test_data
            example_production_workflow
            ;;
        10)
            show_integration_examples
            ;;
        11)
            create_test_data
            example_simple_inference
            example_custom_parameters
            example_batch_directory
            example_batch_list
            example_basic_benchmark
            example_image_benchmark
            example_size_comparison
            example_production_workflow
            show_integration_examples
            ;;
        0)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
}

# Run main function
main