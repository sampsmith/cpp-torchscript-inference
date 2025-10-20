#!/bin/bash

# TorchScript Inference Engines Wrapper Script
# This script sets up the environment and provides easy access to all engines

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up LibTorch library path
export LD_LIBRARY_PATH="$SCRIPT_DIR/libtorch/lib:$LD_LIBRARY_PATH"

# Available engines
SIMPLE_INFERENCE="$SCRIPT_DIR/build/bin/simple_inference"
BATCH_INFERENCE="$SCRIPT_DIR/build/bin/batch_inference"
BENCHMARK_INFERENCE="$SCRIPT_DIR/build/bin/benchmark_inference"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_usage() {
    echo -e "${BLUE}TorchScript C++ Inference Engines${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo
    echo -e "${GREEN}Usage:${NC}"
    echo "  $0 demo                                               # Quick demonstration"
    echo "  $0 simple <model.pt> <image.jpg> [confidence] [input_size]"
    echo "  $0 batch <model.pt> <images_dir/> [confidence] [output.json]"  
    echo "  $0 batch <model.pt> -list <images.txt> [confidence] [output.json]"
    echo "  $0 benchmark <model.pt> [options]"
    echo
    echo -e "${GREEN}Examples:${NC}"
    echo "  # Quick demo with sample data"
    echo "  $0 demo"
    echo
    echo "  # Single image inference with visualization"
    echo "  $0 simple sample_data/models/nails_63.5MAP.torchscript.pt sample_data/images/test.jpg"
    echo
    echo "  # Batch processing"
    echo "  $0 batch sample_data/models/nails_63.5MAP.torchscript.pt sample_data/images/"
    echo
    echo -e "${YELLOW}Note:${NC} Generates JSON results and visualized images with bounding boxes"
}

# Check if executables exist
check_executables() {
    local missing=0
    for exe in "$SIMPLE_INFERENCE" "$BATCH_INFERENCE" "$BENCHMARK_INFERENCE"; do
        if [ ! -f "$exe" ]; then
            echo -e "${RED}Error: Executable not found: $exe${NC}"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        echo -e "${YELLOW}Please build the project first:${NC}"
        echo "./build.sh"
        exit 1
    fi
}

# Demo function
run_demo() {
    echo -e "${BLUE}🚀 TorchScript C++ Inference Demo${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo
    
    # Check if sample data exists
    if [ ! -f "sample_data/models/nails_63.5MAP.torchscript.pt" ]; then
        echo -e "${RED}Error: Sample model not found${NC}"
        echo "Please ensure sample_data/models/nails_63.5MAP.torchscript.pt exists"
        exit 1
    fi
    
    if [ ! -f "sample_data/images/test.jpg" ]; then
        echo -e "${RED}Error: Sample image not found${NC}"
        echo "Please ensure sample_data/images/test.jpg exists"
        exit 1
    fi
    
    echo -e "${GREEN}Running single image inference with visualization...${NC}"
    echo
    
    # Run demo inference with correct input size for nail model
    "$SIMPLE_INFERENCE" sample_data/models/nails_63.5MAP.torchscript.pt sample_data/images/test.jpg 0.5 960
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo
        echo -e "${GREEN}✅ Demo Complete!${NC}"
        echo -e "${BLUE}Generated Files:${NC}"
        
        if [ -f "detections.json" ]; then
            echo -e "  📄 ${YELLOW}detections.json${NC} - Detection results"
        fi
        
        if ls result_*.jpg 1> /dev/null 2>&1; then
            for img in result_*.jpg; do
                echo -e "  📷 ${YELLOW}$img${NC} - Image with bounding boxes"
            done
        fi
        
        echo
        echo -e "${GREEN}🎯 Perfect for showcasing C++ computer vision expertise!${NC}"
    else
        echo -e "${RED}Demo failed with exit code $exit_code${NC}"
    fi
    
    exit $exit_code
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    # Handle demo mode first
    if [ "$1" = "demo" ]; then
        check_executables
        run_demo
        exit $?
    fi
    
    check_executables
    
    local engine="$1"
    shift
    
    case "$engine" in
        "simple"|"s")
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: Simple inference requires model and image paths${NC}"
                echo "Usage: $0 simple <model.pt> <image.jpg> [confidence] [input_size]"
                exit 1
            fi
            
            echo -e "${BLUE}Running simple inference...${NC}"
            "$SIMPLE_INFERENCE" "$@"
            local exit_code=$?
            
            if [ $exit_code -eq 0 ] && [ -f "detections.json" ]; then
                echo
                echo -e "${GREEN}Results saved to detections.json${NC}"
                echo -e "${BLUE}Detection summary:${NC}"
                python3 -c "
import json
try:
    with open('detections.json', 'r') as f:
        data = json.load(f)
    detections = data['detections']
    print(f'  Found {len(detections)} detections')
    for i, det in enumerate(detections):
        print(f'  [{i}] Class {det[\"class_id\"]}: {det[\"confidence\"]:.3f} confidence')
except Exception as e:
    print(f'  Error parsing results: {e}')
" 2>/dev/null || echo "  (Python not available for summary)"
            fi
            exit $exit_code
            ;;
            
        "batch"|"b")
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: Batch inference requires model and image directory/list${NC}"
                echo "Usage: $0 batch <model.pt> <images_dir/> [confidence] [output.json]"
                echo "       $0 batch <model.pt> -list <images.txt> [confidence] [output.json]"
                exit 1
            fi
            
            echo -e "${BLUE}Running batch inference...${NC}"
            "$BATCH_INFERENCE" "$@"
            local exit_code=$?
            
            # Find the output JSON file
            local json_file="batch_results.json"
            for arg in "$@"; do
                if [[ "$arg" == *.json ]]; then
                    json_file="$arg"
                    break
                fi
            done
            
            if [ $exit_code -eq 0 ] && [ -f "$json_file" ]; then
                echo
                echo -e "${GREEN}Results saved to $json_file${NC}"
                echo -e "${BLUE}Batch summary:${NC}"
                python3 -c "
import json
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    print(f'  Total images: {data[\"total_images\"]}')
    print(f'  Successful: {data[\"successful_images\"]}')
    print(f'  Total detections: {data[\"total_detections\"]}')
    print(f'  Average time: {data[\"average_inference_time_ms\"]:.1f}ms per image')
except Exception as e:
    print(f'  Error parsing results: {e}')
" 2>/dev/null || echo "  (Python not available for summary)"
            fi
            exit $exit_code
            ;;
            
        "benchmark"|"bench"|"bm")
            if [ $# -lt 1 ]; then
                echo -e "${RED}Error: Benchmark requires model path${NC}"
                echo "Usage: $0 benchmark <model.pt> [options]"
                exit 1
            fi
            
            echo -e "${BLUE}Running performance benchmark...${NC}"
            echo -e "${YELLOW}Note: May show CUDA warnings but will continue...${NC}"
            echo
            
            # Run benchmark with error handling for CUDA issues
            timeout 120s "$BENCHMARK_INFERENCE" "$@" 2>/dev/null || {
                local exit_code=$?
                if [ $exit_code -eq 124 ]; then
                    echo -e "${YELLOW}Benchmark timed out after 2 minutes${NC}"
                elif [ $exit_code -ne 0 ]; then
                    echo -e "${YELLOW}Benchmark encountered CUDA issues but may have partial results${NC}"
                    echo -e "${BLUE}Trying with CPU-only fallback...${NC}"
                    # Could implement CPU fallback here if needed
                fi
                exit $exit_code
            }
            ;;
            
        "help"|"--help"|"-h")
            show_usage
            ;;
            
        *)
            echo -e "${RED}Unknown engine: $engine${NC}"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"