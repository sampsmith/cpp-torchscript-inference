#!/bin/bash

# TorchScript Inference Engines Build Script
# Supports Linux, macOS, and Windows (via WSL/MSYS2)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}TorchScript Inference Engines Build${NC}"
echo -e "${BLUE}=====================================${NC}"

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_DIR=${BUILD_DIR:-build}
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Build type: ${GREEN}$BUILD_TYPE${NC}"
echo -e "  Build directory: ${GREEN}$BUILD_DIR${NC}"
echo -e "  Parallel jobs: ${GREEN}$PARALLEL_JOBS${NC}"
echo

# Parse command line arguments
CLEAN=false
VERBOSE=false
INSTALL=false
CPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --release)
            BUILD_TYPE=Release
            shift
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --clean        Clean build directory before building"
            echo "  --verbose      Enable verbose build output"
            echo "  --install      Install binaries after building"
            echo "  --cpu-only     Build without CUDA support"
            echo "  --debug        Build in Debug mode"
            echo "  --release      Build in Release mode (default)"
            echo "  -j, --jobs N   Use N parallel jobs (default: auto-detect)"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf $BUILD_DIR
fi

# Create build directory
echo -e "${BLUE}Creating build directory...${NC}"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Check for required tools
echo -e "${BLUE}Checking build tools...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake is required but not installed${NC}"
    echo "Please install CMake 3.18 or later"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo -e "  CMake: ${GREEN}$CMAKE_VERSION${NC}"

# Detect compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | cut -d' ' -f4)
    echo -e "  GCC: ${GREEN}$GCC_VERSION${NC}"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1 | cut -d' ' -f4)
    echo -e "  Clang: ${GREEN}$CLANG_VERSION${NC}"
fi

# Check for OpenCV
echo -e "${BLUE}Checking dependencies...${NC}"

if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo -e "${YELLOW}Warning: OpenCV not found via pkg-config${NC}"
    echo -e "${YELLOW}Make sure OpenCV is installed and CMAKE_PREFIX_PATH is set${NC}"
else
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
    else
        OPENCV_VERSION=$(pkg-config --modversion opencv)
    fi
    echo -e "  OpenCV: ${GREEN}$OPENCV_VERSION${NC}"
fi

# Check for CUDA
if [ "$CPU_ONLY" = false ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
        echo -e "  CUDA: ${GREEN}$CUDA_VERSION${NC}"
    else
        echo -e "  CUDA: ${YELLOW}Not found (CPU-only build)${NC}"
    fi
fi

echo

# Configure CMake
# Download LibTorch if not present
if [ ! -d "libtorch" ]; then
    echo -e "${BLUE}Downloading LibTorch...${NC}"
    if [ "$CPU_ONLY" = true ]; then
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        echo -e "  Downloading CPU-only version..."
    else
        if command -v nvcc &> /dev/null; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
            echo -e "  Downloading CUDA version..."
        else
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
            echo -e "  CUDA not found, downloading CPU version..."
        fi
    fi
    
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download LibTorch${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Extracting LibTorch...${NC}"
    unzip -q libtorch.zip
    rm libtorch.zip
    echo -e "${GREEN}LibTorch ready${NC}"
else
    echo -e "${GREEN}LibTorch already available${NC}"
fi

echo -e "${BLUE}Configuring CMake...${NC}"

# Set CMAKE_PREFIX_PATH to include LibTorch
export CMAKE_PREFIX_PATH="$(pwd)/libtorch:$CMAKE_PREFIX_PATH"

CMAKE_ARGS=""
if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

if [ "$CPU_ONLY" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCUDA_TOOLKIT_ROOT_DIR="
fi

# Platform-specific settings
case "$(uname -s)" in
    Linux*)
        echo -e "  Platform: ${GREEN}Linux${NC}"
        ;;
    Darwin*)
        echo -e "  Platform: ${GREEN}macOS${NC}"
        # On macOS, might need to specify paths
        if [ -d "/usr/local/opt/opencv" ]; then
            CMAKE_ARGS="$CMAKE_ARGS -DOpenCV_DIR=/usr/local/opt/opencv/lib/cmake/opencv4"
        fi
        ;;
    CYGWIN*|MINGW*|MSYS*)
        echo -e "  Platform: ${GREEN}Windows${NC}"
        ;;
    *)
        echo -e "  Platform: ${YELLOW}Unknown$(uname -s)${NC}"
        ;;
esac

# Run CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE $CMAKE_ARGS

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration successful!${NC}"
echo

# Build
echo -e "${BLUE}Building...${NC}"
if [ "$VERBOSE" = true ]; then
    cmake --build . --config $BUILD_TYPE --parallel $PARALLEL_JOBS -- VERBOSE=1
else
    cmake --build . --config $BUILD_TYPE --parallel $PARALLEL_JOBS
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"
echo

# Install if requested
if [ "$INSTALL" = true ]; then
    echo -e "${BLUE}Installing...${NC}"
    cmake --install . --config $BUILD_TYPE
    echo -e "${GREEN}Installation complete!${NC}"
    echo
fi

# List built executables
echo -e "${BLUE}Built executables:${NC}"
BIN_DIR="bin"
if [ -d "$BIN_DIR" ]; then
    for exe in "$BIN_DIR"/*; do
        if [ -x "$exe" ]; then
            SIZE=$(du -h "$exe" | cut -f1)
            echo -e "  ${GREEN}$(basename "$exe")${NC} (${SIZE})"
        fi
    done
else
    # Fallback for different CMake generators
    for exe in simple_inference batch_inference benchmark_inference; do
        if [ -f "$exe" ] || [ -f "$exe.exe" ]; then
            if [ -f "$exe.exe" ]; then
                exe="$exe.exe"
            fi
            SIZE=$(du -h "$exe" | cut -f1)
            echo -e "  ${GREEN}$exe${NC} (${SIZE})"
        fi
    done
fi

echo
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}=====================================${NC}"

# Quick test
echo -e "${BLUE}Running quick test...${NC}"
if [ -f "bin/simple_inference" ]; then
    ./bin/simple_inference --help > /dev/null 2>&1 && echo -e "${GREEN}✓ simple_inference executable works${NC}" || echo -e "${RED}✗ simple_inference test failed${NC}"
elif [ -f "simple_inference" ]; then
    ./simple_inference --help > /dev/null 2>&1 && echo -e "${GREEN}✓ simple_inference executable works${NC}" || echo -e "${RED}✗ simple_inference test failed${NC}"
fi

echo
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Copy your TorchScript model (.pt file) to this directory"
echo -e "  2. Run inference with: ${GREEN}./bin/simple_inference model.torchscript.pt image.jpg${NC}"
echo -e "  3. For batch processing: ${GREEN}./bin/batch_inference model.torchscript.pt /path/to/images/${NC}"
echo -e "  4. For benchmarking: ${GREEN}./bin/benchmark_inference model.torchscript.pt${NC}"