cmake_minimum_required(VERSION 3.18)
project(torchscript_inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# CUDA support
if(TORCH_CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found - GPU acceleration enabled")
else()
    message(STATUS "CUDA not found - CPU only")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Common source files (if any shared utilities are added)
set(COMMON_SOURCES
    # Add any shared utility files here
)

# Simple inference executable
add_executable(simple_inference
    simple_inference.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(simple_inference 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# Batch inference executable
add_executable(batch_inference
    batch_inference.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(batch_inference 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# Benchmark inference executable
add_executable(benchmark_inference
    benchmark_inference.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(benchmark_inference 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(simple_inference PRIVATE WIN32_LEAN_AND_MEAN)
    target_compile_definitions(batch_inference PRIVATE WIN32_LEAN_AND_MEAN)
    target_compile_definitions(benchmark_inference PRIVATE WIN32_LEAN_AND_MEAN)
    
    # Copy DLLs for Windows
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(TORCH_DLLS ${TORCH_INSTALL_PREFIX}/lib/*.dll)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*_d.dll")
    else()
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    endif()
    
    foreach(DLL ${TORCH_DLLS})
        add_custom_command(TARGET simple_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}" $<TARGET_FILE_DIR:simple_inference>)
    endforeach()
endif()

# macOS specific settings
if(APPLE)
    set_target_properties(simple_inference PROPERTIES
        MACOSX_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
    set_target_properties(batch_inference PROPERTIES
        MACOSX_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
    set_target_properties(benchmark_inference PROPERTIES
        MACOSX_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
endif()

# Linux specific settings
if(UNIX AND NOT APPLE)
    set_target_properties(simple_inference PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN"
    )
    set_target_properties(batch_inference PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN"
    )
    set_target_properties(benchmark_inference PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Installation rules
install(TARGETS simple_inference batch_inference benchmark_inference
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# Optional build info removed for simplicity
