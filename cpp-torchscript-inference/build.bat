@echo off
REM TorchScript Inference Engines Build Script for Windows
REM Requires Visual Studio Build Tools and vcpkg or manually installed dependencies

setlocal EnableDelayedExpansion

echo =====================================
echo TorchScript Inference Engines Build
echo =====================================

REM Configuration
set BUILD_TYPE=Release
set BUILD_DIR=build
set PARALLEL_JOBS=%NUMBER_OF_PROCESSORS%
set CLEAN=false
set VERBOSE=false
set INSTALL=false
set CPU_ONLY=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--clean" (
    set CLEAN=true
    shift & goto :parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=true
    shift & goto :parse_args
)
if "%~1"=="--install" (
    set INSTALL=true
    shift & goto :parse_args
)
if "%~1"=="--cpu-only" (
    set CPU_ONLY=true
    shift & goto :parse_args
)
if "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift & goto :parse_args
)
if "%~1"=="--release" (
    set BUILD_TYPE=Release
    shift & goto :parse_args
)
if "%~1"=="-j" (
    set PARALLEL_JOBS=%~2
    shift & shift & goto :parse_args
)
if "%~1"=="--jobs" (
    set PARALLEL_JOBS=%~2
    shift & shift & goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help

echo Unknown option: %~1
goto :error

:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --clean        Clean build directory before building
echo   --verbose      Enable verbose build output
echo   --install      Install binaries after building
echo   --cpu-only     Build without CUDA support
echo   --debug        Build in Debug mode
echo   --release      Build in Release mode (default)
echo   -j, --jobs N   Use N parallel jobs (default: auto-detect)
echo   --help, -h     Show this help message
exit /b 0

:end_parse

echo Configuration:
echo   Build type: %BUILD_TYPE%
echo   Build directory: %BUILD_DIR%
echo   Parallel jobs: %PARALLEL_JOBS%
echo.

REM Clean if requested
if "%CLEAN%"=="true" (
    echo Cleaning build directory...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
)

REM Create build directory
echo Creating build directory...
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

REM Check for required tools
echo Checking build tools...

where cmake >nul 2>nul
if errorlevel 1 (
    echo Error: CMake is required but not installed
    echo Please install CMake 3.18 or later
    goto :error
)

for /f "tokens=3" %%i in ('cmake --version ^| findstr /r "cmake version"') do set CMAKE_VERSION=%%i
echo   CMake: %CMAKE_VERSION%

REM Detect Visual Studio
if defined VS170COMNTOOLS (
    echo   Visual Studio: 2022
) else if defined VS160COMNTOOLS (
    echo   Visual Studio: 2019
) else if defined VS150COMNTOOLS (
    echo   Visual Studio: 2017
) else (
    echo Warning: Visual Studio not detected
    echo Make sure Visual Studio Build Tools are installed
)

REM Check for CUDA
echo Checking dependencies...
if "%CPU_ONLY%"=="false" (
    where nvcc >nul 2>nul
    if not errorlevel 1 (
        for /f "tokens=6 delims=, " %%i in ('nvcc --version ^| findstr /r "release"') do set CUDA_VERSION=%%i
        echo   CUDA: !CUDA_VERSION!
    ) else (
        echo   CUDA: Not found (CPU-only build)
    )
)

echo.

REM Configure CMake
echo Configuring CMake...

set CMAKE_ARGS=-G "Visual Studio 17 2022" -A x64

if "%VERBOSE%"=="true" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_VERBOSE_MAKEFILE=ON
)

if "%CPU_ONLY%"=="true" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCUDA_TOOLKIT_ROOT_DIR=""
)

REM Look for vcpkg
if exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo   Using vcpkg toolchain
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
)

REM Run CMake configuration
echo Platform: Windows
cmake .. -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %CMAKE_ARGS%

if errorlevel 1 (
    echo CMake configuration failed!
    goto :error
)

echo Configuration successful!
echo.

REM Build
echo Building...
if "%VERBOSE%"=="true" (
    cmake --build . --config %BUILD_TYPE% --parallel %PARALLEL_JOBS% -- /v:detailed
) else (
    cmake --build . --config %BUILD_TYPE% --parallel %PARALLEL_JOBS%
)

if errorlevel 1 (
    echo Build failed!
    goto :error
)

echo Build successful!
echo.

REM Install if requested
if "%INSTALL%"=="true" (
    echo Installing...
    cmake --install . --config %BUILD_TYPE%
    echo Installation complete!
    echo.
)

REM List built executables
echo Built executables:
if exist "bin\%BUILD_TYPE%" (
    for %%f in (bin\%BUILD_TYPE%\*.exe) do (
        echo   %%~nxf
    )
) else if exist "%BUILD_TYPE%" (
    for %%f in (%BUILD_TYPE%\*.exe) do (
        echo   %%~nxf
    )
) else (
    for %%f in (*.exe) do (
        echo   %%~nxf
    )
)

echo.
echo =====================================
echo Build completed successfully!
echo =====================================

REM Quick test
echo Running quick test...
set TEST_EXE=
if exist "bin\%BUILD_TYPE%\simple_inference.exe" set TEST_EXE=bin\%BUILD_TYPE%\simple_inference.exe
if exist "%BUILD_TYPE%\simple_inference.exe" set TEST_EXE=%BUILD_TYPE%\simple_inference.exe
if exist "simple_inference.exe" set TEST_EXE=simple_inference.exe

if defined TEST_EXE (
    "%TEST_EXE%" >nul 2>nul
    if not errorlevel 1 (
        echo √ simple_inference executable works
    ) else (
        echo × simple_inference test failed
    )
)

echo.
echo Next steps:
echo   1. Copy your TorchScript model (.pt file) to this directory
echo   2. Run inference with: %TEST_EXE% model.torchscript.pt image.jpg
echo   3. For batch processing: batch_inference.exe model.torchscript.pt \path\to\images\
echo   4. For benchmarking: benchmark_inference.exe model.torchscript.pt

goto :end

:error
echo Build failed with errors.
exit /b 1

:end
cd ..
endlocal