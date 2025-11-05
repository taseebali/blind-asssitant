@echo off
REM Docker build and run script for Blind Assistant (Windows)

setlocal enabledelayedexpansion

set IMAGE_NAME=blind-assistant
set IMAGE_TAG=latest
set CONTAINER_NAME=blind-assistant

REM Colors (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM Check arguments
set COMMAND=%1

if "%COMMAND%"=="" (
    call :show_help
    exit /b 0
)

if "%COMMAND%"=="build" (
    call :build_image
    exit /b 0
)

if "%COMMAND%"=="run" (
    set CAMERA_IP=%2
    if "%CAMERA_IP%"=="" set CAMERA_IP=192.168.1.100
    set SHOW_DISPLAY=%3
    if "%SHOW_DISPLAY%"=="" set SHOW_DISPLAY=true
    call :run_container
    exit /b 0
)

if "%COMMAND%"=="compose" (
    call :run_compose
    exit /b 0
)

if "%COMMAND%"=="stop" (
    call :stop_container
    exit /b 0
)

if "%COMMAND%"=="test" (
    call :test_image
    exit /b 0
)

if "%COMMAND%"=="clean" (
    call :clean_up
    exit /b 0
)

if "%COMMAND%"=="help" (
    call :show_help
    exit /b 0
)

echo %RED%[ERROR]%NC% Unknown command: %COMMAND%
call :show_help
exit /b 1

REM Functions
:print_info
echo %GREEN%[INFO]%NC% %~1
exit /b 0

:print_warning
echo %YELLOW%[WARN]%NC% %~1
exit /b 0

:print_error
echo %RED%[ERROR]%NC% %~1
exit /b 0

:check_docker
docker version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not installed or not running"
    exit /b 1
)
call :print_info "Docker is available"
exit /b 0

:build_image
call :print_info "Building Docker image: %IMAGE_NAME%:%IMAGE_TAG%"

docker build -t %IMAGE_NAME%:%IMAGE_TAG% -f docker\Dockerfile .

if errorlevel 1 (
    call :print_error "Build failed"
    exit /b 1
)

call :print_info "Build complete"
exit /b 0

:run_container
call :check_docker
if errorlevel 1 exit /b 1

call :print_info "Starting container: %CONTAINER_NAME%"
call :print_info "Camera IP: %CAMERA_IP%"
call :print_info "Show display: %SHOW_DISPLAY%"

REM Check for GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :print_warning "GPU not available, running CPU-only mode"
    set GPU_FLAG=
) else (
    call :print_info "GPU available"
    set GPU_FLAG=--gpus all
)

REM Run container
docker run --rm -it ^
    %GPU_FLAG% ^
    --name %CONTAINER_NAME% ^
    --network host ^
    -e CAMERA_IP=%CAMERA_IP% ^
    -e SHOW_DISPLAY=%SHOW_DISPLAY% ^
    -v "%cd%\models":/app/models ^
    -v "%cd%\test_output":/app/test_output ^
    %IMAGE_NAME%:%IMAGE_TAG%

exit /b 0

:run_compose
call :print_info "Starting with Docker Compose"

docker-compose -f docker\docker-compose.yml up

exit /b 0

:stop_container
call :print_info "Stopping container: %CONTAINER_NAME%"

docker stop %CONTAINER_NAME% >nul 2>&1

if errorlevel 1 (
    call :print_warning "Container not running"
) else (
    call :print_info "Container stopped"
)

exit /b 0

:test_image
call :print_info "Testing Docker image"

REM Test CUDA
docker run --rm %IMAGE_NAME%:%IMAGE_TAG% python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

REM Test imports
docker run --rm %IMAGE_NAME%:%IMAGE_TAG% python -c "from src.services.detection.object_detector import ObjectDetector; print('Imports OK')"

call :print_info "Tests passed"
exit /b 0

:clean_up
call :print_info "Cleaning up Docker resources"

docker-compose -f docker\docker-compose.yml down >nul 2>&1
docker container prune -f
docker image prune -f

call :print_info "Cleanup complete"
exit /b 0

:show_help
echo.
echo Blind Assistant Docker Management Script (Windows)
echo.
echo Usage: docker-build.bat [COMMAND] [OPTIONS]
echo.
echo Commands:
echo     build                   Build Docker image
echo     run [IP] [DISPLAY]      Run container (default IP: 192.168.1.100, display: true)
echo     compose                 Run with docker-compose
echo     stop                    Stop running container
echo     test                    Test Docker image
echo     clean                   Clean up Docker resources
echo     help                    Show this help message
echo.
echo Examples:
echo     docker-build.bat build                                # Build image
echo     docker-build.bat run                                  # Run with defaults
echo     docker-build.bat run 192.168.1.150                   # Run with custom IP
echo     docker-build.bat run 192.168.1.150 false             # Run headless mode
echo     docker-build.bat compose                              # Run with compose
echo     docker-build.bat stop                                 # Stop container
echo     docker-build.bat test                                 # Test image
echo     docker-build.bat clean                                # Clean up
echo.
echo Requirements:
echo     - Docker Desktop for Windows
echo     - NVIDIA Docker runtime (for GPU support)
echo     - IP Webcam app on Android phone
echo.
exit /b 0
