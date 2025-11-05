#!/bin/bash
# Docker build and run script for Blind Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="blind-assistant"
IMAGE_TAG="latest"
CONTAINER_NAME="blind-assistant"

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_info "Docker is installed"
}

check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not available"
        print_warning "GPU acceleration will not work"
        return 1
    fi
    print_info "NVIDIA Docker runtime is available"
    return 0
}

build_image() {
    print_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    DOCKER_BUILDKIT=1 docker build \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -f docker/Dockerfile \
        .
    
    print_info "Build complete"
}

run_container() {
    local camera_ip=${1:-192.168.1.100}
    local show_display=${2:-true}
    
    print_info "Starting container: ${CONTAINER_NAME}"
    print_info "Camera IP: ${camera_ip}"
    print_info "Show display: ${show_display}"
    
    # Allow X11 connections
    xhost +local:docker 2>/dev/null || print_warning "Could not configure X11"
    
    # Check if GPU is available
    local gpu_flag=""
    if check_nvidia_docker; then
        gpu_flag="--gpus all"
    else
        print_warning "Running without GPU acceleration"
    fi
    
    # Run container
    docker run --rm -it \
        ${gpu_flag} \
        --name ${CONTAINER_NAME} \
        --network host \
        -e DISPLAY=${DISPLAY:-:0} \
        -e CAMERA_IP=${camera_ip} \
        -e SHOW_DISPLAY=${show_display} \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/test_output:/app/test_output \
        ${IMAGE_NAME}:${IMAGE_TAG}
}

run_compose() {
    print_info "Starting with Docker Compose"
    
    # Allow X11 connections
    xhost +local:docker 2>/dev/null || print_warning "Could not configure X11"
    
    docker-compose -f docker/docker-compose.yml up
}

stop_container() {
    print_info "Stopping container: ${CONTAINER_NAME}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || print_warning "Container not running"
}

clean_up() {
    print_info "Cleaning up Docker resources"
    docker-compose -f docker/docker-compose.yml down 2>/dev/null || true
    docker container prune -f
    docker image prune -f
    print_info "Cleanup complete"
}

test_image() {
    print_info "Testing Docker image"
    
    # Test CUDA
    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Test imports
    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
        python -c "from src.services.detection.object_detector import ObjectDetector; print('Imports OK')"
    
    print_info "Tests passed"
}

show_help() {
    cat << EOF
Blind Assistant Docker Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build                   Build Docker image
    run [IP] [DISPLAY]      Run container (default IP: 192.168.1.100, display: true)
    compose                 Run with docker-compose
    stop                    Stop running container
    test                    Test Docker image
    clean                   Clean up Docker resources
    help                    Show this help message

Examples:
    $0 build                                    # Build image
    $0 run                                      # Run with defaults
    $0 run 192.168.1.150                       # Run with custom IP
    $0 run 192.168.1.150 false                 # Run headless mode
    $0 compose                                  # Run with compose
    $0 stop                                     # Stop container
    $0 test                                     # Test image
    $0 clean                                    # Clean up

Requirements:
    - Docker installed
    - NVIDIA Docker runtime (for GPU support)
    - IP Webcam app on Android phone

EOF
}

# Main script
main() {
    check_docker
    
    case "${1:-help}" in
        build)
            build_image
            ;;
        run)
            run_container "$2" "$3"
            ;;
        compose)
            run_compose
            ;;
        stop)
            stop_container
            ;;
        test)
            test_image
            ;;
        clean)
            clean_up
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
