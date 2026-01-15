#!/bin/bash

# FastTracker CUDA Build Script
# Usage: ./scripts/build.sh [clean|rebuild|run]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
BIN_DIR="$BUILD_DIR/bin"

echo "=== FastTracker CUDA Build Script ==="
echo "Project directory: $PROJECT_DIR"
echo "Build directory: $BUILD_DIR"

case "$1" in
    clean)
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        echo "Clean complete!"
        ;;
    rebuild)
        echo "Rebuilding..."
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake ..
        make -j$(nproc)
        echo "Rebuild complete!"
        ;;
    run)
        if [ -f "$BIN_DIR/FastTrackerCUDA" ]; then
            echo "Running FastTrackerCUDA..."
            cd "$BIN_DIR"
            ./FastTrackerCUDA
        else
            echo "Executable not found. Please build first."
            exit 1
        fi
        ;;
    *)
        echo "Building FastTrackerCUDA..."
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake ..
        make -j$(nproc)
        echo "Build complete!"
        echo "Run with: ./scripts/build.sh run"
        ;;
esac
