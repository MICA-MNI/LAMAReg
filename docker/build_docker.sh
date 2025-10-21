#!/bin/bash
set -eu

echo "🏗️  Building LAMAReg Docker Image"
echo "================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" || ! -d "lamareg" ]]; then
    echo "❌ Error: Not in LAMAReg build directory"
    echo "   Required files: pyproject.toml, lamareg/"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image: lamareg:latest"
echo "⏱️  Expected time: 10-15 minutes"
echo ""

docker build -t lamareg:latest .
BUILD_EXIT_CODE=$?

if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo "🎯 Image: lamareg:latest"
    echo ""
    echo "🧪 Test commands:"
    echo "   docker run --rm lamareg:latest lamareg --help"
    echo "   docker run --rm -v \$(pwd)/data:/data lamareg:latest lamareg [options]"
    echo ""
    echo "📊 Image info:"
    docker images lamareg:latest
else
    echo ""
    echo "❌ Docker build failed (exit code: $BUILD_EXIT_CODE)"
    echo "   Check the build output above for errors"
    exit $BUILD_EXIT_CODE
fi