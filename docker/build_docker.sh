#!/bin/bash
set -euo pipefail

echo "🏗️  Building LAMAReg Docker Image (Server Optimized)"
echo "=================================================="
echo ""

# Verify server environment
echo "🔍 Verifying server environment..."
echo "📍 Current directory: $PWD"
echo "📁 Expected server location: /host/cassio/export03/data/enning/lamareg_build"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" || ! -d "lamareg" ]]; then
    echo "❌ Error: Not in LAMAReg build directory"
    echo "   Required files: pyproject.toml, lamareg/"
    echo "   Current directory: $(pwd)"
    echo ""
    echo "💡 Make sure you're in the build directory with migrated files"
    exit 1
fi

# Check Docker connectivity (without sudo)
echo "🐳 Checking Docker connectivity..."
if ! docker info >/dev/null 2>&1; then
    echo "❌ Cannot connect to Docker daemon"
    echo "   This usually means:"
    echo "   1. Docker daemon is not running"
    echo "   2. User is not in docker group"
    echo "   3. Docker socket permissions issue"
    echo ""
    echo "💡 Try: docker run --rm hello-world"
    echo "   If that fails, contact your system administrator"
    exit 1
fi

echo "✅ Docker connectivity verified"

# Environment setup for server
export DOCKER_CONTENT_TRUST=0
export BUILDKIT_PROGRESS=plain

# Build configuration with server optimizations
BUILD_LOG="build_lamareg_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Build will be logged to: $BUILD_LOG"

# Build the Docker image with server-specific settings
echo "📦 Building Docker image: lamareg:latest"
echo "⏱️  Expected time: 10-15 minutes"
echo "💾 Using server optimizations for /host/cassio/export03/data/enning"
echo ""

# Build with memory limits and server-specific settings
docker build \
    --memory=8g \
    --memory-swap=12g \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg CUSTOM_TMPDIR="/host/cassio/export03/data/enning" \
    --tag lamareg:latest \
    . 2>&1 | tee "$BUILD_LOG"

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