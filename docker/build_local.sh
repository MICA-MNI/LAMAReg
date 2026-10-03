#!/bin/bash
set -eu

# LAMAReg Local Docker Build with Server Transfer
# ==============================================
# This script builds Docker image locally then transfers to server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_LAMAREG="$(dirname "$SCRIPT_DIR")"

echo "🏗️  LAMAReg Local Docker Build"
echo "============================="
echo ""
echo "📍 Source: $HOME_LAMAREG"
echo "🔧 Strategy: Build locally, then transfer image"
echo ""

# Check Docker is working locally
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running locally"
    echo "   Please start Docker Desktop or Docker service"
    exit 1
fi

echo "✅ Docker is running locally"

# Build locally from the parent directory
cd "$HOME_LAMAREG"

echo "📦 Building Docker image locally..."
echo "⏱️  Expected time: 10-15 minutes"

docker build -f docker/Dockerfile -t lamareg:latest .
BUILD_EXIT_CODE=$?

if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
    echo "❌ Local Docker build failed"
    exit $BUILD_EXIT_CODE
fi

echo "✅ Docker image built successfully!"

# Save image to tar file
echo "💾 Saving Docker image to tar file..."
docker save lamareg:latest -o lamareg-docker-image.tar

echo "✅ Image saved as: lamareg-docker-image.tar"
echo ""
echo "📋 Next steps to transfer to server:"
echo "   1. Copy to server: scp lamareg-docker-image.tar user@server:/path/"
echo "   2. Load on server: docker load -i lamareg-docker-image.tar"
echo "   3. Test on server: docker run --rm lamareg:latest lamareg --help"
echo ""
echo "🗂️  Image size:"
ls -lh lamareg-docker-image.tar