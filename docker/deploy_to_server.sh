#!/bin/bash
set -eu

# LAMAReg Automated Docker Deployment
# ===================================
# This script automatically:
# 1. Migrat# Check Docker connectivity first
echo "🐳 Docker is working on server - proceeding with build" is confirmed working - proceeding with build to server
# 2. Builds Docker image on server
# 3. Tests the built image
# 4. Provides usage instructions

# Configuration
SERVER_BASE_DIR="/host/cassio/export03/data/enning"
BUILD_DIR="$SERVER_BASE_DIR/lamareg_build"
BACKUP_DIR="$SERVER_BASE_DIR/lamareg_backup"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_LAMAREG="$(dirname "$SCRIPT_DIR")"  # Parent directory of docker folder

echo ""
echo "🚀 LAMAReg Server Deployment"
echo "============================"
echo ""
echo "📍 Server path: $SERVER_DIR"
echo "� Docker registry: $DOCKER_REGISTRY"
echo "🏷️  Image tag: $IMAGE_TAG"
echo "💾 SIF output: $BASE_DIR/singularity/lamareg_latest.sif"
echo ""

# Verify source LAMAReg directory
if [[ ! -f "$HOME_LAMAREG/pyproject.toml" || ! -d "$HOME_LAMAREG/lamareg" ]]; then
    echo "❌ Invalid LAMAReg directory: $HOME_LAMAREG"
    echo "   Required: pyproject.toml and lamareg/ directory"
    exit 1
fi

# Check server accessibility
if [[ ! -d "$SERVER_BASE_DIR" ]]; then
    echo "❌ Server directory not accessible: $SERVER_BASE_DIR"
    echo "   Please ensure the server is mounted"
    exit 1
fi

echo "✅ Pre-flight checks passed"
echo ""

# ================================================
# STEP 1: MIGRATE FILES TO SERVER
# ================================================
echo "📋 STEP 1: Migrating files to server..."
echo "======================================="

# Check for changes
SOURCE_CHANGED=false

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "📁 Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    SOURCE_CHANGED=true
fi

if [[ ! -f "$BUILD_DIR/.last_sync_lamareg" ]]; then
    SOURCE_CHANGED=true
    echo "🔄 First time setup - copying all files"
elif [[ -n "$(find "$HOME_LAMAREG" -name '*.py' -o -name 'Dockerfile*' -o -name '*.toml' -o -name '*.txt' -o -name '*.md' -newer "$BUILD_DIR/.last_sync_lamareg" 2>/dev/null)" ]]; then
    SOURCE_CHANGED=true
    echo "🔄 Source files changed - updating build directory"
else
    echo "✅ Source files up to date - skipping migration"
fi

if $SOURCE_CHANGED; then
    # Create backup if needed
    if [[ -d "$BUILD_DIR" && ! -d "$BACKUP_DIR" ]]; then
        echo "💾 Creating backup..."
        cp -r "$BUILD_DIR" "$BACKUP_DIR"
    fi
    
    echo "📋 Copying files to server..."
    
    # Copy Docker files from docker directory
    cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/"
    cp "$SCRIPT_DIR/.dockerignore" "$BUILD_DIR/"
    cp "$SCRIPT_DIR/build_docker.sh" "$BUILD_DIR/"
    cp "$SCRIPT_DIR/test_docker.sh" "$BUILD_DIR/"
    chmod +x "$BUILD_DIR"/*.sh
    
    # Copy Python package files
    cp "$HOME_LAMAREG/pyproject.toml" "$BUILD_DIR/"
    cp "$HOME_LAMAREG/requirements.txt" "$BUILD_DIR/" 2>/dev/null || true
    cp "$HOME_LAMAREG/setup.py" "$BUILD_DIR/" 2>/dev/null || true
    cp "$HOME_LAMAREG/MANIFEST.in" "$BUILD_DIR/" 2>/dev/null || true
    
    # Copy source code
    echo "   Copying LAMAReg source code..."
    cp -r "$HOME_LAMAREG/lamareg" "$BUILD_DIR/"
    
    # Copy essential files
    cp "$HOME_LAMAREG/README.md" "$BUILD_DIR/" 2>/dev/null || true
    cp "$HOME_LAMAREG/LICENSE" "$BUILD_DIR/" 2>/dev/null || true
    
    # Copy tests if they exist
    if [[ -d "$HOME_LAMAREG/tests" ]]; then
        cp -r "$HOME_LAMAREG/tests" "$BUILD_DIR/"
    fi
    
    # Copy docs (excluding large files)
    if [[ -d "$HOME_LAMAREG/docs" ]]; then
        cp -r "$HOME_LAMAREG/docs" "$BUILD_DIR/"
    fi
    
    # Mark sync time
    touch "$BUILD_DIR/.last_sync_lamareg"
    echo "✅ Files migrated successfully"
else
    echo "✅ Migration skipped - files up to date"
fi

# Verify critical files
echo "🔍 Verifying build setup..."
CRITICAL_FILES=("pyproject.toml" "lamareg/__init__.py" "Dockerfile" "build_docker.sh")
for file in "${CRITICAL_FILES[@]}"; do
    if [[ -f "$BUILD_DIR/$file" ]]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file - MISSING!"
        exit 1
    fi
done

echo ""

# ================================================
# STEP 2: BUILD DOCKER IMAGE
# ================================================
echo "🏗️  STEP 2: Building Docker image..."
echo "===================================="

pushd "$BUILD_DIR" > /dev/null

# Check Docker connectivity first
echo "� Checking Docker connectivity..."
if ! docker info >/dev/null 2>&1; then
    echo "❌ Cannot connect to Docker daemon"
    echo "   Contact your system administrator to:"
    echo "   1. Ensure Docker daemon is running"
    echo "   2. Add your user to docker group"
    echo "   3. Fix Docker socket permissions"
    echo ""
    echo "🧪 Test command: docker run --rm hello-world"
    popd > /dev/null
    exit 1
fi

echo "✅ Docker connectivity verified"
echo "�📦 Building Docker image: lamareg:latest"
echo "📍 Build location: $BUILD_DIR"
echo "⏱️  Expected time: 10-15 minutes"
echo ""

# Build with server optimizations and detailed logging
BUILD_LOG="build_lamareg_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Build logged to: $BUILD_LOG"

# Use the build script which has all server optimizations
./build_docker.sh
BUILD_EXIT_CODE=$?

popd > /dev/null

if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "❌ Docker build failed (exit code: $BUILD_EXIT_CODE)"
    echo "   Check the build output above for errors"
    exit $BUILD_EXIT_CODE
fi

echo ""
echo "✅ Docker image built successfully!"
echo ""

# Skip Docker testing - proceed directly to Singularity option
echo "� Docker image ready: lamareg:latest"
echo "📊 Image size: $(docker images lamareg:latest --format 'table {{.Size}}' | tail -1)"

# ================================================
# DEPLOYMENT COMPLETE - READY FOR SINGULARITY
# ================================================
echo "🎉 DOCKER BUILD COMPLETE!"
echo "========================="
echo ""
echo "� Docker Image Information:"
docker images lamareg:latest
echo ""
echo "🧪 Quick Test Commands:"
echo "   # Docker is built - proceed to Singularity for testing"
echo "   ./build_singularity.sh"
echo "   ./test_singularity.sh"
echo ""
echo "📋 Usage Examples:"
echo "   # Basic usage with data mounting"
echo "   docker run --rm -v /path/to/data:/data lamareg:latest python -m lamareg.cli --input /data/input.nii.gz --output /data/output.nii.gz"
echo ""
echo "   # Interactive mode"
echo "   docker run --rm -it -v /path/to/data:/data lamareg:latest bash"
echo ""
echo "   # Background processing"
echo "   docker run -d --name lamareg-job -v /path/to/data:/data lamareg:latest python -m lamareg.cli [options]"
echo ""
echo "💡 Next Steps:"
echo "   1. Test with your data: docker run --rm -v /your/data:/data lamareg:latest python -m lamareg.cli --help"
echo "   2. Build Singularity SIF: ./build_singularity.sh"
echo "   3. Test Singularity: ./test_singularity.sh"
echo "   4. Deploy to HPC clusters using the SIF file"
echo ""
echo "🔄 Build Singularity SIF now?"
echo ""
echo "Options:"
echo "   1) Build Singularity SIF from Docker image"
echo "   2) Skip Singularity build"
echo ""
read -p "Enter your choice (1/2): " -n 1 -r SIF_CHOICE
echo
echo

case $SIF_CHOICE in
    1)
        echo "🏗️  Building Singularity SIF..."
        ./build_singularity.sh
        SIF_BUILD_EXIT_CODE=$?
        
        if [[ $SIF_BUILD_EXIT_CODE -eq 0 ]]; then
            echo "✅ Singularity SIF built successfully!"
            echo "🧪 Testing SIF..."
            ./test_singularity.sh
            TEST_EXIT_CODE=$?
            
            if [[ $TEST_EXIT_CODE -eq 0 ]]; then
                echo "🎉 LAMAReg Docker + Singularity deployment complete!"
            else
                echo "⚠️  SIF built but tests failed - check manually"
            fi
        else
            echo "❌ Singularity SIF build failed"
        fi
        ;;
    2)
        echo "📁 Docker build complete - Singularity skipped"
        echo ""
        echo "🚀 To build SIF later:"
        echo "   ./build_singularity.sh"
        echo "   ./test_singularity.sh"
        ;;
    *)
        echo "❌ Invalid choice - Singularity build skipped"
        echo ""
        echo "🚀 To build SIF later:"
        echo "   ./build_singularity.sh"
        echo "   ./test_singularity.sh"
        ;;
esac
echo ""
echo "📁 Build artifacts saved at: $BUILD_DIR"