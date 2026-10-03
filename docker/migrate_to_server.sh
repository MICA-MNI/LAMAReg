#!/bin/bash
set -eu

# LAMAReg Docker Build - Server Migration Script
# ===============================================
# This script migrates LAMAReg code to server for Docker builds
# Copies from: current directory (local development)
# Copies to: /host/cassio/export03/data/enning/lamareg_build (server with space)

# Configuration
SERVER_BASE_DIR="/host/cassio/export03/data/enning"
BUILD_DIR="$SERVER_BASE_DIR/lamareg_build"
BACKUP_DIR="$SERVER_BASE_DIR/lamareg_backup"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_LAMAREG="$(dirname "$SCRIPT_DIR")"  # Parent directory of docker folder

echo "🚚 LAMAReg Docker Build - Server Migration"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: This script copies files TO the server"
echo "   Docker builds will happen ON THE SERVER at: $BUILD_DIR"
echo "   NOT in your home directory!"
echo ""
echo "📍 Server base: $SERVER_BASE_DIR"
echo "📁 Build directory (ON SERVER): $BUILD_DIR"
echo "🏠 Source code (LOCAL): $HOME_LAMAREG"
echo ""
echo "📦 Build Strategy:"
echo "   Single-stage build with Python 3.11 + neuroimaging dependencies"
echo "   Expected build time: 10-15 minutes"
echo ""

# Verify source LAMAReg directory exists
if [[ ! -d "$HOME_LAMAREG" ]]; then
    echo "❌ Source LAMAReg directory not found: $HOME_LAMAREG"
    echo "   Please run this script from your LAMAReg directory"
    exit 1
fi

# Check if we have the required files
if [[ ! -f "$HOME_LAMAREG/pyproject.toml" ]]; then
    echo "❌ pyproject.toml not found. Are you in the LAMAReg directory?"
    exit 1
fi

# Check if server directory is accessible
if [[ ! -d "$SERVER_BASE_DIR" ]]; then
    echo "❌ Server directory not accessible: $SERVER_BASE_DIR"
    echo "   Please ensure the server is mounted or the path is correct"
    exit 1
fi

echo "✅ Server directory accessible: $SERVER_BASE_DIR"

# Create backup if source exists
if [[ -d "$BUILD_DIR" && ! -d "$BACKUP_DIR" ]]; then
    echo "💾 Creating backup of existing build directory..."
    cp -r "$BUILD_DIR" "$BACKUP_DIR"
    echo "✅ Backup created: $BACKUP_DIR"
elif [[ -d "$BACKUP_DIR" ]]; then
    echo "✅ Backup already exists: $BACKUP_DIR"
fi

# Check for source file changes
SOURCE_CHANGED=false

# Create build directory if it doesn't exist
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "📁 Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    SOURCE_CHANGED=true
fi

if [[ ! -f "$BUILD_DIR/.last_sync_lamareg" ]]; then
    SOURCE_CHANGED=true
    echo "🔄 First time setup - copying all source files"
elif [[ -n "$(find "$HOME_LAMAREG" -name '*.py' -o -name 'Dockerfile*' -o -name '*.toml' -o -name '*.txt' -o -name '*.md' -newer "$BUILD_DIR/.last_sync_lamareg" 2>/dev/null)" ]]; then
    SOURCE_CHANGED=true
    echo "🔄 Source files changed - updating build directory"
else
    echo "✅ Source files up to date"
fi

if $SOURCE_CHANGED; then
    echo "📋 Copying LAMAReg files to server..."
    
    # Copy Docker files from docker directory
    echo "   Copying Dockerfile..."
    cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/"
    
    echo "   Copying .dockerignore..."
    cp "$SCRIPT_DIR/.dockerignore" "$BUILD_DIR/"
    
    echo "   Copying build scripts..."
    cp "$SCRIPT_DIR/build_docker.sh" "$BUILD_DIR/"
    cp "$SCRIPT_DIR/test_docker.sh" "$BUILD_DIR/"
    cp "$SCRIPT_DIR/build_singularity.sh" "$BUILD_DIR/"
    cp "$SCRIPT_DIR/test_singularity.sh" "$BUILD_DIR/"
    
    # Copy Python package files
    echo "   Copying Python configuration files..."
    cp "$HOME_LAMAREG/pyproject.toml" "$BUILD_DIR/"
    cp "$HOME_LAMAREG/requirements.txt" "$BUILD_DIR/" 2>/dev/null || true
    cp "$HOME_LAMAREG/setup.py" "$BUILD_DIR/" 2>/dev/null || true
    cp "$HOME_LAMAREG/MANIFEST.in" "$BUILD_DIR/" 2>/dev/null || true
    
    # Copy source code
    echo "   Copying LAMAReg source code..."
    cp -r "$HOME_LAMAREG/lamareg" "$BUILD_DIR/"
    
    # Copy essential files
    echo "   Copying documentation and license..."
    cp "$HOME_LAMAREG/README.md" "$BUILD_DIR/" 2>/dev/null || true
    cp "$HOME_LAMAREG/LICENSE" "$BUILD_DIR/" 2>/dev/null || true
    
    # Copy example data if small enough (skip large .nii files)
    if [[ -d "$HOME_LAMAREG/example_data" ]]; then
        echo "   Copying example data (excluding large files)..."
        mkdir -p "$BUILD_DIR/example_data"
        find "$HOME_LAMAREG/example_data" -type f ! -name "*.nii.gz" ! -name "*.nii" -exec cp {} "$BUILD_DIR/example_data/" \; 2>/dev/null || true
    fi
    
    # Copy tests if they exist
    if [[ -d "$HOME_LAMAREG/tests" ]]; then
        echo "   Copying tests..."
        cp -r "$HOME_LAMAREG/tests" "$BUILD_DIR/" 2>/dev/null || true
    fi
    
    # Copy docs directory (excluding large files)
    if [[ -d "$HOME_LAMAREG/docs" ]]; then
        echo "   Copying documentation..."
        cp -r "$HOME_LAMAREG/docs" "$BUILD_DIR/" 2>/dev/null || true
    fi
    
    echo "   ✅ LAMAReg files copied to server"
    
    # Mark sync time
    touch "$BUILD_DIR/.last_sync_lamareg"
    echo "✅ LAMAReg build files copied to server"
else
    echo "✅ Source files already up to date"
fi

# Verify critical files are in place
echo "🔍 Verifying LAMAReg build setup..."
CRITICAL_FILES=(
    "pyproject.toml"
    "lamareg/__init__.py"
    "lamareg/cli.py"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [[ -f "$BUILD_DIR/$file" ]]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file - MISSING!"
        echo "Error: Critical file missing. Migration failed."
        exit 1
    fi
done

# Check if Dockerfile exists, create if missing
if [[ ! -f "$BUILD_DIR/Dockerfile" ]]; then
    echo "📝 Creating Dockerfile..."
    cat > "$BUILD_DIR/Dockerfile" << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget \
    curl \
    unzip \
    git \
    build-essential \
    # ANTsPy dependencies
    cmake \
    # Neuroimaging dependencies
    libfreetype6-dev \
    pkg-config \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install LAMAReg in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 lamareg && \
    chown -R lamareg:lamareg /app
USER lamareg

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/home/lamareg/.local/bin:$PATH

# Default command
CMD ["lamareg", "--help"]
EOF
    echo "   ✅ Dockerfile created"
fi

echo ""
echo "🎯 Migration Complete!"
echo "======================"
echo "Build directory: $BUILD_DIR"
echo "Strategy: Single-stage Python build"
echo ""
echo "⚠️  CRITICAL: DO NOT build from ~/LAMAReg (home directory)!"
echo "   Builds MUST happen from: $BUILD_DIR"
echo ""
echo "📋 Build Commands:"
echo "   cd $BUILD_DIR"
echo "   docker build -t lamareg:latest ."
echo "   ⏱️  Expected time: 10-15 minutes"
echo ""
echo "🧪 Test Commands:"
echo "   docker run --rm lamareg:latest lamareg --help"
echo "   docker run --rm -v /path/to/data:/data lamareg:latest lamareg [options]"
echo ""

# Interactive build option
echo "🤔 Would you like to build the Docker image now?"
echo ""
echo "Options:"
echo "   1) Build Docker image (10-15 min)"
echo "   2) Just migrate files (no build)"
echo ""
read -p "Enter your choice (1/2): " -n 1 -r BUILD_CHOICE
echo
echo

case $BUILD_CHOICE in
    1)
        echo "🏗️  Building LAMAReg Docker image..."
        echo "📍 Build location: $BUILD_DIR"
        echo "⏱️  Expected time: 10-15 minutes"
        echo ""
        pushd "$BUILD_DIR"
        docker build -t lamareg:latest .
        BUILD_EXIT_CODE=$?
        popd
        
        if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
            echo "✅ Docker image built successfully!"
            echo "🎯 lamareg:latest is ready to use!"
            echo ""
            echo "🧪 Test it with:"
            echo "   docker run --rm lamareg:latest lamareg --help"
        else
            echo "❌ Docker build failed (exit code: $BUILD_EXIT_CODE)"
        fi
        ;;
    2)
        echo "📁 Files migrated only - no build started"
        echo ""
        echo "🚀 To build later:"
        echo "   cd $BUILD_DIR"
        echo "   docker build -t lamareg:latest ."
        ;;
    *)
        echo "❌ Invalid choice. Files migrated only."
        echo ""
        echo "🚀 To build later:"
        echo "   cd $BUILD_DIR"
        echo "   docker build -t lamareg:latest ."
        ;;
esac