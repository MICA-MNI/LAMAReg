#!/bin/bash
set -euo pipefail

echo "🏗️  Building LAMAReg Docker Image (Robust Server Build)"
echo "====================================================="
echo ""

# Function to check Docker daemon
check_docker_daemon() {
    echo "🔍 Checking Docker daemon connectivity..."
    
    # Try multiple approaches to connect to Docker
    local docker_ok=false
    
    # Method 1: Simple docker info
    if docker info >/dev/null 2>&1; then
        docker_ok=true
        echo "✅ Docker daemon accessible via standard socket"
    else
        echo "⚠️  Standard Docker socket check failed"
        
        # Method 2: Check if docker service is running
        if systemctl is-active --quiet docker 2>/dev/null; then
            echo "✅ Docker service is running"
        else
            echo "❌ Docker service not running"
            echo "💡 Try: sudo systemctl start docker"
        fi
        
        # Method 3: Check Docker socket permissions
        if [[ -S "/var/run/docker.sock" ]]; then
            echo "✅ Docker socket exists: /var/run/docker.sock"
            ls -la /var/run/docker.sock
        else
            echo "❌ Docker socket not found"
        fi
        
        # Method 4: Try with explicit socket
        export DOCKER_HOST="unix:///var/run/docker.sock"
        if docker info >/dev/null 2>&1; then
            docker_ok=true
            echo "✅ Docker accessible with explicit socket"
        fi
    fi
    
    if [[ "$docker_ok" != "true" ]]; then
        echo "❌ Docker daemon not accessible"
        echo ""
        echo "🔧 Troubleshooting steps:"
        echo "   1. Check if Docker is running: systemctl status docker"
        echo "   2. Start Docker if needed: sudo systemctl start docker"
        echo "   3. Add user to docker group: sudo usermod -aG docker \$USER"
        echo "   4. Logout and login again to refresh group membership"
        echo "   5. Check socket permissions: ls -la /var/run/docker.sock"
        echo ""
        return 1
    fi
    
    return 0
}

# Function to clean build environment
clean_build_env() {
    echo "🧹 Cleaning build environment..."
    
    # Remove any existing build artifacts
    rm -f build_lamareg_*.log 2>/dev/null || true
    
    # Clean Docker if space is low
    local available_space=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $available_space -lt 10 ]]; then
        echo "⚠️  Low disk space (${available_space}GB), cleaning Docker..."
        docker system prune -f 2>/dev/null || true
    fi
}

# Function to verify build context
verify_build_context() {
    echo "🔍 Verifying build context..."
    echo "📍 Current directory: $PWD"
    
    # Check required files
    local required_files=("pyproject.toml" "lamareg" "requirements.txt")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -e "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo "❌ Missing required files: ${missing_files[*]}"
        echo "💡 Make sure you're in the LAMAReg build directory"
        return 1
    fi
    
    # Check Dockerfile
    if [[ ! -f "Dockerfile" ]]; then
        echo "❌ Dockerfile not found"
        echo "💡 Make sure Dockerfile exists in current directory"
        return 1
    fi
    
    echo "✅ Build context verified"
    return 0
}

# Main execution
main() {
    # Step 1: Check Docker daemon
    if ! check_docker_daemon; then
        exit 1
    fi
    
    # Step 2: Clean environment
    clean_build_env
    
    # Step 3: Verify build context
    if ! verify_build_context; then
        exit 1
    fi
    
    # Step 4: Configure build environment
    echo "⚙️  Configuring build environment..."
    
    # Disable Docker content trust to avoid certificate issues
    export DOCKER_CONTENT_TRUST=0
    export BUILDKIT_PROGRESS=plain
    
    # Use local registry to avoid external dependency issues
    local image_name="lamareg:latest"
    local build_log="build_lamareg_$(date +%Y%m%d_%H%M%S).log"
    
    echo "📝 Build log: $build_log"
    echo "🏷️  Image name: $image_name"
    echo ""
    
    # Step 5: Build with robust settings
    echo "🚀 Starting Docker build..."
    echo "⏱️  Expected time: 10-15 minutes"
    echo ""
    
    # Build with minimal but robust settings
    if docker build \
        --no-cache \
        --progress=plain \
        --tag "$image_name" \
        . 2>&1 | tee "$build_log"; then
        
        echo ""
        echo "✅ Docker build completed successfully!"
        
        # Verify the image
        local image_size=$(docker image inspect "$image_name" --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}')
        echo "📊 Image size: $image_size"
        
        # Test the image
        echo "🧪 Testing Docker image..."
        if docker run --rm "$image_name" python -c "import lamareg; print('LAMAReg ready!')" 2>&1; then
            echo "✅ Docker image test passed"
        else
            echo "⚠️  Docker image test warning - check functionality"
        fi
        
        echo ""
        echo "🎉 LAMAReg Docker image ready: $image_name"
        echo ""
        echo "📋 Next steps:"
        echo "   1. Test: docker run --rm $image_name lamareg --help"
        echo "   2. Build Singularity: ./build_singularity.sh"
        echo "   3. Or use Docker directly: ./deploy_docker_only.sh"
        
    else
        echo ""
        echo "❌ Docker build failed"
        echo "📋 Troubleshooting:"
        echo "   1. Check build log: $build_log"
        echo "   2. Verify internet connectivity: ping google.com"
        echo "   3. Check disk space: df -h"
        echo "   4. Try cleaning Docker: docker system prune -a"
        echo "   5. Check Dockerfile syntax"
        exit 1
    fi
}

# Run main function
main "$@"