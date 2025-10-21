#!/bin/bash
set -euo pipefail

echo "🔍 LAMAReg Docker Server Diagnostics"
echo "===================================="
echo ""

# Server environment check
echo "🖥️  Server Environment:"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Current directory: $(pwd)"
echo "Expected location: /host/cassio/export03/data/enning/lamareg_build"
echo ""

# Check Docker installation
echo "📋 Docker Installation:"
if command -v docker >/dev/null 2>&1; then
    echo "✅ Docker found in PATH"
    docker --version
else
    echo "❌ Docker not found in PATH"
    echo "   Check if Docker is installed or PATH is correct"
fi
echo ""

# Check Docker daemon connectivity (main issue)
echo "🔧 Docker Daemon Connectivity:"
if docker info >/dev/null 2>&1; then
    echo "✅ Docker daemon accessible"
    echo "Docker info summary:"
    docker info 2>/dev/null | grep -E "(Server Version|Storage Driver|Cgroup Driver|Runtimes)" || true
else
    echo "❌ Cannot connect to Docker daemon"
    echo ""
    echo "🔍 Diagnostic information:"
    echo "Docker socket: $(ls -la /var/run/docker.sock 2>/dev/null || echo 'Not accessible')"
    echo "User groups: $(groups)"
    echo "Docker environment variables:"
    env | grep -i docker || echo "None set"
    echo ""
    echo "💡 Common fixes (ask admin to run):"
    echo "   1. systemctl start docker"
    echo "   2. usermod -aG docker $(whoami)"
    echo "   3. chmod 666 /var/run/docker.sock"
fi
echo ""

# Check current directory and files
echo "📁 Current Directory: $(pwd)"
echo "📋 Files in build directory:"
ls -la
echo ""

# Check build context size
echo "📊 Build Context Analysis:"
echo "Total directory size: $(du -sh . | cut -f1)"
echo "File count: $(find . -type f | wc -l)"
echo "Large files (>10MB):"
find . -type f -size +10M -exec ls -lh {} \; 2>/dev/null || echo "No large files found"
echo ""

# Check Dockerfile
echo "🐳 Dockerfile Check:"
if [[ -f "Dockerfile" ]]; then
    echo "✅ Dockerfile exists ($(wc -l < Dockerfile) lines)"
    echo "First 10 lines:"
    head -10 Dockerfile
else
    echo "❌ Dockerfile not found"
fi
echo ""

# Check .dockerignore
echo "🚫 .dockerignore Check:"
if [[ -f ".dockerignore" ]]; then
    echo "✅ .dockerignore exists ($(wc -l < .dockerignore) lines)"
else
    echo "❌ .dockerignore not found"
fi
echo ""

# Check Docker socket
echo "🔌 Docker Socket Check:"
ls -la /var/run/docker.sock 2>/dev/null || echo "❌ Docker socket not accessible"
echo ""

# Check user groups
echo "👤 User Groups:"
groups
echo ""

# Test simple Docker command
echo "🧪 Docker Functionality Test:"
if docker run --rm hello-world >/dev/null 2>&1; then
    echo "✅ Docker working perfectly"
    echo "   Ready to build LAMAReg container"
else
    echo "❌ Docker test failed"
    echo "   Issue: $(docker run --rm hello-world 2>&1 | head -2 || echo 'Cannot run Docker commands')"
fi
echo ""

# Test Docker build capability
echo "🏗️  Docker Build Test:"
if docker build --help >/dev/null 2>&1; then
    echo "✅ Docker build command available"
else
    echo "❌ Docker build command not working"
fi
echo ""

echo "🎯 Ready for LAMAReg build?"
if docker info >/dev/null 2>&1 && docker run --rm hello-world >/dev/null 2>&1; then
    echo "✅ YES - All Docker checks passed"
    echo "   Run: ./deploy_to_server.sh"
else
    echo "❌ NO - Docker issues detected"
    echo "   Contact system administrator to fix Docker daemon"
fi